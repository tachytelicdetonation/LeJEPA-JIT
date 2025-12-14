"""
SIGReg Loss implementation for LeJEPA.

SIGReg (Signature Regularization) is based on characteristic function matching.
Instead of directly comparing embeddings, it compares the empirical characteristic
functions of the embedding distributions.

Supports both univariate (sliced) and multivariate modes:
- Univariate: Projects to 256 random 1D directions, computes 1D characteristic functions
- Multivariate: Computes full D-dimensional characteristic function directly
"""

import torch
import torch.nn as nn


def _dist_is_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _dist_world_size() -> int:
    if _dist_is_initialized():
        return torch.distributed.get_world_size()
    return 1


def _dist_all_reduce_avg_(x: torch.Tensor) -> torch.Tensor:
    if not _dist_is_initialized():
        return x
    torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
    x.div_(float(_dist_world_size()))
    return x


def _dist_all_reduce_max_(x: torch.Tensor) -> torch.Tensor:
    if not _dist_is_initialized():
        return x
    torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.MAX)
    return x


class SIGReg(nn.Module):
    """
    SIGReg (Signature Regularization) loss using characteristic function matching.

    This loss compares the empirical characteristic function of the embedding
    distribution to a standard Gaussian using trapezoidal integration.

    Key properties:
    - No stop-gradients needed
    - Prevents representation collapse
    - Based on Epps-Pulley test statistic
    """

    def __init__(
        self,
        knots: int = 17,
        t_max: float = 3.0,
        num_slices: int = 256,
        clip_value: float | None = None,
    ):
        """
        Args:
            knots: Number of quadrature points for characteristic function
        """
        super().__init__()
        if knots < 3 or knots % 2 != 1:
            # Paper uses 17; odd helps with some quadrature rules (even though we use trapezoid).
            raise ValueError(f"knots must be an odd integer >= 3, got {knots}")
        if num_slices <= 0:
            raise ValueError(f"num_slices must be > 0, got {num_slices}")

        self.num_slices = int(num_slices)
        self.clip_value = clip_value

        t = torch.linspace(0, float(t_max), knots, dtype=torch.float32)
        dt = float(t_max) / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))

        self._generator: torch.Generator | None = None
        self._generator_device: torch.device | None = None

    def _get_generator(self, device: torch.device, seed: int) -> torch.Generator:
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator_device = device
        self._generator.manual_seed(int(seed))
        return self._generator

    @torch.no_grad()
    def _sample_slices(self, dim: int, device: torch.device) -> torch.Tensor:
        # Deterministic seed across ranks (paper Algorithm 1 uses global_step)
        seed_t = self.global_step.clone()
        _dist_all_reduce_max_(seed_t)
        g = self._get_generator(device, int(seed_t.item()))

        A = torch.randn((dim, self.num_slices), device=device, generator=g)
        A.div_(A.norm(p=2, dim=0).clamp_min_(1e-12))

        self.global_step.add_(1)
        return A

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        Compute SIGReg loss.

        Args:
            proj: (*, N, D) projected embeddings where N is batch/samples, D is dim

        Returns:
            Scalar loss value
        """
        # proj: (*, N, D) where N is samples (batch), D is feature dim
        if proj.ndim < 2:
            raise ValueError(f"proj must have shape (*, N, D); got {proj.shape}")
        if proj.size(-2) <= 1:
            # Avoid degenerate N scaling; still return something finite
            return proj.new_tensor(0.0)

        A = self._sample_slices(dim=proj.size(-1), device=proj.device)

        # Project: (*, N, K)
        x = proj @ A

        # Ensure buffers are on same device as proj
        t = self.t.to(proj.device)
        phi = self.phi.to(proj.device)
        weights = self.weights.to(proj.device)

        # Epps–Pulley characteristic function matching over t in [0, t_max]
        # x_t: (*, N, K, knots)
        x_t = x.unsqueeze(-1) * t
        cos_mean = x_t.cos().mean(dim=-3)  # (*, K, knots)
        sin_mean = x_t.sin().mean(dim=-3)  # (*, K, knots)

        # DDP: synchronize means (paper's ref implementation averages across ranks)
        _dist_all_reduce_avg_(cos_mean)
        _dist_all_reduce_avg_(sin_mean)

        err = (cos_mean - phi).square() + sin_mean.square()
        stats = (err @ weights) * float(proj.size(-2) * _dist_world_size())  # (*, K)
        if self.clip_value is not None:
            stats = stats.masked_fill(stats < float(self.clip_value), 0.0)
        return stats.mean()


class MultivariateSIGReg(nn.Module):
    """
    Multivariate SIGReg loss using full D-dimensional characteristic function.

    Instead of slicing to 1D, this directly evaluates the multivariate
    characteristic function at random frequency vectors.

    Based on the multivariate Epps-Pulley test statistic.
    """

    def __init__(self, num_frequencies: int = 256, sigma: float = 1.0):
        """
        Args:
            num_frequencies: Number of random frequency vectors to sample (M)
            sigma: Standard deviation for frequency sampling
        """
        super().__init__()
        self.num_frequencies = num_frequencies
        self.sigma = sigma

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        Compute multivariate SIGReg loss.

        Args:
            proj: (*, N, D) projected embeddings where N is batch/samples, D is dim

        Returns:
            Scalar loss value
        """
        D = proj.size(-1)
        N = proj.size(-2)

        # Sample frequency vectors: (M, D)
        t = torch.randn(self.num_frequencies, D, device=proj.device) * self.sigma

        # Compute phase: (*, N, M)
        phase = proj @ t.T

        # Empirical characteristic function: (*, M)
        phi_real = phase.cos().mean(dim=-2)
        phi_imag = phase.sin().mean(dim=-2)

        # Theoretical Gaussian characteristic function: (M,)
        t_norm_sq = t.pow(2).sum(dim=-1)
        phi_gauss = torch.exp(-t_norm_sq / 2)

        # Squared error with Gaussian weighting
        err_real = (phi_real - phi_gauss).pow(2)
        err_imag = phi_imag.pow(2)

        # Weight by Gaussian dampening (same as phi_gauss)
        weighted_err = (err_real + err_imag) * phi_gauss

        # Average over frequencies and scale by N
        statistic = weighted_err.mean(dim=-1) * N

        return statistic.mean()


class LeJEPALoss(nn.Module):
    """
    Combined LeJEPA loss: SIGReg + Invariance.

    Total loss = λ * SIGReg + (1-λ) * Invariance

    Supports both univariate (sliced) and multivariate SIGReg modes.
    No stop-gradients or momentum encoders needed.
    """

    def __init__(
        self,
        lambda_sigreg: float = 0.05,
        knots: int = 17,
        t_max: float = 3.0,
        num_slices: int = 1024,
        clip_value: float | None = None,
        multivariate: bool = False,
        num_frequencies: int = 256,
        sigma: float = 1.0,
        num_global_views: int = 2,
    ):
        """
        Args:
            lambda_sigreg: Weight for SIGReg loss (1-lambda for invariance)
            knots: Number of quadrature points for univariate SIGReg (paper uses 17)
            t_max: Upper integration bound for Epps–Pulley (paper uses 3)
            num_slices: Number of random 1D slices for univariate SIGReg (paper uses ~1024)
            clip_value: Optional threshold to zero-out tiny slice statistics
            multivariate: If True, use multivariate SIGReg (CF variant)
            num_frequencies: Number of frequency vectors for multivariate mode
            sigma: Frequency scale for multivariate mode
            num_global_views: Number of global views used to form prediction targets (paper uses 2)
        """
        super().__init__()
        self.lambda_sigreg = lambda_sigreg
        self.multivariate = multivariate
        self.num_global_views = int(num_global_views)

        if multivariate:
            self.sigreg = MultivariateSIGReg(
                num_frequencies=num_frequencies,
                sigma=sigma,
            )
        else:
            self.sigreg = SIGReg(
                knots=knots,
                t_max=t_max,
                num_slices=num_slices,
                clip_value=clip_value,
            )

    def forward(
        self,
        proj: torch.Tensor,
        *,
        global_proj: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute combined LeJEPA loss.

        Args:
            proj: (V, B, proj_dim) projected embeddings from multiple views
                  V = num_views, B = batch_size
            global_proj: Optional (Vg, B, proj_dim) projections for global views only.
                         If provided, prediction targets are formed from this tensor
                         (useful for teacher/SWA targets).

        Returns:
            Dictionary with loss components
        """
        # SIGReg loss on projections
        sigreg_loss = self.sigreg(proj)

        # Prediction loss (paper): all views predict the global-view centroid
        # proj shape: (V, B, D); centroid shape: (B, D)
        if global_proj is None:
            v_global = min(self.num_global_views, int(proj.size(0)))
            if v_global <= 0:
                v_global = int(proj.size(0))
            centers = proj[:v_global].mean(dim=0)
        else:
            centers = global_proj.mean(dim=0)

        pred_loss = (centers - proj).square().mean()

        # Combined loss
        loss = self.lambda_sigreg * sigreg_loss + (1 - self.lambda_sigreg) * pred_loss

        return {
            "loss": loss,
            "sigreg_loss": sigreg_loss,
            "prediction_loss": pred_loss,
        }
