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

    def __init__(self, knots: int = 17):
        """
        Args:
            knots: Number of quadrature points for characteristic function
        """
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        Compute SIGReg loss.

        Args:
            proj: (*, N, D) projected embeddings where N is batch/samples, D is dim

        Returns:
            Scalar loss value
        """
        # Random projections for slicing high-dim to 1D
        A = torch.randn(proj.size(-1), 256, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))

        # Project and compute characteristic function
        # Ensure buffers are on same device as proj
        t = self.t.to(proj.device)
        phi = self.phi.to(proj.device)
        weights = self.weights.to(proj.device)

        x_t = (proj @ A).unsqueeze(-1) * t
        err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ weights) * proj.size(-2)
        return statistic.mean()


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
        lambda_sigreg: float = 0.02,
        knots: int = 17,
        multivariate: bool = True,
        num_frequencies: int = 256,
        sigma: float = 1.0,
    ):
        """
        Args:
            lambda_sigreg: Weight for SIGReg loss (1-lambda for invariance)
            knots: Number of quadrature points for univariate SIGReg
            multivariate: If True, use multivariate SIGReg (default)
            num_frequencies: Number of frequency vectors for multivariate mode
            sigma: Frequency scale for multivariate mode
        """
        super().__init__()
        self.lambda_sigreg = lambda_sigreg
        self.multivariate = multivariate

        if multivariate:
            self.sigreg = MultivariateSIGReg(
                num_frequencies=num_frequencies,
                sigma=sigma,
            )
        else:
            self.sigreg = SIGReg(knots=knots)

    def forward(self, proj: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Compute combined LeJEPA loss.

        Args:
            proj: (V, B, proj_dim) projected embeddings from multiple views
                  V = num_views, B = batch_size

        Returns:
            Dictionary with loss components
        """
        # SIGReg loss on projections
        sigreg_loss = self.sigreg(proj)

        # Invariance loss: views should cluster around their mean
        # proj shape: (V, B, D) -> mean over views, then MSE
        inv_loss = (proj.mean(0) - proj).square().mean()

        # Combined loss
        loss = self.lambda_sigreg * sigreg_loss + (1 - self.lambda_sigreg) * inv_loss

        return {
            "loss": loss,
            "sigreg_loss": sigreg_loss,
            "invariance_loss": inv_loss,
        }
