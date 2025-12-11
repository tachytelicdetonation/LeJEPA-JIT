"""
SIGReg Loss implementation for LeJEPA.

SIGReg (Signature Regularization) is based on characteristic function matching.
Instead of directly comparing embeddings, it compares the empirical characteristic
functions of the embedding distributions.

Matches the reference implementation from LeJEPA MINIMAL.md exactly.
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
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class LeJEPALoss(nn.Module):
    """
    Combined LeJEPA loss: SIGReg + Invariance.

    Total loss = λ * SIGReg + (1-λ) * Invariance

    No stop-gradients or momentum encoders needed.
    """

    def __init__(
        self,
        lambda_sigreg: float = 0.02,
        knots: int = 17,
    ):
        """
        Args:
            lambda_sigreg: Weight for SIGReg loss (1-lambda for invariance)
            knots: Number of quadrature points for SIGReg
        """
        super().__init__()
        self.lambda_sigreg = lambda_sigreg
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
