"""
SIGReg Loss implementation for LeJEPA.

SIGReg (Signature Regularization) is based on characteristic function matching.
Instead of directly comparing embeddings, it compares the empirical characteristic
functions of the embedding distributions.

The total LeJEPA loss combines:
1. SIGReg loss: Distributional regularization preventing collapse
2. Invariance loss: Encourages embeddings to be similar across augmented views

References:
- LeJEPA: Learning Joint-Embeddings with Prediction Agents
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGRegLoss(nn.Module):
    """
    SIGReg (Signature Regularization) loss using characteristic function matching.

    This loss compares the empirical characteristic function of the embedding
    distribution to a target distribution (typically standard Gaussian) using
    Gaussian quadrature.

    Key properties:
    - No stop-gradients needed
    - Prevents representation collapse
    - Based on Epps-Pulley test statistic
    """

    def __init__(
        self,
        num_knots: int = 17,
        max_t: float = 3.0,
        target_std: float = 1.0,
    ):
        """
        Args:
            num_knots: Number of quadrature points for characteristic function
            max_t: Maximum value for quadrature range [0, max_t]
            target_std: Standard deviation of target distribution
        """
        super().__init__()
        self.num_knots = num_knots
        self.max_t = max_t
        self.target_std = target_std

        # Create quadrature knots (evenly spaced in [0, max_t])
        knots = torch.linspace(0, max_t, num_knots)
        self.register_buffer("knots", knots)

        # Gaussian weights for quadrature (weight by exp(-t^2/2))
        weights = torch.exp(-0.5 * knots**2)
        weights = weights / weights.sum()
        self.register_buffer("weights", weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute SIGReg loss.

        Args:
            x: (B, D) embedding tensor

        Returns:
            Scalar loss value
        """
        B, D = x.shape

        # Standardize embeddings (center and scale)
        x_centered = x - x.mean(dim=0, keepdim=True)
        x_std = x_centered.std(dim=0, keepdim=True).clamp(min=1e-6)
        x_normalized = x_centered / x_std

        # Generate random projection directions for slicing
        # Project high-dimensional embeddings to 1D for characteristic function
        num_projections = min(D, 64)  # Limit projections for efficiency
        projections = torch.randn(D, num_projections, device=x.device)
        projections = F.normalize(projections, dim=0)

        # Project embeddings
        x_proj = x_normalized @ projections  # (B, num_projections)

        # Compute empirical characteristic function
        # φ(t) = E[exp(itX)] = E[cos(tX)] + i*E[sin(tX)]
        # For real X, we use the real part: E[cos(tX)]

        # Expand for broadcasting: (B, num_projections, 1) * (num_knots,)
        x_proj_expanded = x_proj.unsqueeze(-1)  # (B, num_projections, 1)
        knots = self.knots.view(1, 1, -1)  # (1, 1, num_knots)

        # Compute cos(t * x) for all combinations
        cos_tx = torch.cos(x_proj_expanded * knots)  # (B, num_projections, num_knots)
        sin_tx = torch.sin(x_proj_expanded * knots)

        # Empirical characteristic function (average over batch)
        phi_real = cos_tx.mean(dim=0)  # (num_projections, num_knots)
        phi_imag = sin_tx.mean(dim=0)

        # Target characteristic function for standard Gaussian: exp(-t^2/2)
        target_phi_real = torch.exp(-0.5 * self.knots**2 * self.target_std**2)
        target_phi_imag = torch.zeros_like(target_phi_real)

        # Squared difference weighted by quadrature weights
        diff_real = (phi_real - target_phi_real) ** 2
        diff_imag = phi_imag**2  # Should be 0 for symmetric distribution

        # Weighted sum over knots
        loss = (diff_real * self.weights).sum(dim=-1) + (diff_imag * self.weights).sum(dim=-1)

        # Average over projections
        loss = loss.mean()

        return loss


class InvarianceLoss(nn.Module):
    """
    Invariance loss encouraging embeddings to be similar across augmented views.

    For each sample, we want the embeddings of different augmented views to
    cluster around their mean.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, num_views: int = 2) -> torch.Tensor:
        """
        Compute invariance loss.

        Args:
            x: (B*V, D) embeddings from multiple views concatenated
            num_views: Number of views per sample

        Returns:
            Scalar loss value
        """
        B_total, D = x.shape
        B = B_total // num_views

        # Reshape to (B, V, D)
        x = x.view(B, num_views, D)

        # Mean embedding per sample
        mean_emb = x.mean(dim=1, keepdim=True)  # (B, 1, D)

        # Squared distance from mean
        loss = ((x - mean_emb) ** 2).mean()

        return loss


class LeJEPALoss(nn.Module):
    """
    Combined LeJEPA loss: SIGReg + Invariance.

    Total loss = λ * SIGReg + (1-λ) * Invariance

    No stop-gradients or momentum encoders needed.
    """

    def __init__(
        self,
        lambda_sigreg: float = 0.5,
        num_knots: int = 17,
        max_t: float = 3.0,
        num_views: int = 2,
    ):
        """
        Args:
            lambda_sigreg: Weight for SIGReg loss (1-lambda for invariance)
            num_knots: Number of quadrature points for SIGReg
            max_t: Maximum quadrature range
            num_views: Number of augmented views per sample
        """
        super().__init__()
        self.lambda_sigreg = lambda_sigreg
        self.num_views = num_views

        self.sigreg = SIGRegLoss(num_knots=num_knots, max_t=max_t)
        self.invariance = InvarianceLoss()

    def forward(
        self,
        projections: torch.Tensor,
        embeddings: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute combined LeJEPA loss.

        Args:
            projections: (B*V, proj_dim) projected embeddings
            embeddings: (B*V, embed_dim) encoder embeddings (optional, for logging)

        Returns:
            Dictionary with:
                - loss: Combined loss
                - sigreg_loss: SIGReg component
                - invariance_loss: Invariance component
        """
        # SIGReg loss on projections
        sigreg_loss = self.sigreg(projections)

        # Invariance loss on projections
        invariance_loss = self.invariance(projections, self.num_views)

        # Combined loss
        loss = self.lambda_sigreg * sigreg_loss + (1 - self.lambda_sigreg) * invariance_loss

        return {
            "loss": loss,
            "sigreg_loss": sigreg_loss,
            "invariance_loss": invariance_loss,
        }
