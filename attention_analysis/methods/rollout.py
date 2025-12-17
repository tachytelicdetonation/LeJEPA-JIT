"""
Attention Rollout.

Tracks information flow by recursively multiplying attention matrices across layers.

Reference:
    "Quantifying Attention Flow in Transformers" (Abnar & Zuidema, ACL 2020)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from attention_analysis.methods.base import AttributionMethod
from attention_analysis.utils.hooks import HookManager
from attention_analysis.utils.model_utils import get_patch_grid_size


class AttentionRollout(AttributionMethod):
    """
    Attention Rollout: Accumulates attention flow across transformer layers.

    Computes how much each input token contributes to each output token
    by recursively multiplying attention matrices, accounting for residual
    connections.

    Formula:
        A_hat = 0.5 * A + 0.5 * I  (residual connection)
        A_hat = A_hat / sum(A_hat, dim=-1)  (renormalize)
        Rollout = A_hat_L @ A_hat_{L-1} @ ... @ A_hat_1

    Args:
        head_fusion: How to aggregate heads before rollout. Options:
            - "mean": Average across heads (default)
            - "max": Maximum across heads
            - "min": Minimum across heads
        residual_weight: Weight for residual connection (default 0.5).
            A_hat = (1 - w) * A + w * I
        discard_ratio: Fraction of lowest attention values to zero out.
            Helps reduce noise but loses information.
        start_layer: First layer to include (0-indexed).
        end_layer: Last layer to include (None = last).
    """

    def __init__(
        self,
        head_fusion: str = "mean",
        residual_weight: float = 0.5,
        discard_ratio: float = 0.0,
        start_layer: int = 0,
        end_layer: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.head_fusion = head_fusion
        self.residual_weight = residual_weight
        self.discard_ratio = discard_ratio
        self.start_layer = start_layer
        self.end_layer = end_layer

    @torch.no_grad()
    def attribute(
        self,
        model: nn.Module,
        image: torch.Tensor,
        target_class: int | None = None,
    ) -> torch.Tensor:
        """
        Compute attention rollout.

        Args:
            model: The ViT model.
            image: Input image tensor.
            target_class: Ignored (rollout is not class-specific).

        Returns:
            Rollout attention map of shape (H, W) in patch grid.
        """
        image, was_unbatched = self._prepare_input(image)
        device = next(model.parameters()).device
        image = image.to(device)

        # Use HookManager for clean extraction
        with HookManager(model) as hooks:
            hooks.enable_attention_capture()
            hooks.register_attention_hooks()

            # Forward pass
            model.eval()
            _ = model.encoder(image) if hasattr(model, "encoder") else model(image)

            # Get attention maps
            attentions = hooks.get_attention_maps()

        if not attentions:
            grid_h, grid_w = get_patch_grid_size(model)
            return torch.zeros(grid_h, grid_w)

        # Select layer range
        end_layer = self.end_layer if self.end_layer is not None else len(attentions)
        attentions = attentions[self.start_layer : end_layer]

        if not attentions:
            grid_h, grid_w = get_patch_grid_size(model)
            return torch.zeros(grid_h, grid_w)

        B, H, N, _ = attentions[0].shape

        # Initialize rollout with identity matrix
        rollout = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)

        for attn in attentions:
            attn = attn.to(device)

            # Fuse heads
            if self.head_fusion == "mean":
                attn_fused = attn.mean(dim=1)
            elif self.head_fusion == "max":
                attn_fused = attn.max(dim=1)[0]
            elif self.head_fusion == "min":
                attn_fused = attn.min(dim=1)[0]
            else:
                raise ValueError(f"Unknown head_fusion: {self.head_fusion}")

            # Optional: discard low attention values
            if self.discard_ratio > 0:
                attn_fused = self._discard_low_attention(attn_fused)

            # Add residual connection
            identity = torch.eye(N, device=device).unsqueeze(0)
            attn_fused = (
                1 - self.residual_weight
            ) * attn_fused + self.residual_weight * identity

            # Renormalize rows to sum to 1
            attn_fused = attn_fused / attn_fused.sum(dim=-1, keepdim=True).clamp(
                min=1e-8
            )

            # Accumulate rollout
            rollout = torch.matmul(attn_fused, rollout)

        # Extract attention map
        # Check if model uses CLS token
        has_cls = self._model_has_cls(model)

        if has_cls:
            # CLS token (index 0) attending to patches
            attn_map = rollout[:, 0, 1:]  # (B, N-1)
        else:
            # Mean pooling: average attention from all tokens
            attn_map = rollout.mean(dim=1)  # (B, N)

        # Take first image
        attn_map = attn_map[0]

        # Reshape to grid
        attn_map = self._reshape_to_grid(attn_map, model)

        # Normalize
        attn_map = self._normalize_attribution(attn_map)

        return attn_map

    def _discard_low_attention(self, attn: torch.Tensor) -> torch.Tensor:
        """Zero out the lowest attention values."""
        B, N, _ = attn.shape
        flat = attn.view(B, -1)

        # Find threshold for discard_ratio
        k = int(flat.shape[-1] * self.discard_ratio)
        if k > 0:
            threshold = flat.kthvalue(k, dim=-1, keepdim=True)[0]
            mask = flat >= threshold.unsqueeze(-1).expand_as(flat)
            flat = flat * mask.float()

        return flat.view(B, N, N)

    def _model_has_cls(self, model: nn.Module) -> bool:
        """Check if model uses CLS token."""
        if hasattr(model, "encoder"):
            if hasattr(model.encoder, "pool"):
                return model.encoder.pool == "cls"
            if hasattr(model.encoder, "cls_token"):
                return model.encoder.cls_token is not None
        if hasattr(model, "cls_token"):
            return model.cls_token is not None
        return False

    def requires_gradient(self) -> bool:
        return False

    def is_class_specific(self) -> bool:
        return False


class AttentionRolloutPerHead(AttributionMethod):
    """
    Per-head Attention Rollout.

    Computes rollout separately for each attention head, useful for
    analyzing head specialization.

    Returns:
        Tensor of shape (num_heads, H, W) with per-head rollout maps.
    """

    def __init__(
        self,
        residual_weight: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.residual_weight = residual_weight

    @torch.no_grad()
    def attribute(
        self,
        model: nn.Module,
        image: torch.Tensor,
        target_class: int | None = None,
    ) -> torch.Tensor:
        """
        Compute per-head attention rollout.

        Returns:
            Tensor of shape (num_heads, H, W).
        """
        image, was_unbatched = self._prepare_input(image)
        device = next(model.parameters()).device
        image = image.to(device)

        with HookManager(model) as hooks:
            hooks.enable_attention_capture()
            hooks.register_attention_hooks()

            model.eval()
            _ = model.encoder(image) if hasattr(model, "encoder") else model(image)

            attentions = hooks.get_attention_maps()

        if not attentions:
            grid_h, grid_w = get_patch_grid_size(model)
            return torch.zeros(6, grid_h, grid_w)  # Assume 6 heads

        B, num_heads, N, _ = attentions[0].shape

        # Compute rollout per head
        head_rollouts = []

        for h in range(num_heads):
            # Initialize rollout
            rollout = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)

            for attn in attentions:
                attn = attn.to(device)
                attn_h = attn[:, h]  # (B, N, N)

                # Add residual
                identity = torch.eye(N, device=device).unsqueeze(0)
                attn_h = (
                    1 - self.residual_weight
                ) * attn_h + self.residual_weight * identity

                # Renormalize
                attn_h = attn_h / attn_h.sum(dim=-1, keepdim=True).clamp(min=1e-8)

                # Accumulate
                rollout = torch.matmul(attn_h, rollout)

            # Extract map (mean pooling)
            attn_map = rollout.mean(dim=1)[0]  # (N,)

            # Reshape
            grid_h, grid_w = get_patch_grid_size(model)
            if len(attn_map) == grid_h * grid_w + 1:
                attn_map = attn_map[1:]  # Remove CLS
            attn_map = attn_map.view(grid_h, grid_w)

            head_rollouts.append(attn_map)

        result = torch.stack(head_rollouts)  # (num_heads, H, W)

        if self.normalize:
            # Normalize each head independently
            for i in range(result.shape[0]):
                result[i] = self._normalize_attribution(result[i])

        return result
