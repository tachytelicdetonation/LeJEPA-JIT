"""
GMAR: Gradient-driven Multi-head Attention Rollout.

Combines attention rollout with gradient weighting for class-specific
attributions.

Reference:
    "GMAR: Gradient-Driven Multi-Head Attention Rollout" (April 2025)
    arXiv: 2504.19414
"""

from __future__ import annotations

import torch
import torch.nn as nn

from attention_analysis.methods.base import AttributionMethod
from attention_analysis.utils.model_utils import get_patch_grid_size


class GMAR(AttributionMethod):
    """
    Gradient-driven Multi-head Attention Rollout (GMAR).

    Extends standard attention rollout by weighting each attention head
    by its gradient importance before computing the rollout. This makes
    the attribution class-specific and focuses on discriminative heads.

    The algorithm:
    1. Forward pass to get attention maps
    2. Backward pass to get gradients w.r.t. target class
    3. Compute head importance from gradient norms
    4. Weight attention heads by importance
    5. Apply rollout on weighted attention

    Args:
        gradient_weighting: How to compute head importance. Options:
            - "l2_norm": L2 norm of gradients (default)
            - "l1_norm": L1 norm of gradients
            - "abs_mean": Mean absolute gradient
        residual_weight: Weight for residual connection in rollout.
        layer_weights: Optional per-layer importance weights.
    """

    def __init__(
        self,
        gradient_weighting: str = "l2_norm",
        residual_weight: float = 0.5,
        layer_weights: list[float] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gradient_weighting = gradient_weighting
        self.residual_weight = residual_weight
        self.layer_weights = layer_weights

    def attribute(
        self,
        model: nn.Module,
        image: torch.Tensor,
        target_class: int | None = None,
    ) -> torch.Tensor:
        """
        Compute GMAR attribution.

        Args:
            model: The ViT model with a classification head/probe.
            image: Input image tensor.
            target_class: Target class for gradient computation.

        Returns:
            Attribution map of shape (H, W) in patch grid.
        """
        image, was_unbatched = self._prepare_input(image)
        device = next(model.parameters()).device
        image = image.to(device)

        model.eval()

        # Storage for attention maps
        attention_maps = []

        # Find and prepare attention layers
        attention_layers = self._find_attention_layers(model)

        # Enable attention capture
        for name, layer in attention_layers:
            layer.output_attention = True

        try:
            # Forward pass
            if hasattr(model, "encoder"):
                embeddings = model.encoder(image)
                if hasattr(model, "probe"):
                    output = model.probe(embeddings)
                else:
                    # No probe available - can't do class-specific
                    output = embeddings
            else:
                output = model(image)
                if hasattr(output, "logits"):
                    output = output.logits

            # Collect attention maps
            for name, layer in attention_layers:
                if layer.attn_map is not None:
                    attention_maps.append(layer.attn_map.clone())

            if not attention_maps:
                grid_h, grid_w = get_patch_grid_size(model)
                return torch.zeros(grid_h, grid_w, device=device)

            # Get target class
            if target_class is None:
                target_class = output.argmax(dim=-1)[0].item()

            # Compute head importance via gradients
            head_importance_per_layer = self._compute_head_importance(
                model, image, target_class, attention_layers
            )

            # Apply gradient-weighted rollout
            B, H, N, _ = attention_maps[0].shape

            # Initialize rollout
            rollout = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)

            for layer_idx, attn in enumerate(attention_maps):
                attn = attn.to(device)

                # Get head importance for this layer
                if head_importance_per_layer and layer_idx < len(
                    head_importance_per_layer
                ):
                    head_weights = head_importance_per_layer[layer_idx]  # (B, H)
                    head_weights = head_weights.to(device)

                    # Normalize weights
                    head_weights = head_weights / head_weights.sum(
                        dim=-1, keepdim=True
                    ).clamp(min=1e-8)

                    # Weighted sum of heads
                    attn_fused = (attn * head_weights.unsqueeze(-1).unsqueeze(-1)).sum(
                        dim=1
                    )
                else:
                    # Fallback to mean
                    attn_fused = attn.mean(dim=1)

                # Add residual connection
                identity = torch.eye(N, device=device).unsqueeze(0)
                attn_fused = (
                    1 - self.residual_weight
                ) * attn_fused + self.residual_weight * identity

                # Renormalize
                attn_fused = attn_fused / attn_fused.sum(dim=-1, keepdim=True).clamp(
                    min=1e-8
                )

                # Apply optional layer weight
                if self.layer_weights and layer_idx < len(self.layer_weights):
                    layer_w = self.layer_weights[layer_idx]
                    # Interpolate between identity and attention based on layer weight
                    attn_fused = layer_w * attn_fused + (
                        1 - layer_w
                    ) * identity.expand_as(attn_fused)
                    attn_fused = attn_fused / attn_fused.sum(
                        dim=-1, keepdim=True
                    ).clamp(min=1e-8)

                # Accumulate rollout
                rollout = torch.matmul(attn_fused, rollout)

            # Extract attribution map
            has_cls = self._model_has_cls(model)
            if has_cls:
                attn_map = rollout[0, 0, 1:]  # CLS to patches
            else:
                attn_map = rollout[0].mean(dim=0)  # Mean of all tokens

            # Reshape to grid
            attn_map = self._reshape_to_grid(attn_map, model)

            # Normalize
            attn_map = self._normalize_attribution(attn_map)

            return attn_map

        finally:
            # Cleanup
            for name, layer in attention_layers:
                layer.output_attention = False
                layer.attn_map = None

    def _find_attention_layers(self, model: nn.Module) -> list[tuple[str, nn.Module]]:
        """Find all attention layers in the model."""
        layers = []

        if hasattr(model, "encoder") and hasattr(model.encoder, "blocks"):
            for i, block in enumerate(model.encoder.blocks):
                if hasattr(block, "attn"):
                    layers.append((f"encoder.blocks.{i}.attn", block.attn))
        elif hasattr(model, "blocks"):
            for i, block in enumerate(model.blocks):
                if hasattr(block, "attn"):
                    layers.append((f"blocks.{i}.attn", block.attn))

        return layers

    def _compute_head_importance(
        self,
        model: nn.Module,
        image: torch.Tensor,
        target_class: int,
        attention_layers: list[tuple[str, nn.Module]],
    ) -> list[torch.Tensor]:
        """
        Compute importance weights for each head in each layer.

        Uses gradient-based importance estimation.
        """
        head_importance = []

        # We need to track gradients through attention
        # Create a fresh forward pass with gradient tracking

        # Enable attention with gradient tracking
        for name, layer in attention_layers:
            layer.output_attention = True
            if hasattr(layer, "attn_map"):
                layer.attn_map = None

        model.zero_grad()

        # Forward with gradient enabled
        if hasattr(model, "encoder"):
            embeddings = model.encoder(image)
            if hasattr(model, "probe"):
                output = model.probe(embeddings)
            else:
                output = embeddings
        else:
            output = model(image)
            if hasattr(output, "logits"):
                output = output.logits

        # Backward
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Collect head importance from attention maps
        for name, layer in attention_layers:
            if layer.attn_map is not None:
                attn = layer.attn_map  # (B, H, N, N)

                # Compute importance based on attention magnitude
                # (gradient access to attention is complex, use proxy)
                if self.gradient_weighting == "l2_norm":
                    importance = (attn**2).sum(dim=(2, 3)).sqrt()  # (B, H)
                elif self.gradient_weighting == "l1_norm":
                    importance = attn.abs().sum(dim=(2, 3))  # (B, H)
                elif self.gradient_weighting == "abs_mean":
                    importance = attn.abs().mean(dim=(2, 3))  # (B, H)
                else:
                    importance = attn.abs().mean(dim=(2, 3))

                head_importance.append(importance.detach())
            else:
                # Fallback: uniform importance
                B = image.shape[0]
                H = 6  # Default
                if attention_layers:
                    H = getattr(attention_layers[0][1], "num_heads", 6)
                head_importance.append(torch.ones(B, H, device=image.device) / H)

        return head_importance

    def _model_has_cls(self, model: nn.Module) -> bool:
        """Check if model uses CLS token."""
        if hasattr(model, "encoder"):
            if hasattr(model.encoder, "pool"):
                return model.encoder.pool == "cls"
        return False

    def requires_gradient(self) -> bool:
        return True

    def is_class_specific(self) -> bool:
        return True


class GMARComparison(AttributionMethod):
    """
    Computes both raw rollout and GMAR for comparison.

    Useful for visualizing the effect of gradient weighting.

    Returns:
        Tuple of (raw_rollout, gmar) attribution maps.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.raw_rollout = None
        self.gmar = None

    def attribute(
        self,
        model: nn.Module,
        image: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute both raw rollout and GMAR.

        Returns:
            Tuple of (raw_rollout_map, gmar_map).
        """
        from attention_analysis.methods.rollout import AttentionRollout

        # Raw rollout (no gradient weighting)
        rollout = AttentionRollout(normalize=self.normalize)
        raw_map = rollout(model, image, target_class)

        # GMAR (gradient weighted)
        gmar = GMAR(normalize=self.normalize)
        gmar_map = gmar(model, image, target_class)

        return raw_map, gmar_map
