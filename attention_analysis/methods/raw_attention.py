"""
Raw Attention Extraction.

Directly extracts and aggregates attention weights without any processing.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from attention_analysis.methods.base import AttributionMethod
from attention_analysis.utils.hooks import HookManager
from attention_analysis.utils.model_utils import get_patch_grid_size


class RawAttention(AttributionMethod):
    """
    Raw attention weight extraction.

    Extracts attention weights from specified layer(s) and aggregates them.
    This is the simplest form of attention visualization but is NOT faithful
    to model behavior - use for quick sanity checks only.

    Args:
        layer: Which layer to extract from. Options:
            - int: Specific layer index
            - "last": Last layer (default)
            - "all": Average across all layers
        head_fusion: How to aggregate heads. Options:
            - "mean": Average across heads (default)
            - "max": Maximum across heads
            - "min": Minimum across heads
            - int: Specific head index
        query_token: Which token's attention to visualize. Options:
            - "cls": CLS token (index 0)
            - "mean": Average of all tokens
            - int: Specific token index
    """

    def __init__(
        self,
        layer: int | str = "last",
        head_fusion: str | int = "mean",
        query_token: str | int = "mean",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layer = layer
        self.head_fusion = head_fusion
        self.query_token = query_token

    @torch.no_grad()
    def attribute(
        self,
        model: nn.Module,
        image: torch.Tensor,
        target_class: int | None = None,
    ) -> torch.Tensor:
        """
        Extract raw attention weights.

        Args:
            model: The ViT model.
            image: Input image tensor.
            target_class: Ignored (raw attention is not class-specific).

        Returns:
            Attention map of shape (H, W) in patch grid.
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

        # Select layer(s)
        if self.layer == "last":
            attn = attentions[-1]  # (B, H, N, N)
        elif self.layer == "all":
            attn = torch.stack(attentions).mean(dim=0)
        elif isinstance(self.layer, int):
            attn = attentions[self.layer]
        else:
            raise ValueError(f"Unknown layer specification: {self.layer}")

        # Move to device if needed
        attn = attn.to(device)

        # Aggregate heads
        if self.head_fusion == "mean":
            attn = attn.mean(dim=1)  # (B, N, N)
        elif self.head_fusion == "max":
            attn = attn.max(dim=1)[0]
        elif self.head_fusion == "min":
            attn = attn.min(dim=1)[0]
        elif isinstance(self.head_fusion, int):
            attn = attn[:, self.head_fusion]
        else:
            raise ValueError(f"Unknown head_fusion: {self.head_fusion}")

        # Select query token
        if self.query_token == "cls":
            # CLS token attending to all patches
            attn_map = attn[:, 0, 1:]  # (B, N-1)
        elif self.query_token == "mean":
            # Average attention from all tokens
            attn_map = attn.mean(dim=1)  # (B, N)
            # If CLS exists, we might want to exclude it
            if attn_map.shape[-1] > 196:  # Has CLS
                attn_map = attn_map[:, 1:]
        elif isinstance(self.query_token, int):
            attn_map = attn[:, self.query_token]
            if attn_map.shape[-1] > 196:  # Has CLS
                attn_map = attn_map[:, 1:]
        else:
            raise ValueError(f"Unknown query_token: {self.query_token}")

        # Take first image in batch
        attn_map = attn_map[0]

        # Reshape to grid
        attn_map = self._reshape_to_grid(attn_map, model)

        # Normalize
        attn_map = self._normalize_attribution(attn_map)

        return attn_map

    def requires_gradient(self) -> bool:
        return False

    def is_class_specific(self) -> bool:
        return False
