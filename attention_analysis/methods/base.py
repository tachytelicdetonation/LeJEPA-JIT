"""
Base class for attribution methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from attention_analysis.utils.model_utils import (
    get_patch_grid_size,
)


class AttributionMethod(ABC):
    """
    Abstract base class for attention attribution methods.

    All attribution methods should inherit from this class and implement
    the `attribute` method.

    Attributes:
        normalize: Whether to normalize output to [0, 1].
        include_cls: Whether to include CLS token in attribution.
    """

    def __init__(
        self,
        normalize: bool = True,
        include_cls: bool = False,
        **kwargs,
    ):
        """
        Initialize the attribution method.

        Args:
            normalize: Whether to normalize attribution to [0, 1].
            include_cls: Whether to include CLS token in output.
            **kwargs: Method-specific parameters.
        """
        self.normalize = normalize
        self.include_cls = include_cls
        self.config = kwargs

    @abstractmethod
    def attribute(
        self,
        model: nn.Module,
        image: torch.Tensor,
        target_class: int | None = None,
    ) -> torch.Tensor:
        """
        Compute attribution map for an image.

        Args:
            model: The Vision Transformer model.
            image: Input image tensor, shape (C, H, W) or (B, C, H, W).
            target_class: Target class for class-specific methods.

        Returns:
            Attribution map, typically shape (H_patches, W_patches).
        """
        pass

    def __call__(
        self,
        model: nn.Module,
        image: torch.Tensor,
        target_class: int | None = None,
    ) -> torch.Tensor:
        """Make the method callable."""
        return self.attribute(model, image, target_class)

    def _prepare_input(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        """
        Prepare input image for processing.

        Args:
            image: Input image.

        Returns:
            (batched_image, was_unbatched)
        """
        if image.dim() == 3:
            return image.unsqueeze(0), True
        return image, False

    def _get_target_class(
        self,
        model: nn.Module,
        image: torch.Tensor,
        target_class: int | None,
    ) -> int:
        """
        Get target class for attribution.

        Args:
            model: The model.
            image: Input image (batched).
            target_class: Specified target class or None.

        Returns:
            Target class index.
        """
        if target_class is not None:
            return target_class

        # Get prediction
        with torch.no_grad():
            output = self._get_model_output(model, image)
            if isinstance(output, tuple):
                output = output[0]
            return output.argmax(dim=-1)[0].item()

    def _get_model_output(
        self,
        model: nn.Module,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get model output logits.

        Handles different model types.
        """
        # LeJEPA models - need to use encoder + probe
        if hasattr(model, "encoder"):
            # This assumes a linear probe is attached for classification
            embeddings = model.encoder(image)
            if hasattr(model, "probe"):
                return model.probe(embeddings)
            # If no probe, return embeddings (won't work for class-specific)
            return embeddings

        # Standard classification models
        output = model(image)
        if hasattr(output, "logits"):
            return output.logits
        return output

    def _reshape_to_grid(
        self,
        attribution: torch.Tensor,
        model: nn.Module,
        img_size: int | None = None,
    ) -> torch.Tensor:
        """
        Reshape flat attribution to spatial grid.

        Args:
            attribution: Flat attribution, shape (N,) or (N_with_cls,).
            model: The model (to get grid size).
            img_size: Optional image size override.

        Returns:
            Attribution grid, shape (H, W).
        """
        grid_h, grid_w = get_patch_grid_size(model, img_size)
        expected_patches = grid_h * grid_w

        # Handle CLS token
        if len(attribution) == expected_patches + 1:
            if self.include_cls:
                # Keep CLS in first position, but return patch grid
                # CLS token stored separately if needed
                patch_attr = attribution[1:].view(grid_h, grid_w)
                return patch_attr
            else:
                attribution = attribution[1:]  # Remove CLS

        return attribution.view(grid_h, grid_w)

    def _normalize_attribution(
        self,
        attribution: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize attribution to [0, 1]."""
        if not self.normalize:
            return attribution

        vmin, vmax = attribution.min(), attribution.max()
        if vmax - vmin > 1e-8:
            return (attribution - vmin) / (vmax - vmin)
        return torch.zeros_like(attribution)

    @classmethod
    def name(cls) -> str:
        """Return the method name."""
        return cls.__name__

    @classmethod
    def description(cls) -> str:
        """Return a description of the method."""
        return cls.__doc__ or "No description available."

    def requires_gradient(self) -> bool:
        """Whether this method requires gradient computation."""
        return False

    def is_class_specific(self) -> bool:
        """Whether this method produces class-specific attributions."""
        return False

    def get_config(self) -> dict[str, Any]:
        """Return the method configuration."""
        return {
            "normalize": self.normalize,
            "include_cls": self.include_cls,
            **self.config,
        }
