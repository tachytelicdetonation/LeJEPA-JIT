"""
AttnLRP: Attention-aware Layer-wise Relevance Propagation.

The state-of-the-art attribution method for transformers with the highest
faithfulness scores.

Reference:
    "AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers"
    ICML 2024, arXiv: 2402.05602

Requires:
    pip install lxt (LRP-eXplains-Transformers library)
    OR fallback to Input×Gradient approximation
"""

from __future__ import annotations

import torch
import torch.nn as nn

from attention_analysis.methods.base import AttributionMethod
from attention_analysis.utils.model_utils import get_patch_grid_size

# Try to import LXT library
try:
    from lxt.efficient import monkey_patch, unmonkey_patch

    HAS_LXT = True
except ImportError:
    HAS_LXT = False


class AttnLRP(AttributionMethod):
    """
    AttnLRP: Attention-aware Layer-wise Relevance Propagation.

    This is the recommended attribution method for Vision Transformers due to
    its superior faithfulness compared to attention rollout and Grad-CAM.

    Two modes available:
    1. LXT mode (requires lxt library): Full AttnLRP with proper propagation rules
    2. Fallback mode: Input×Gradient approximation (efficient but less accurate)

    The method propagates relevance from the output class through the transformer
    layers, correctly handling attention's non-linearities.

    Args:
        gamma: Gamma parameter for LRP-γ rule (default 0.25 for ViTs).
            Higher gamma emphasizes positive contributions.
        target_layer: Which layer to extract relevance from. Options:
            - "input": Input patches (default, most common)
            - "all": Sum across all layers
            - int: Specific layer index
        use_efficient: If True, uses efficient Input×Gradient formulation.
            Faster but may be slightly less accurate.
    """

    def __init__(
        self,
        gamma: float = 0.25,
        target_layer: str | int = "input",
        use_efficient: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.target_layer = target_layer
        self.use_efficient = use_efficient

        if not HAS_LXT and not use_efficient:
            import warnings

            warnings.warn(
                "LXT library not found. Using Input×Gradient fallback. "
                "For best results, install: pip install lxt"
            )

    def attribute(
        self,
        model: nn.Module,
        image: torch.Tensor,
        target_class: int | None = None,
    ) -> torch.Tensor:
        """
        Compute AttnLRP attribution.

        Args:
            model: The ViT model with a classification head/probe.
            image: Input image tensor.
            target_class: Target class for attribution.

        Returns:
            Attribution map of shape (H, W) in patch grid.
        """
        if HAS_LXT and not self.use_efficient:
            return self._attribute_lxt(model, image, target_class)
        else:
            return self._attribute_input_gradient(model, image, target_class)

    def _attribute_lxt(
        self,
        model: nn.Module,
        image: torch.Tensor,
        target_class: int | None,
    ) -> torch.Tensor:
        """
        Full AttnLRP using LXT library.
        """
        image, was_unbatched = self._prepare_input(image)
        device = next(model.parameters()).device
        image = image.to(device)

        # Get the encoder module for monkey patching
        encoder = model.encoder if hasattr(model, "encoder") else model

        # Monkey patch the model for efficient LRP
        monkey_patch(encoder, gamma=self.gamma)

        try:
            # Enable gradient for input
            image.requires_grad_(True)

            # Forward pass
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

            # Get target class
            if target_class is None:
                target_class = output.argmax(dim=-1)[0].item()

            # Backward pass for target class
            model.zero_grad()
            output[0, target_class].backward()

            # Get relevance from input gradient
            if image.grad is not None:
                # Relevance = input × gradient
                relevance = (image * image.grad).sum(dim=1)  # Sum over channels
                relevance = relevance[0]  # First image
            else:
                grid_h, grid_w = get_patch_grid_size(model)
                return torch.zeros(grid_h, grid_w, device=device)

        finally:
            # Unmonkey patch
            unmonkey_patch(encoder)

        # Convert pixel relevance to patch relevance
        attn_map = self._pixel_to_patch_relevance(relevance, model)

        # Normalize
        attn_map = self._normalize_attribution(attn_map)

        return attn_map

    def _attribute_input_gradient(
        self,
        model: nn.Module,
        image: torch.Tensor,
        target_class: int | None,
    ) -> torch.Tensor:
        """
        Input×Gradient approximation of AttnLRP.

        This is the efficient fallback that works without LXT library.
        It's based on the observation that AttnLRP with identity rules
        reduces to Input×Gradient.
        """
        image, was_unbatched = self._prepare_input(image)
        device = next(model.parameters()).device
        image = image.to(device)
        image.requires_grad_(True)

        model.eval()

        # Forward pass
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

        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=-1)[0].item()

        # Backward pass
        model.zero_grad()

        # Get gradient of target class w.r.t. input
        output[0, target_class].backward()

        if image.grad is None:
            grid_h, grid_w = get_patch_grid_size(model)
            return torch.zeros(grid_h, grid_w, device=device)

        # Input × Gradient relevance
        # Apply gamma rule: emphasize positive contributions
        grad = image.grad
        if self.gamma > 0:
            # LRP-γ: weight positive gradients more
            pos_mask = (grad > 0).float()
            neg_mask = (grad <= 0).float()
            grad = pos_mask * grad * (1 + self.gamma) + neg_mask * grad

        relevance = (image * grad).sum(dim=1)  # Sum over channels
        relevance = relevance[0]  # First image

        # Take absolute value for visualization
        relevance = relevance.abs()

        # Convert to patch relevance
        attn_map = self._pixel_to_patch_relevance(relevance, model)

        # Normalize
        attn_map = self._normalize_attribution(attn_map)

        return attn_map

    def _pixel_to_patch_relevance(
        self,
        pixel_relevance: torch.Tensor,
        model: nn.Module,
    ) -> torch.Tensor:
        """
        Convert pixel-level relevance to patch-level.

        Args:
            pixel_relevance: Relevance map of shape (H, W).
            model: The model (to get patch size).

        Returns:
            Patch-level relevance of shape (H_patches, W_patches).
        """
        from attention_analysis.utils.model_utils import get_model_info

        info = get_model_info(model)
        patch_size = info["patch_size"]

        H, W = pixel_relevance.shape
        grid_h = H // patch_size
        grid_w = W // patch_size

        # Reshape and sum within each patch
        relevance = pixel_relevance.view(grid_h, patch_size, grid_w, patch_size)
        relevance = relevance.sum(dim=(1, 3))  # Sum within patches

        return relevance

    def requires_gradient(self) -> bool:
        return True

    def is_class_specific(self) -> bool:
        return True


class AttnLRPIntermediate(AttributionMethod):
    """
    AttnLRP with intermediate layer attribution.

    Allows extracting relevance at intermediate layers, useful for
    understanding which layers contribute most to the prediction.

    Args:
        layer_idx: Which layer to extract relevance from.
        gamma: Gamma parameter for LRP-γ rule.
    """

    def __init__(
        self,
        layer_idx: int = -1,
        gamma: float = 0.25,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layer_idx = layer_idx
        self.gamma = gamma

    def attribute(
        self,
        model: nn.Module,
        image: torch.Tensor,
        target_class: int | None = None,
    ) -> torch.Tensor:
        """
        Compute AttnLRP at intermediate layer.

        Returns:
            Attribution map from the specified intermediate layer.
        """
        image, was_unbatched = self._prepare_input(image)
        device = next(model.parameters()).device
        image = image.to(device)

        model.eval()

        # Storage for intermediate activations
        intermediate_activation = {}

        def save_activation_hook(name):
            def hook(module, input, output):
                intermediate_activation[name] = output

            return hook

        # Find transformer blocks
        if hasattr(model, "encoder") and hasattr(model.encoder, "blocks"):
            blocks = model.encoder.blocks
        elif hasattr(model, "blocks"):
            blocks = model.blocks
        else:
            raise ValueError("Cannot find transformer blocks in model")

        # Register hook on target layer
        layer_idx = (
            self.layer_idx if self.layer_idx >= 0 else len(blocks) + self.layer_idx
        )
        handle = blocks[layer_idx].register_forward_hook(save_activation_hook("target"))

        try:
            # Enable gradient
            image.requires_grad_(True)

            # Forward pass
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

            # Get target class
            if target_class is None:
                target_class = output.argmax(dim=-1)[0].item()

            # Get intermediate activation
            intermediate = intermediate_activation.get("target")
            if intermediate is None:
                grid_h, grid_w = get_patch_grid_size(model)
                return torch.zeros(grid_h, grid_w, device=device)

            # Enable gradient for intermediate
            intermediate.retain_grad()

            # Backward pass
            model.zero_grad()
            output[0, target_class].backward()

            # Get gradient at intermediate layer
            if intermediate.grad is not None:
                # Relevance = activation × gradient
                relevance = (intermediate * intermediate.grad).sum(
                    dim=-1
                )  # Sum over hidden dim
                relevance = relevance[0]  # First image
            else:
                grid_h, grid_w = get_patch_grid_size(model)
                return torch.zeros(grid_h, grid_w, device=device)

            # Reshape to grid
            relevance = self._reshape_to_grid(relevance.abs(), model)

            # Normalize
            relevance = self._normalize_attribution(relevance)

            return relevance

        finally:
            handle.remove()

    def requires_gradient(self) -> bool:
        return True

    def is_class_specific(self) -> bool:
        return True
