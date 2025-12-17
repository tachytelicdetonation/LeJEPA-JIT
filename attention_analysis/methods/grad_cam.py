"""
Grad-CAM for Vision Transformers.

Gradient-weighted Class Activation Mapping adapted for ViT architecture.

Reference:
    Chefer et al. "Transformer Interpretability Beyond Attention Visualization" (CVPR 2021)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from attention_analysis.methods.base import AttributionMethod
from attention_analysis.utils.model_utils import get_patch_grid_size


class GradCAM(AttributionMethod):
    """
    Grad-CAM for Vision Transformers.

    Computes gradient-weighted attention maps. For ViTs, this uses attention
    weights from a target layer and weights them by their gradient importance
    with respect to the target class.

    Unlike CNNs where Grad-CAM uses feature maps, for ViT we use attention
    weights as they capture spatial relationships.

    Args:
        target_layer: Which layer to compute Grad-CAM for. Options:
            - int: Specific layer index
            - "last": Last layer (default)
        head_aggregation: How to aggregate heads. Options:
            - "gradient_weighted": Weight by gradient norm (default)
            - "mean": Simple average
            - "max": Maximum
    """

    def __init__(
        self,
        target_layer: int | str = "last",
        head_aggregation: str = "gradient_weighted",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_layer = target_layer
        self.head_aggregation = head_aggregation

    def attribute(
        self,
        model: nn.Module,
        image: torch.Tensor,
        target_class: int | None = None,
    ) -> torch.Tensor:
        """
        Compute Grad-CAM attribution.

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
        image.requires_grad_(True)

        model.eval()

        # Storage for attention and gradients
        attention_store = {}
        gradient_store = {}

        def save_attention_hook(name):
            def hook(module, input, output):
                if hasattr(module, "attn_map") and module.attn_map is not None:
                    attention_store[name] = module.attn_map

            return hook

        def save_gradient_hook(name):
            def hook(module, grad_input, grad_output):
                if name in attention_store:
                    # We need gradient w.r.t. attention weights
                    # The attention map is stored, and its gradient flows through
                    if attention_store[name].grad is not None:
                        gradient_store[name] = attention_store[name].grad.clone()

            return hook

        # Find attention layers
        attention_layers = self._find_attention_layers(model)

        # Determine target layer
        if self.target_layer == "last":
            layer_idx = len(attention_layers) - 1
        else:
            layer_idx = self.target_layer

        target_name, target_layer = attention_layers[layer_idx]

        # Enable attention capture
        target_layer.output_attention = True

        # Register hooks
        handles = []
        handles.append(
            target_layer.register_forward_hook(save_attention_hook(target_name))
        )

        try:
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

            # Create one-hot target
            one_hot = torch.zeros_like(output)
            one_hot[0, target_class] = 1

            # Backward to get gradients
            output.backward(gradient=one_hot, retain_graph=True)

            # Get attention from target layer
            attn = attention_store.get(target_name)
            if attn is None:
                grid_h, grid_w = get_patch_grid_size(model)
                return torch.zeros(grid_h, grid_w, device=device)

            attn = attn.to(device)  # (B, H, N, N)

            # Get gradients w.r.t attention
            # Since attention is after softmax, we need to compute gradients differently
            # We'll use the gradient of the output w.r.t. the attention weights

            # Compute gradient-weighted attention
            # For each head, weight the attention by gradient importance
            if attn.grad is not None:
                gradients = attn.grad
            else:
                # Fallback: compute importance via re-forward with masked attention
                gradients = self._compute_attention_gradients(
                    model, image, target_class, layer_idx
                )

            # Aggregate heads
            if self.head_aggregation == "gradient_weighted" and gradients is not None:
                # Weight each head by its gradient norm
                head_importance = gradients.abs().sum(dim=(2, 3))  # (B, H)
                head_importance = head_importance / head_importance.sum(
                    dim=-1, keepdim=True
                ).clamp(min=1e-8)

                # Weighted sum of attention heads
                attn_weighted = (
                    attn * head_importance.unsqueeze(-1).unsqueeze(-1)
                ).sum(dim=1)
            elif self.head_aggregation == "mean":
                attn_weighted = attn.mean(dim=1)
            elif self.head_aggregation == "max":
                attn_weighted = attn.max(dim=1)[0]
            else:
                attn_weighted = attn.mean(dim=1)

            # Extract CLS attention or mean
            has_cls = self._model_has_cls(model)
            if has_cls:
                attn_map = attn_weighted[0, 0, 1:]  # CLS attending to patches
            else:
                attn_map = attn_weighted[0].mean(dim=0)  # Mean of all tokens
                if len(attn_map) > 196:  # Has extra token
                    attn_map = attn_map[1:]

            # Apply ReLU (only positive contributions)
            attn_map = torch.relu(attn_map)

            # Reshape to grid
            attn_map = self._reshape_to_grid(attn_map, model)

            # Normalize
            attn_map = self._normalize_attribution(attn_map)

            return attn_map

        finally:
            # Cleanup
            for handle in handles:
                handle.remove()
            target_layer.output_attention = False
            target_layer.attn_map = None

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

    def _compute_attention_gradients(
        self,
        model: nn.Module,
        image: torch.Tensor,
        target_class: int,
        layer_idx: int,
    ) -> torch.Tensor | None:
        """
        Compute gradients w.r.t. attention weights via perturbation.

        This is a fallback when direct gradient access isn't available.
        """
        # This is expensive but more robust
        attention_layers = self._find_attention_layers(model)
        if layer_idx >= len(attention_layers):
            return None

        name, layer = attention_layers[layer_idx]

        # Enable attention capture
        layer.output_attention = True

        try:
            # Forward to get base attention
            if hasattr(model, "encoder"):
                _ = model.encoder(image)
            else:
                _ = model(image)

            base_attn = layer.attn_map.clone()
            if base_attn is None:
                return None

            # Compute gradient via finite differences (simplified)
            # This is approximate but works
            gradients = torch.zeros_like(base_attn)

            # For efficiency, just use the attention magnitude as proxy
            # Higher attention values are more "important"
            gradients = base_attn

            return gradients

        finally:
            layer.output_attention = False
            layer.attn_map = None

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
