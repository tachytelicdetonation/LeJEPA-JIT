"""
Hook Manager for Vision Transformer attention extraction.

Provides a unified interface for extracting attention weights and activations
from any ViT-style model without modifying model code.
"""

from __future__ import annotations

from typing import Callable
from collections import OrderedDict

import torch
import torch.nn as nn


class HookManager:
    """
    Manages forward and backward hooks for attention extraction.

    Works with:
    - timm models (vit_base_patch16_224, etc.)
    - HuggingFace ViT models
    - Custom ViT implementations (JiT, etc.)

    Example:
        >>> hook_mgr = HookManager(model)
        >>> hook_mgr.register_attention_hooks()
        >>> output = model(images)
        >>> attentions = hook_mgr.get_attention_maps()
        >>> hook_mgr.clear()
    """

    def __init__(self, model: nn.Module):
        """
        Initialize HookManager.

        Args:
            model: The model to attach hooks to.
        """
        self.model = model
        self._handles: list = []
        self._activations: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._gradients: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._attention_maps: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._attention_gradients: OrderedDict[str, torch.Tensor] = OrderedDict()

    def _get_activation_hook(self, name: str) -> Callable:
        """Create a forward hook that stores activations."""

        def hook(module: nn.Module, input: tuple, output: torch.Tensor) -> None:
            # Handle different output types
            if isinstance(output, tuple):
                # Some attention modules return (output, attention_weights)
                self._activations[name] = output[0].detach()
                if len(output) > 1 and output[1] is not None:
                    self._attention_maps[name] = output[1].detach()
            else:
                self._activations[name] = output.detach()

        return hook

    def _get_gradient_hook(self, name: str) -> Callable:
        """Create a backward hook that stores gradients."""

        def hook(module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
            if grad_output[0] is not None:
                self._gradients[name] = grad_output[0].detach()

        return hook

    def _get_attention_hook(self, name: str) -> Callable:
        """
        Create a hook specifically for attention weight extraction.

        This hook is designed to work with attention modules that compute
        attention weights internally.
        """

        def hook(module: nn.Module, input: tuple, output: torch.Tensor) -> None:
            # Try to get attention weights from module attributes
            # Works with our JiT/ViT encoders that store attn_map
            if hasattr(module, "attn_map") and module.attn_map is not None:
                self._attention_maps[name] = module.attn_map.detach().cpu()
            elif hasattr(module, "attention_weights"):
                self._attention_maps[name] = module.attention_weights.detach().cpu()

        return hook

    def _get_attention_gradient_hook(self, name: str) -> Callable:
        """Create a hook for attention gradient extraction."""

        def hook(module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
            if hasattr(module, "attn_map") and module.attn_map is not None:
                # Gradient w.r.t. attention weights if available
                if module.attn_map.grad is not None:
                    self._attention_gradients[name] = (
                        module.attn_map.grad.detach().cpu()
                    )

        return hook

    def register_forward_hook(
        self, layer: nn.Module, name: str, hook_fn: Callable | None = None
    ) -> None:
        """
        Register a forward hook on a specific layer.

        Args:
            layer: The module to attach the hook to.
            name: Identifier for this hook's outputs.
            hook_fn: Optional custom hook function. If None, uses default activation hook.
        """
        if hook_fn is None:
            hook_fn = self._get_activation_hook(name)
        handle = layer.register_forward_hook(hook_fn)
        self._handles.append(handle)

    def register_backward_hook(
        self, layer: nn.Module, name: str, hook_fn: Callable | None = None
    ) -> None:
        """
        Register a backward hook on a specific layer.

        Args:
            layer: The module to attach the hook to.
            name: Identifier for this hook's outputs.
            hook_fn: Optional custom hook function.
        """
        if hook_fn is None:
            hook_fn = self._get_gradient_hook(name)
        handle = layer.register_full_backward_hook(hook_fn)
        self._handles.append(handle)

    def register_attention_hooks(
        self,
        include_gradients: bool = False,
        layer_indices: list[int] | None = None,
    ) -> list[str]:
        """
        Auto-detect and register hooks on all attention layers.

        Args:
            include_gradients: Whether to also register backward hooks.
            layer_indices: Specific layer indices to hook. None = all layers.

        Returns:
            List of registered layer names.
        """
        registered = []
        attention_layers = self._find_attention_layers()

        for idx, (name, layer) in enumerate(attention_layers):
            if layer_indices is not None and idx not in layer_indices:
                continue

            # Register forward hook for attention capture
            self.register_forward_hook(layer, name, self._get_attention_hook(name))

            if include_gradients:
                self.register_backward_hook(
                    layer, f"{name}_grad", self._get_attention_gradient_hook(name)
                )

            registered.append(name)

        return registered

    def _find_attention_layers(self) -> list[tuple[str, nn.Module]]:
        """
        Find all attention layers in the model.

        Supports:
        - timm ViT: model.blocks[i].attn
        - HuggingFace ViT: model.vit.encoder.layer[i].attention
        - LeJEPA-JIT: model.encoder.blocks[i].attn
        """
        attention_layers = []

        # Try LeJEPA-JIT structure first
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "blocks"):
            for i, block in enumerate(self.model.encoder.blocks):
                if hasattr(block, "attn"):
                    attention_layers.append((f"encoder.blocks.{i}.attn", block.attn))
            return attention_layers

        # Try timm structure
        if hasattr(self.model, "blocks"):
            for i, block in enumerate(self.model.blocks):
                if hasattr(block, "attn"):
                    attention_layers.append((f"blocks.{i}.attn", block.attn))
            return attention_layers

        # Try HuggingFace structure
        if hasattr(self.model, "vit"):
            encoder = self.model.vit.encoder
            for i, layer in enumerate(encoder.layer):
                if hasattr(layer, "attention"):
                    attention_layers.append(
                        (f"vit.encoder.layer.{i}.attention", layer.attention)
                    )
            return attention_layers

        # Fallback: search recursively for attention modules
        for name, module in self.model.named_modules():
            if "attn" in name.lower() or "attention" in name.lower():
                # Skip if it's a parent module of an attention we already found
                if not any(
                    name.startswith(existing) for existing, _ in attention_layers
                ):
                    attention_layers.append((name, module))

        return attention_layers

    def enable_attention_capture(self) -> None:
        """
        Enable attention capture on model's attention modules.

        This sets the output_attention flag on modules that support it
        (like our JiT/ViT encoders).
        """
        for name, layer in self._find_attention_layers():
            if hasattr(layer, "output_attention"):
                layer.output_attention = True

    def disable_attention_capture(self) -> None:
        """Disable attention capture and clear cached maps."""
        for name, layer in self._find_attention_layers():
            if hasattr(layer, "output_attention"):
                layer.output_attention = False
            if hasattr(layer, "attn_map"):
                layer.attn_map = None

    def get_attention_maps(self) -> list[torch.Tensor]:
        """
        Get captured attention maps in layer order.

        Returns:
            List of attention tensors, shape (B, H, N, N) each.
        """
        return list(self._attention_maps.values())

    def get_attention_maps_dict(self) -> OrderedDict[str, torch.Tensor]:
        """Get attention maps as an ordered dictionary."""
        return self._attention_maps.copy()

    def get_attention_gradients(self) -> list[torch.Tensor]:
        """Get captured attention gradients in layer order."""
        return list(self._attention_gradients.values())

    def get_activations(self) -> OrderedDict[str, torch.Tensor]:
        """Get all stored activations."""
        return self._activations.copy()

    def get_gradients(self) -> OrderedDict[str, torch.Tensor]:
        """Get all stored gradients."""
        return self._gradients.copy()

    def clear_cache(self) -> None:
        """Clear all cached activations and gradients."""
        self._activations.clear()
        self._gradients.clear()
        self._attention_maps.clear()
        self._attention_gradients.clear()

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def clear(self) -> None:
        """Remove hooks and clear all cached data."""
        self.remove_hooks()
        self.clear_cache()
        self.disable_attention_capture()

    def __enter__(self) -> "HookManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - clean up hooks."""
        self.clear()

    def get_num_layers(self) -> int:
        """Get the number of attention layers in the model."""
        return len(self._find_attention_layers())

    def get_layer_names(self) -> list[str]:
        """Get names of all attention layers."""
        return [name for name, _ in self._find_attention_layers()]
