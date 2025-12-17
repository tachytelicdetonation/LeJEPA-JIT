"""
Attribution Methods for Vision Transformers.

This module provides a unified interface for various attention attribution methods.

Available Methods:
- AttnLRP: Layer-wise Relevance Propagation optimized for transformers (ICML 2024)
- GradCAM: Gradient-weighted Class Activation Mapping for ViT
- AttentionRollout: Attention flow accumulation across layers
- GMAR: Gradient-driven Multi-head Attention Rollout
- RawAttention: Direct attention weight extraction

Usage:
    >>> from attention_analysis.methods import get_attribution
    >>> attribution = get_attribution(model, image, method="attn_lrp", target_class=0)
"""

from attention_analysis.methods.base import AttributionMethod
from attention_analysis.methods.attn_lrp import AttnLRP
from attention_analysis.methods.grad_cam import GradCAM
from attention_analysis.methods.rollout import AttentionRollout
from attention_analysis.methods.gmar import GMAR
from attention_analysis.methods.raw_attention import RawAttention

# Registry of available methods
_METHODS: dict[str, type[AttributionMethod]] = {
    "attn_lrp": AttnLRP,
    "grad_cam": GradCAM,
    "rollout": AttentionRollout,
    "gmar": GMAR,
    "raw_attention": RawAttention,
}


def list_methods() -> list[str]:
    """List all available attribution methods."""
    return list(_METHODS.keys())


def get_method(name: str, **kwargs) -> AttributionMethod:
    """
    Get an attribution method by name.

    Args:
        name: Method name (see list_methods()).
        **kwargs: Method-specific configuration.

    Returns:
        Configured AttributionMethod instance.
    """
    if name not in _METHODS:
        raise ValueError(f"Unknown method: {name}. Available: {list_methods()}")
    return _METHODS[name](**kwargs)


def get_attribution(
    model,
    image,
    method: str = "attn_lrp",
    target_class: int | None = None,
    **kwargs,
):
    """
    Compute attribution map for an image.

    This is the main entry point for computing attributions.

    Args:
        model: The Vision Transformer model.
        image: Input image tensor, shape (C, H, W) or (B, C, H, W).
        method: Attribution method name.
        target_class: Target class for class-specific methods.
            If None, uses predicted class.
        **kwargs: Method-specific arguments.

    Returns:
        Attribution map tensor of shape (H_patches, W_patches) or (H, W).
    """
    method_instance = get_method(method, **kwargs)
    return method_instance(model, image, target_class=target_class)


def register_method(name: str, method_class: type[AttributionMethod]) -> None:
    """
    Register a custom attribution method.

    Args:
        name: Name for the method.
        method_class: Class inheriting from AttributionMethod.
    """
    _METHODS[name] = method_class


__all__ = [
    # Functions
    "list_methods",
    "get_method",
    "get_attribution",
    "register_method",
    # Base class
    "AttributionMethod",
    # Methods
    "AttnLRP",
    "GradCAM",
    "AttentionRollout",
    "GMAR",
    "RawAttention",
]
