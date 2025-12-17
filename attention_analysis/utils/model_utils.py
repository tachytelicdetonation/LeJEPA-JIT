"""
Model utilities for attention analysis.

Helper functions for extracting model information and architecture details.
"""

from __future__ import annotations

import torch.nn as nn


def get_attention_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """
    Find all attention layers in a model.

    Supports:
    - timm ViT models
    - HuggingFace ViT models
    - LeJEPA-JIT models

    Args:
        model: The model to inspect.

    Returns:
        List of (name, module) tuples for attention layers.
    """
    attention_layers = []

    # LeJEPA-JIT structure
    if hasattr(model, "encoder") and hasattr(model.encoder, "blocks"):
        for i, block in enumerate(model.encoder.blocks):
            if hasattr(block, "attn"):
                attention_layers.append((f"encoder.blocks.{i}.attn", block.attn))
        return attention_layers

    # timm structure
    if hasattr(model, "blocks"):
        for i, block in enumerate(model.blocks):
            if hasattr(block, "attn"):
                attention_layers.append((f"blocks.{i}.attn", block.attn))
        return attention_layers

    # HuggingFace structure
    if hasattr(model, "vit"):
        encoder = model.vit.encoder
        for i, layer in enumerate(encoder.layer):
            if hasattr(layer, "attention"):
                attention_layers.append(
                    (f"vit.encoder.layer.{i}.attention", layer.attention)
                )
        return attention_layers

    # Fallback: recursive search
    for name, module in model.named_modules():
        module_name = module.__class__.__name__.lower()
        if "attention" in module_name or name.endswith(".attn"):
            attention_layers.append((name, module))

    return attention_layers


def get_model_info(model: nn.Module) -> dict:
    """
    Extract model architecture information.

    Args:
        model: The model to inspect.

    Returns:
        Dictionary with model information:
        - model_type: Type of model (jit, vit, timm, huggingface)
        - num_layers: Number of transformer blocks
        - num_heads: Number of attention heads
        - embed_dim: Embedding dimension
        - patch_size: Patch size (if applicable)
        - img_size: Expected image size
    """
    info = {
        "model_type": "unknown",
        "num_layers": 0,
        "num_heads": 0,
        "embed_dim": 0,
        "patch_size": 16,
        "img_size": 224,
    }

    # LeJEPA-JIT model
    if hasattr(model, "encoder"):
        encoder = model.encoder
        info["model_type"] = "lejepa"

        if hasattr(encoder, "blocks"):
            info["num_layers"] = len(encoder.blocks)
            if len(encoder.blocks) > 0:
                block = encoder.blocks[0]
                if hasattr(block, "attn"):
                    attn = block.attn
                    if hasattr(attn, "num_heads"):
                        info["num_heads"] = attn.num_heads
                    if hasattr(attn, "head_dim"):
                        info["embed_dim"] = attn.num_heads * attn.head_dim

        if hasattr(encoder, "patch_embed"):
            pe = encoder.patch_embed
            if hasattr(pe, "patch_size"):
                ps = pe.patch_size
                info["patch_size"] = ps[0] if isinstance(ps, tuple) else ps
            if hasattr(pe, "img_size"):
                ims = pe.img_size
                info["img_size"] = ims[0] if isinstance(ims, tuple) else ims

        # Determine JiT vs ViT
        if hasattr(encoder, "blocks") and len(encoder.blocks) > 0:
            block = encoder.blocks[0]
            if hasattr(block, "mlp") and hasattr(block.mlp, "w1"):
                info["model_type"] = "jit"  # SwiGLU MLP indicates JiT
            else:
                info["model_type"] = "vit"

        return info

    # timm model
    if hasattr(model, "blocks"):
        info["model_type"] = "timm"
        info["num_layers"] = len(model.blocks)

        if len(model.blocks) > 0:
            block = model.blocks[0]
            if hasattr(block, "attn"):
                attn = block.attn
                if hasattr(attn, "num_heads"):
                    info["num_heads"] = attn.num_heads

        if hasattr(model, "embed_dim"):
            info["embed_dim"] = model.embed_dim
        if hasattr(model, "patch_embed"):
            if hasattr(model.patch_embed, "patch_size"):
                ps = model.patch_embed.patch_size
                info["patch_size"] = ps[0] if isinstance(ps, tuple) else ps

        return info

    # HuggingFace model
    if hasattr(model, "config"):
        info["model_type"] = "huggingface"
        config = model.config
        if hasattr(config, "num_hidden_layers"):
            info["num_layers"] = config.num_hidden_layers
        if hasattr(config, "num_attention_heads"):
            info["num_heads"] = config.num_attention_heads
        if hasattr(config, "hidden_size"):
            info["embed_dim"] = config.hidden_size
        if hasattr(config, "patch_size"):
            info["patch_size"] = config.patch_size
        if hasattr(config, "image_size"):
            info["img_size"] = config.image_size
        return info

    return info


def get_patch_grid_size(
    model: nn.Module,
    img_size: int | None = None,
) -> tuple[int, int]:
    """
    Get the patch grid dimensions for a ViT model.

    Args:
        model: The model.
        img_size: Override image size. If None, infer from model.

    Returns:
        (grid_height, grid_width) tuple.
    """
    info = get_model_info(model)
    patch_size = info["patch_size"]
    if img_size is None:
        img_size = info["img_size"]

    grid_size = img_size // patch_size
    return (grid_size, grid_size)


def get_num_patches(model: nn.Module, img_size: int | None = None) -> int:
    """
    Get the number of patches for a ViT model.

    Args:
        model: The model.
        img_size: Override image size.

    Returns:
        Number of patches (excluding CLS token).
    """
    h, w = get_patch_grid_size(model, img_size)
    return h * w


def has_cls_token(model: nn.Module) -> bool:
    """
    Check if model uses a CLS token.

    Args:
        model: The model.

    Returns:
        True if model has a CLS token.
    """
    # LeJEPA-JIT models don't use CLS token
    if hasattr(model, "encoder"):
        encoder = model.encoder
        if hasattr(encoder, "cls_token"):
            return encoder.cls_token is not None
        return False

    # timm models typically have CLS
    if hasattr(model, "cls_token"):
        return model.cls_token is not None

    # HuggingFace ViT uses CLS
    if hasattr(model, "config") and hasattr(model.config, "num_labels"):
        return True

    return False


def get_embed_dim(model: nn.Module) -> int:
    """Get the embedding dimension of the model."""
    info = get_model_info(model)
    return info["embed_dim"]


def get_num_heads(model: nn.Module) -> int:
    """Get the number of attention heads."""
    info = get_model_info(model)
    return info["num_heads"]
