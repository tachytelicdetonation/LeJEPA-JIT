"""
LeJEPA model wrapper combining encoder with projector head.

The LeJEPA architecture consists of:
1. Encoder: ViT or JiT backbone that produces embeddings
2. Projector: Multi-layer projection head for contrastive learning

The projector maps encoder embeddings to a higher-dimensional space
where the SIGReg loss is applied.

Matches reference implementation from LeJEPA MINIMAL.md.
"""

from typing import Union

import torch
import torch.nn as nn
from torchvision.ops import MLP

from .jit_encoder import JiTEncoder
from .vit_encoder import ViTEncoder


class LeJEPA(nn.Module):
    """
    LeJEPA model combining encoder and projector.

    The model takes images and produces:
    1. Embeddings: From the encoder (used for linear probe)
    2. Projections: From the projector (used for SIGReg loss)

    Key design choices:
    - No stop-gradients or teacher-student setup
    - Single network with direct optimization
    - Multiple augmented views processed together
    """

    def __init__(
        self,
        encoder: Union[JiTEncoder, ViTEncoder],
        proj_dim: int = 128,
    ):
        super().__init__()
        self.encoder = encoder
        # Use torchvision MLP like reference: 512 -> 2048 -> 2048 -> 128
        self.proj = MLP(
            encoder.embed_dim,
            [2048, 2048, proj_dim],
            norm_layer=nn.BatchNorm1d,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder and projector.

        Args:
            x: (B, V, C, H, W) multi-view input

        Returns:
            (emb, proj) where:
                - emb: (B*V, embed_dim * 2) encoder embeddings (last 2 layers)
                - proj: (V, B, proj_dim) projections (views first for loss)
        """
        B, V, C, H, W = x.shape

        # Flatten views into batch dimension
        x_flat = x.flatten(0, 1)  # (B*V, C, H, W)

        # Encoder - Get features from last 2 layers
        # Returns (B*V, embed_dim * 2) where first half is layer N-1, second half is layer N
        emb = self.encoder(x_flat, return_last_two=True)

        # Extract last layer for projector (second half)
        embed_dim = emb.shape[1] // 2
        last_layer_emb = emb[:, embed_dim:]

        # Projector uses last layer
        proj = self.proj(last_layer_emb)  # (B*V, proj_dim)

        # Reshape proj to (V, B, proj_dim) for loss computation
        proj = proj.reshape(B, V, -1).transpose(0, 1)

        return emb, proj

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get only encoder embeddings (for linear probe)."""
        if x.dim() == 5:
            x = x.flatten(0, 1)
        # Paper uses last 2 layers concatenated
        return self.encoder(x, return_last_two=True)


class LinearProbe(nn.Module):
    """
    Linear classifier for evaluating representation quality.

    Trained on frozen embeddings to measure linear separability.
    Uses LayerNorm before linear layer like reference.
    """

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def create_lejepa(
    encoder_type: str = "jit",
    img_size: int = 128,
    patch_size: int = 8,
    embed_dim: int = 512,
    depth: int = 12,
    num_heads: int = 8,
    proj_dim: int = 128,
    **kwargs,
) -> LeJEPA:
    """
    Factory function to create LeJEPA model.

    Args:
        encoder_type: "jit" or "vit"
        img_size: Input image size
        patch_size: Patch size for tokenization
        embed_dim: Encoder embedding dimension
        depth: Number of transformer layers
        num_heads: Number of attention heads
        proj_dim: Projector output dimension
        **kwargs: Additional encoder arguments

    Defaults to pool="cls" for paper compliance (required for proper feature concatenation).
    """
    # Force pool="cls" if not specified
    if "pool" not in kwargs:
        kwargs["pool"] = "cls"

    if encoder_type == "jit":
        encoder = JiTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            **kwargs,
        )
    elif encoder_type == "vit":
        encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    return LeJEPA(encoder=encoder, proj_dim=proj_dim)


def create_lejepa_small(encoder_type: str = "jit", **kwargs) -> LeJEPA:
    """Create LeJEPA with small encoder (ViT-Small or JiT-Small)."""
    return create_lejepa(
        encoder_type=encoder_type,
        embed_dim=512,
        depth=12,
        num_heads=8,
        **kwargs,
    )


def create_lejepa_base(encoder_type: str = "jit", **kwargs) -> LeJEPA:
    """Create LeJEPA with base encoder (ViT-Base or JiT-Base)."""
    return create_lejepa(
        encoder_type=encoder_type,
        embed_dim=768,
        depth=12,
        num_heads=12,
        **kwargs,
    )
