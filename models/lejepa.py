"""
LeJEPA model wrapper combining encoder with projector head.

The LeJEPA architecture consists of:
1. Encoder: ViT or JiT backbone that produces embeddings
2. Projector: Multi-layer projection head for contrastive learning

The projector maps encoder embeddings to a higher-dimensional space
where the SIGReg loss is applied.
"""

from typing import Union

import torch
import torch.nn as nn

from .jit_encoder import JiTEncoder
from .vit_encoder import ViTEncoder


class Projector(nn.Module):
    """
    Multi-layer projection head from LeJEPA.

    Maps encoder embeddings through hidden layers to projection space
    where the SIGReg loss operates.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 2048,
        out_dim: int = 2048,
        num_layers: int = 3,
    ):
        super().__init__()
        layers = []

        # Input layer
        layers.extend(
            [
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ]
        )

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                ]
            )

        # Output layer (no activation)
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
        proj_hidden_dim: int = 2048,
        proj_dim: int = 2048,
        proj_layers: int = 3,
    ):
        super().__init__()
        self.encoder = encoder
        self.projector = Projector(
            in_dim=encoder.embed_dim,
            hidden_dim=proj_hidden_dim,
            out_dim=proj_dim,
            num_layers=proj_layers,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Forward pass through encoder and projector.

        Args:
            x: (B, C, H, W) or (B, V, C, H, W) for multi-view
            return_embedding: If True, return both embedding and projection

        Returns:
            If return_embedding:
                (embedding, projection) - both (B, dim) or (B*V, dim)
            Else:
                projection only
        """
        # Handle multi-view input
        if x.dim() == 5:
            B, V, C, H, W = x.shape
            x = x.view(B * V, C, H, W)
        else:
            pass

        # Encoder
        embedding = self.encoder(x)  # (B*V, embed_dim) or (B, embed_dim)

        # Projector
        projection = self.projector(embedding)  # (B*V, proj_dim) or (B, proj_dim)

        if return_embedding:
            return embedding, projection
        return projection

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get only encoder embeddings (for linear probe)."""
        if x.dim() == 5:
            B, V, C, H, W = x.shape
            x = x.view(B * V, C, H, W)
        return self.encoder(x)


class LinearProbe(nn.Module):
    """
    Linear classifier for evaluating representation quality.

    Trained on frozen embeddings to measure linear separability.
    """

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def create_lejepa(
    encoder_type: str = "jit",
    img_size: int = 128,
    patch_size: int = 8,
    embed_dim: int = 512,
    depth: int = 12,
    num_heads: int = 8,
    proj_hidden_dim: int = 2048,
    proj_dim: int = 2048,
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
        proj_hidden_dim: Projector hidden dimension
        proj_dim: Projector output dimension
        **kwargs: Additional encoder arguments

    Returns:
        LeJEPA model
    """
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

    return LeJEPA(
        encoder=encoder,
        proj_hidden_dim=proj_hidden_dim,
        proj_dim=proj_dim,
    )


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
