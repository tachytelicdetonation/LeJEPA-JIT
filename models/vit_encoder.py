"""
Standard Vision Transformer (ViT) Encoder for baseline comparison.

This implements a standard ViT-Small architecture matching the one used
in the original LeJEPA paper for fair comparison with JiT encoder.
"""

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange


class PatchEmbed(nn.Module):
    """Standard patch embedding with single convolution."""

    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input images

        Returns:
            (B, num_patches, embed_dim) patch embeddings
        """
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = rearrange(x, "b c h w -> b (h w) c")  # (B, num_patches, embed_dim)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.output_attention = False
        self.attn_map = None

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        if getattr(self, "output_attention", False):
            self.attn_map = attn.detach()

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    """Standard MLP with GELU activation."""

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ViTBlock(nn.Module):
    """Standard ViT transformer block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(
            dim,
            hidden_dim=int(dim * mlp_ratio),
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """
    Standard Vision Transformer Encoder.

    Baseline implementation for comparison with JiT encoder.
    Uses:
    - Standard patch embedding
    - Learned positional embeddings
    - Standard LayerNorm
    - Standard GELU MLP
    """

    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        pool: str = "mean",  # "mean" or "cls"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.pool = pool

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        # CLS token (optional)
        if pool == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            num_tokens = self.num_patches + 1
        else:
            num_tokens = self.num_patches

        # Learned positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                ViTBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                )
                for _ in range(depth)
            ]
        )

        # Final norm
        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(N**0.5), int(N**0.5), dim).permute(
                0, 3, 1, 2
            ),
            scale_factor=(w0 / (N**0.5), h0 / (N**0.5)),
            mode="bicubic",
        )
        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, x: torch.Tensor, return_last_two: bool = False) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input images
            return_last_two: Whether to return concat of last 2 layers

        Returns:
            (B, embed_dim) if return_last_two=False
            (B, embed_dim * 2) if return_last_two=True
        """
        B, C, H, W = x.shape

        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add CLS token if using cls pooling
        if self.pool == "cls":
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

            # Interpolate positional embeddings
            pos_embed = self.interpolate_pos_encoding(x, W, H)
            x = x + pos_embed
        else:
            # Standard handling without CLS (not fully updated for variable size + mean pool here based on snippet logic but focusing on CLS for Paper compat)
            # Assuming CLS usage for paper comparison as per README "CLS token from last two layers"
            x = x + self.pos_embed

        x = self.pos_drop(x)

        # Apply transformer blocks
        output_tokens = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            # Store last 2 layers
            if i >= len(self.blocks) - 2:
                output_tokens.append(x)

        x = self.norm(x)

        # Pool to get final representation
        if return_last_two and self.pool == "cls":
            # Normalize both outputs before concatenating
            out1 = self.norm(output_tokens[0])[:, 0]
            out2 = self.norm(output_tokens[1])[:, 0]  # This is technically x[:,0]
            return torch.cat([out1, out2], dim=-1)

        if self.pool == "cls":
            x = x[:, 0]  # CLS token
        else:
            x = x.mean(dim=1)  # Mean pooling

        return x

    def get_attention_maps(self):
        """Return list of attention maps from all blocks."""
        return [block.attn.attn_map for block in self.blocks]


def vit_small(img_size: int = 128, patch_size: int = 8, **kwargs) -> ViTEncoder:
    """ViT-Small configuration."""
    return ViTEncoder(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        **kwargs,
    )


def vit_base(img_size: int = 128, patch_size: int = 8, **kwargs) -> ViTEncoder:
    """ViT-Base configuration."""
    return ViTEncoder(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        **kwargs,
    )
