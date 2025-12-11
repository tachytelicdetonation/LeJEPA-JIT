"""
JiT (Just image Transformer) Encoder adapted for representation learning.

This implements the JiT architecture from "Back to Basics: Let Denoising Generative
Models Denoise" (Li & He, 2025) but adapted as a pure encoder for LeJEPA.

Key features kept from JiT:
- BottleneckPatchEmbed: Two-stage patch embedding
- RoPE: Rotary Position Embeddings
- SwiGLU FFN: More efficient than standard MLP
- RMSNorm: Applied to Q/K in attention

Removed (diffusion-specific):
- TimestepEmbedder
- LabelEmbedder
- Adaptive Layer Normalization (AdaLN)
- FinalLayer denoising head
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class BottleneckPatchEmbed(nn.Module):
    """
    Two-stage patch embedding from JiT.
    First extracts patches, then projects through a bottleneck.
    """

    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 512,
        bottleneck_dim: int = 128,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # First conv: extract patches into bottleneck dimension
        self.proj1 = nn.Conv2d(
            in_channels,
            bottleneck_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Second conv: project to embedding dimension
        self.proj2 = nn.Conv2d(
            bottleneck_dim,
            embed_dim,
            kernel_size=1,
        )

        self.norm = RMSNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input images

        Returns:
            (B, num_patches, embed_dim) patch embeddings
        """
        x = self.proj1(x)  # (B, bottleneck_dim, H/P, W/P)
        x = F.gelu(x)
        x = self.proj2(x)  # (B, embed_dim, H/P, W/P)
        x = rearrange(x, "b c h w -> b (h w) c")  # (B, num_patches, embed_dim)
        x = self.norm(x)
        return x


class VisionRotaryEmbedding(nn.Module):
    """
    2D Rotary Position Embeddings for Vision Transformers.
    Applies separate rotary embeddings for height and width dimensions.
    """

    def __init__(
        self,
        dim: int,
        max_res: int = 64,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_res = max_res

        # Create frequency bands
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute position indices for 2D
        self._build_cache(max_res)

    def _build_cache(self, resolution: int):
        """Build cos/sin cache for given resolution."""
        # Create 2D position grid
        y_pos = torch.arange(resolution)
        x_pos = torch.arange(resolution)
        grid_y, grid_x = torch.meshgrid(y_pos, x_pos, indexing="ij")

        # Flatten to sequence
        positions_y = grid_y.flatten().float()  # (resolution^2,)
        positions_x = grid_x.flatten().float()

        # Compute frequencies for y and x separately
        # Split dim in half for y and x
        half_dim = self.dim // 2

        freqs_y = torch.outer(positions_y, self.inv_freq[:half_dim])  # (seq, dim/4)
        freqs_x = torch.outer(positions_x, self.inv_freq[:half_dim])

        # Combine y and x frequencies
        freqs = torch.cat([freqs_y, freqs_x], dim=-1)  # (seq, dim/2)

        # Create cos and sin embeddings
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns cos and sin for rotary embedding.

        Args:
            x: Input tensor (for device/dtype)
            seq_len: Sequence length (num_patches)

        Returns:
            (cos, sin) each of shape (seq_len, dim/2)
        """
        return (
            self.cos_cached[:seq_len].to(x.dtype),
            self.sin_cached[:seq_len].to(x.dtype),
        )


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to queries and keys.

    Args:
        q: (B, heads, seq, head_dim)
        k: (B, heads, seq, head_dim)
        cos: (seq, head_dim/2)
        sin: (seq, head_dim/2)

    Returns:
        Rotated (q, k)
    """

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    # Reshape cos/sin for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, dim/2)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Repeat cos/sin to match head_dim
    cos = torch.cat([cos, cos], dim=-1)  # (1, 1, seq, dim)
    sin = torch.cat([sin, sin], dim=-1)

    q_rotated = (q * cos) + (rotate_half(q) * sin)
    k_rotated = (k * cos) + (rotate_half(k) * sin)

    return q_rotated, k_rotated


class Attention(nn.Module):
    """
    Multi-head self-attention with RoPE and RMSNorm on Q/K.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) input tokens
            rope_cos: (N, head_dim/2) cosine for RoPE
            rope_sin: (N, head_dim/2) sine for RoPE

        Returns:
            (B, N, C) output tokens
        """
        B, N, C = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        # RMSNorm on Q and K
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply rotary embeddings
        q, k = apply_rotary_emb(q, k, rope_cos, rope_sin)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network from JiT.
    More efficient than standard MLP with GELU.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 4 * 2 / 3)  # SwiGLU uses 2/3 scaling

        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: (Swish(W1 * x) * (W2 * x))
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class JiTBlock(nn.Module):
    """
    JiT Transformer block with RoPE attention and SwiGLU FFN.
    Simplified version without AdaLN modulation.
    """

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
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLUFFN(
            dim,
            hidden_dim=int(dim * mlp_ratio * 2 / 3),
            drop=drop,
        )

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin)
        x = x + self.mlp(self.norm2(x))
        return x


class JiTEncoder(nn.Module):
    """
    JiT Encoder for representation learning.

    Architecture from JiT paper adapted for self-supervised learning:
    - BottleneckPatchEmbed for patch tokenization
    - RoPE for position encoding
    - SwiGLU FFN
    - RMSNorm on Q/K in attention

    Outputs pooled representations suitable for LeJEPA projector.
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
        bottleneck_dim: int = 128,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        pool: str = "mean",  # "mean" or "cls"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.pool = pool

        # Patch embedding
        self.patch_embed = BottleneckPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            bottleneck_dim=bottleneck_dim,
        )

        # CLS token (optional, for cls pooling)
        if pool == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Rotary position embeddings
        grid_size = img_size // patch_size
        self.rope = VisionRotaryEmbedding(
            dim=embed_dim // num_heads,
            max_res=grid_size,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            JiTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
            )
            for _ in range(depth)
        ])

        # Final norm
        self.norm = RMSNorm(embed_dim)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input images

        Returns:
            (B, embed_dim) pooled representations
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add CLS token if using cls pooling
        if self.pool == "cls":
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        # Get rotary embeddings
        seq_len = x.shape[1]
        if self.pool == "cls":
            # For CLS, we need to handle the extra token
            rope_cos, rope_sin = self.rope(x, self.num_patches)
            # Pad for CLS token (no rotation for CLS)
            rope_cos = F.pad(rope_cos, (0, 0, 1, 0), value=1.0)
            rope_sin = F.pad(rope_sin, (0, 0, 1, 0), value=0.0)
        else:
            rope_cos, rope_sin = self.rope(x, seq_len)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, rope_cos, rope_sin)

        x = self.norm(x)

        # Pool to get final representation
        if self.pool == "cls":
            x = x[:, 0]  # CLS token
        else:
            x = x.mean(dim=1)  # Mean pooling

        return x


def jit_small(img_size: int = 128, patch_size: int = 8, **kwargs) -> JiTEncoder:
    """JiT-Small configuration matching ViT-Small."""
    return JiTEncoder(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        bottleneck_dim=128,
        **kwargs,
    )


def jit_base(img_size: int = 128, patch_size: int = 8, **kwargs) -> JiTEncoder:
    """JiT-Base configuration."""
    return JiTEncoder(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        bottleneck_dim=128,
        **kwargs,
    )
