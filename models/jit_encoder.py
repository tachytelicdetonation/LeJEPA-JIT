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

    def _compute_embeddings(
        self, h: int, w: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute RoPE embeddings for specific resolution."""
        y_pos = torch.arange(h, device=device)
        x_pos = torch.arange(w, device=device)
        grid_y, grid_x = torch.meshgrid(y_pos, x_pos, indexing="ij")

        positions_y = grid_y.flatten().float()
        positions_x = grid_x.flatten().float()

        half_freqs = len(self.inv_freq) // 2
        freqs_y = torch.outer(positions_y, self.inv_freq[:half_freqs])
        freqs_x = torch.outer(positions_x, self.inv_freq[:half_freqs])

        freqs = torch.cat([freqs_y, freqs_x], dim=-1)
        return freqs.cos().to(dtype), freqs.sin().to(dtype)

    def _build_cache(self, resolution: int):
        """Build cos/sin cache for given resolution."""
        cos, sin = self._compute_embeddings(
            resolution, resolution, self.inv_freq.device, torch.float32
        )
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(
        self, x: torch.Tensor, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns cos and sin for rotary embedding.

        Args:
           seq_len: Number of patches
        """
        # Infer grid size from seq_len (assuming square)
        res = int(seq_len**0.5)
        if res * res != seq_len:
            # Fallback/Error if not square? Or handle linear?
            # For now assume square as LeJEPA uses square crops
            pass

        # Check if cached matches
        if self.cos_cached.shape[0] == seq_len:
            return (
                self.cos_cached.to(x.dtype),
                self.sin_cached.to(x.dtype),
            )

        # Determine if we can just slice (only if res matches max_res, unlikely for different scales)
        # Actually simplest is to just recompute if different.
        # Caching optimization: we could cache by resolution, but simple compute is fast enough probably.

        return self._compute_embeddings(res, res, x.device, x.dtype)


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
        head_dim = embed_dim // num_heads
        self.rope = VisionRotaryEmbedding(
            dim=head_dim,  # Full head_dim; _build_cache handles 2D split
            max_res=grid_size,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                JiTBlock(
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

    def forward(self, x: torch.Tensor, return_last_two: bool = False) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input images
            return_last_two: Whether to return concat of last 2 layers

        Returns:
            (B, embed_dim) if return_last_two=False
            (B, embed_dim * 2) if return_last_two=True
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

        # Calculate actual num patches (excluding CLS if present)
        current_num_patches = seq_len - 1 if self.pool == "cls" else seq_len

        # Get RoPE for this specific resolution
        rope_cos, rope_sin = self.rope(x, current_num_patches)

        if self.pool == "cls":
            # Pad for CLS token (no rotation for CLS)
            rope_cos = F.pad(rope_cos, (0, 0, 1, 0), value=1.0)
            rope_sin = F.pad(rope_sin, (0, 0, 1, 0), value=0.0)

        # Apply transformer blocks
        output_tokens = []
        for i, block in enumerate(self.blocks):
            x = block(x, rope_cos, rope_sin)
            if i >= len(self.blocks) - 2:
                output_tokens.append(x)

        x = self.norm(x)

        # Return last 2 layers
        if return_last_two and self.pool == "cls":
            # Normalize both outputs based on final norm logic (approximation, usually norm applies at end)
            # But here we applied norm at end of block sequence.
            # To match "CLS token from last two layers + LayerNorm", we should ideally take block outputs, Norm them, then Concat.
            # The existing code applied norm ONLY at the very end.
            # Let's apply self.norm to the second-to-last output too for consistency.
            out1 = self.norm(output_tokens[0])[:, 0]
            out2 = x[:, 0]  # Output of last block + norm
            return torch.cat([out1, out2], dim=-1)

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
