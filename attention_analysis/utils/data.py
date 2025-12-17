"""
Data utilities for attention analysis.

Functions for image manipulation during faithfulness evaluation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


def create_baseline_image(
    image: torch.Tensor,
    baseline_type: str = "blur",
    blur_sigma: float = 15.0,
) -> torch.Tensor:
    """
    Create a baseline image for insertion/deletion metrics.

    Args:
        image: Input image tensor, shape (C, H, W) or (B, C, H, W).
        baseline_type: Type of baseline. Options:
            - "blur": Gaussian blur
            - "mean": Mean pixel value
            - "black": All zeros
            - "noise": Random noise
        blur_sigma: Sigma for Gaussian blur.

    Returns:
        Baseline image with same shape as input.
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    B, C, H, W = image.shape
    device = image.device
    dtype = image.dtype

    if baseline_type == "blur":
        # Create Gaussian kernel
        kernel_size = int(6 * blur_sigma) | 1  # Ensure odd
        x = torch.arange(kernel_size, dtype=dtype, device=device) - kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (x / blur_sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Apply separable convolution
        kernel_h = kernel_1d.view(1, 1, -1, 1).expand(C, 1, -1, 1)
        kernel_w = kernel_1d.view(1, 1, 1, -1).expand(C, 1, 1, -1)

        # Pad and convolve
        pad_h = kernel_size // 2
        pad_w = kernel_size // 2
        padded = F.pad(image, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
        baseline = F.conv2d(padded, kernel_h, groups=C)
        baseline = F.conv2d(baseline, kernel_w, groups=C)

    elif baseline_type == "mean":
        mean_val = image.mean(dim=(2, 3), keepdim=True)
        baseline = mean_val.expand_as(image)

    elif baseline_type == "black":
        baseline = torch.zeros_like(image)

    elif baseline_type == "noise":
        baseline = torch.randn_like(image) * image.std() + image.mean()

    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")

    if squeeze:
        baseline = baseline.squeeze(0)

    return baseline


def apply_mask_to_image(
    image: torch.Tensor,
    mask: torch.Tensor,
    baseline: torch.Tensor | None = None,
    baseline_type: str = "blur",
) -> torch.Tensor:
    """
    Apply a mask to an image, replacing masked regions with baseline.

    Args:
        image: Input image, shape (C, H, W) or (B, C, H, W).
        mask: Binary mask, shape (H, W) or (B, H, W). 1 = keep, 0 = replace.
        baseline: Optional precomputed baseline. If None, computed from image.
        baseline_type: Type of baseline if not provided.

    Returns:
        Masked image with same shape as input.
    """
    if baseline is None:
        baseline = create_baseline_image(image, baseline_type)

    # Ensure mask has channel dimension
    if image.dim() == 3:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # (1, H, W)
    else:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)  # (B, 1, H, W)

    # Interpolate mask if size doesn't match
    if mask.shape[-2:] != image.shape[-2:]:
        mask = F.interpolate(
            mask.float(),
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    return image * mask + baseline * (1 - mask)


def get_top_k_mask(
    attribution: torch.Tensor,
    k: int | float,
    grid_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """
    Create a mask for the top-k attributed regions.

    Args:
        attribution: Attribution map, shape (N,) for patches or (H, W) for pixels.
        k: Number of top elements to select (int) or fraction (float 0-1).
        grid_size: If provided, reshape to spatial dimensions.

    Returns:
        Binary mask with 1s at top-k positions.
    """
    flat = attribution.flatten()
    n = len(flat)

    if isinstance(k, float):
        k = int(k * n)
    k = min(k, n)

    if k == 0:
        mask = torch.zeros_like(flat)
    else:
        _, indices = torch.topk(flat, k)
        mask = torch.zeros_like(flat)
        mask[indices] = 1.0

    if grid_size is not None:
        mask = mask.view(*grid_size)

    return mask


def get_bottom_k_mask(
    attribution: torch.Tensor,
    k: int | float,
    grid_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """
    Create a mask for the bottom-k attributed regions (inverse of top-k).

    Args:
        attribution: Attribution map.
        k: Number of bottom elements to select (int) or fraction (float 0-1).
        grid_size: If provided, reshape to spatial dimensions.

    Returns:
        Binary mask with 1s at bottom-k positions.
    """
    return get_top_k_mask(-attribution, k, grid_size)


def patches_to_image_mask(
    patch_mask: torch.Tensor,
    patch_size: int,
    img_size: int,
) -> torch.Tensor:
    """
    Convert a patch-level mask to pixel-level mask.

    Args:
        patch_mask: Mask of shape (num_patches,) or (H_p, W_p).
        patch_size: Size of each patch.
        img_size: Full image size.

    Returns:
        Pixel-level mask of shape (img_size, img_size).
    """
    grid_size = img_size // patch_size

    if patch_mask.dim() == 1:
        patch_mask = patch_mask.view(grid_size, grid_size)

    # Upsample to image size
    mask = patch_mask.unsqueeze(0).unsqueeze(0).float()  # (1, 1, H_p, W_p)
    mask = F.interpolate(
        mask,
        size=(img_size, img_size),
        mode="nearest",
    )
    return mask.squeeze()


def image_to_patches(
    image: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    """
    Convert image to patches.

    Args:
        image: Image tensor, shape (C, H, W).
        patch_size: Size of each patch.

    Returns:
        Patches of shape (num_patches, C, patch_size, patch_size).
    """
    C, H, W = image.shape
    assert H % patch_size == 0 and W % patch_size == 0

    patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    # Shape: (C, H//ps, W//ps, ps, ps)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous()
    # Shape: (H//ps, W//ps, C, ps, ps)
    patches = patches.view(-1, C, patch_size, patch_size)
    # Shape: (num_patches, C, ps, ps)
    return patches


def normalize_attribution(
    attribution: torch.Tensor,
    method: str = "minmax",
) -> torch.Tensor:
    """
    Normalize attribution map for visualization.

    Args:
        attribution: Attribution values.
        method: Normalization method:
            - "minmax": Scale to [0, 1]
            - "abs_max": Divide by max absolute value
            - "percentile": Clip outliers and scale

    Returns:
        Normalized attribution.
    """
    if method == "minmax":
        vmin, vmax = attribution.min(), attribution.max()
        if vmax - vmin > 1e-8:
            return (attribution - vmin) / (vmax - vmin)
        return torch.zeros_like(attribution)

    elif method == "abs_max":
        abs_max = attribution.abs().max()
        if abs_max > 1e-8:
            return attribution / abs_max
        return torch.zeros_like(attribution)

    elif method == "percentile":
        flat = attribution.flatten()
        p2, p98 = torch.quantile(flat, torch.tensor([0.02, 0.98]))
        clipped = torch.clamp(attribution, p2, p98)
        return normalize_attribution(clipped, method="minmax")

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def attribution_to_heatmap(
    attribution: torch.Tensor,
    cmap: str = "jet",
    normalize: bool = True,
) -> np.ndarray:
    """
    Convert attribution map to RGB heatmap.

    Args:
        attribution: Attribution values, shape (H, W).
        cmap: Matplotlib colormap name.
        normalize: Whether to normalize to [0, 1] first.

    Returns:
        RGB image as numpy array, shape (H, W, 3).
    """
    import matplotlib.pyplot as plt

    if normalize:
        attribution = normalize_attribution(attribution, "minmax")

    attr_np = attribution.cpu().numpy()
    colormap = plt.get_cmap(cmap)
    heatmap = colormap(attr_np)[:, :, :3]  # Remove alpha channel
    return (heatmap * 255).astype(np.uint8)


def overlay_heatmap(
    image: Image.Image | np.ndarray | torch.Tensor,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> Image.Image:
    """
    Overlay a heatmap on an image.

    Args:
        image: Original image (PIL, numpy, or tensor).
        heatmap: RGB heatmap as numpy array.
        alpha: Blending factor (0 = only image, 1 = only heatmap).

    Returns:
        Blended PIL Image.
    """
    # Convert image to PIL if needed
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image[0]
        if image.shape[0] in (1, 3):
            image = image.permute(1, 2, 0)
        image = image.cpu().numpy()
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        image = Image.fromarray(image)
    elif isinstance(image, np.ndarray):
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)

    # Resize heatmap to match image
    heatmap_pil = Image.fromarray(heatmap)
    if heatmap_pil.size != image.size:
        heatmap_pil = heatmap_pil.resize(image.size, Image.BILINEAR)

    # Blend
    return Image.blend(image.convert("RGB"), heatmap_pil, alpha)
