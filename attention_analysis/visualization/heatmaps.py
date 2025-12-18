"""
Heatmap generation and overlay functions.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

import matplotlib.pyplot as plt


def _fig_to_array(fig) -> np.ndarray:
    """Convert matplotlib figure to numpy array (compatible with all versions)."""
    fig.canvas.draw()
    if hasattr(fig.canvas, "buffer_rgba"):
        buf = np.asarray(fig.canvas.buffer_rgba())
        return buf[..., :3]  # RGBA -> RGB
    # Fallback for older matplotlib
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))


def attribution_to_heatmap(
    attribution: torch.Tensor | np.ndarray,
    cmap: str = "jet",
    normalize: bool = True,
) -> np.ndarray:
    """
    Convert attribution map to RGB heatmap.

    Args:
        attribution: Attribution values, shape (H, W).
        cmap: Matplotlib colormap name.
        normalize: Whether to normalize to [0, 1].

    Returns:
        RGB heatmap as numpy array, shape (H, W, 3), values in [0, 255].
    """
    if isinstance(attribution, torch.Tensor):
        attr = attribution.detach().cpu().numpy()
    else:
        attr = attribution.copy()

    if normalize:
        vmin, vmax = attr.min(), attr.max()
        if vmax - vmin > 1e-8:
            attr = (attr - vmin) / (vmax - vmin)
        else:
            attr = np.zeros_like(attr)

    colormap = plt.get_cmap(cmap)
    heatmap = colormap(attr)[:, :, :3]  # Remove alpha
    return (heatmap * 255).astype(np.uint8)


def overlay_attribution(
    image: Image.Image | np.ndarray | torch.Tensor,
    attribution: torch.Tensor | np.ndarray,
    alpha: float = 0.5,
    cmap: str = "jet",
    resize: bool = True,
) -> Image.Image:
    """
    Overlay attribution heatmap on an image.

    Args:
        image: Original image.
        attribution: Attribution map.
        alpha: Blending factor (0 = only image, 1 = only heatmap).
        cmap: Colormap for heatmap.
        resize: Whether to resize heatmap to match image.

    Returns:
        Blended PIL Image.
    """
    # Convert image to PIL
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image[0]
        if image.shape[0] in (1, 3):
            image = image.permute(1, 2, 0)
        image = image.detach().cpu().numpy()
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        image = Image.fromarray(image)
    elif isinstance(image, np.ndarray):
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)

    # Convert attribution to heatmap
    heatmap = attribution_to_heatmap(attribution, cmap=cmap)
    heatmap_pil = Image.fromarray(heatmap)

    # Resize if needed
    if resize and heatmap_pil.size != image.size:
        heatmap_pil = heatmap_pil.resize(image.size, Image.BILINEAR)

    # Blend
    return Image.blend(image.convert("RGB"), heatmap_pil, alpha)


def create_side_by_side(
    image: Image.Image,
    attribution: torch.Tensor | np.ndarray,
    cmap: str = "jet",
    labels: tuple[str, str, str] = ("Original", "Attribution", "Overlay"),
    figsize: tuple[int, int] = (12, 4),
) -> Image.Image:
    """
    Create side-by-side visualization: original, heatmap, overlay.

    Args:
        image: Original image.
        attribution: Attribution map.
        cmap: Colormap for heatmap.
        labels: Labels for the three panels.
        figsize: Figure size.

    Returns:
        Combined PIL Image.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Original
    axes[0].imshow(image)
    axes[0].set_title(labels[0])
    axes[0].axis("off")

    # Attribution heatmap
    if isinstance(attribution, torch.Tensor):
        attr = attribution.detach().cpu().numpy()
    else:
        attr = attribution
    im = axes[1].imshow(attr, cmap=cmap)
    axes[1].set_title(labels[1])
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Overlay
    overlay = overlay_attribution(image, attribution, cmap=cmap)
    axes[2].imshow(overlay)
    axes[2].set_title(labels[2])
    axes[2].axis("off")

    plt.tight_layout()

    # Convert to PIL
    img_array = _fig_to_array(fig)
    plt.close(fig)

    return Image.fromarray(img_array)


def generate_attribution_grid(
    images: torch.Tensor | list[Image.Image],
    attributions: torch.Tensor | list[torch.Tensor],
    method_name: str = "",
    cmap: str = "jet",
    max_images: int = 8,
    image_size: int = 128,
) -> Image.Image:
    """
    Generate a grid of attribution overlays.

    Args:
        images: Batch of images.
        attributions: Batch of attribution maps.
        method_name: Name of attribution method (for title).
        cmap: Colormap.
        max_images: Maximum number of images to show.
        image_size: Size to resize each image.

    Returns:
        Grid image.
    """
    # Convert inputs to lists
    if isinstance(images, torch.Tensor):
        if images.dim() == 4:
            images = [images[i] for i in range(min(len(images), max_images))]
        else:
            images = [images]

    if isinstance(attributions, torch.Tensor):
        if attributions.dim() == 3:
            attributions = [
                attributions[i] for i in range(min(len(attributions), max_images))
            ]
        else:
            attributions = [attributions]

    n = min(len(images), len(attributions), max_images)

    # Create overlays
    overlays = []
    for i in range(n):
        img = images[i]
        attr = attributions[i]

        # Convert image
        if isinstance(img, torch.Tensor):
            if img.shape[0] in (1, 3):
                img = img.permute(1, 2, 0)
            img = img.detach().cpu().numpy()
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img)

        # Create overlay
        overlay = overlay_attribution(img, attr, cmap=cmap)
        overlay = overlay.resize((image_size, image_size))
        overlays.append(overlay)

    # Create grid
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    grid_width = cols * image_size
    grid_height = rows * image_size
    grid = Image.new("RGB", (grid_width, grid_height))

    for i, overlay in enumerate(overlays):
        row = i // cols
        col = i % cols
        grid.paste(overlay, (col * image_size, row * image_size))

    return grid


def generate_per_layer_heatmaps(
    layer_attributions: list[torch.Tensor],
    figsize: tuple[int, int] | None = None,
    cmap: str = "jet",
) -> Image.Image:
    """
    Generate heatmaps for each layer's attribution.

    Args:
        layer_attributions: List of attribution maps, one per layer.
        figsize: Figure size. Auto-computed if None.
        cmap: Colormap.

    Returns:
        Combined PIL Image with all layer heatmaps.
    """
    n_layers = len(layer_attributions)
    cols = min(n_layers, 4)
    rows = (n_layers + cols - 1) // cols

    if figsize is None:
        figsize = (cols * 3, rows * 3)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).flatten()

    for i, attr in enumerate(layer_attributions):
        if isinstance(attr, torch.Tensor):
            attr = attr.detach().cpu().numpy()

        im = axes[i].imshow(attr, cmap=cmap)
        axes[i].set_title(f"Layer {i}")
        axes[i].axis("off")
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    # Hide unused axes
    for i in range(n_layers, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    # Convert to PIL
    img_array = _fig_to_array(fig)
    plt.close(fig)

    return Image.fromarray(img_array)


def generate_per_head_heatmaps(
    head_attributions: torch.Tensor,
    layer_idx: int = 0,
    figsize: tuple[int, int] | None = None,
    cmap: str = "jet",
) -> Image.Image:
    """
    Generate heatmaps for each attention head's attribution.

    Args:
        head_attributions: Attribution maps per head, shape (H, h, w).
        layer_idx: Layer index for title.
        figsize: Figure size.
        cmap: Colormap.

    Returns:
        Combined PIL Image with all head heatmaps.
    """
    if isinstance(head_attributions, torch.Tensor):
        head_attributions = head_attributions.detach().cpu()

    n_heads = head_attributions.shape[0]
    cols = min(n_heads, 4)
    rows = (n_heads + cols - 1) // cols

    if figsize is None:
        figsize = (cols * 3, rows * 3)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).flatten()

    for i in range(n_heads):
        attr = head_attributions[i].numpy()
        axes[i].imshow(attr, cmap=cmap)
        axes[i].set_title(f"L{layer_idx} Head {i}")
        axes[i].axis("off")

    for i in range(n_heads, len(axes)):
        axes[i].axis("off")

    plt.suptitle(f"Layer {layer_idx} Attention Heads")
    plt.tight_layout()

    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return Image.fromarray(img_array)
