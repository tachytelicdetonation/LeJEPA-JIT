"""
Visualization utilities for training and attention dynamics.

Includes:
- PCA visualization of patch embeddings
- Attention rollout and raw attention maps
- Layer-wise attention evolution
- Per-head attention patterns
- Gradient-weighted attention (GMAR-style)
- Head importance/diversity visualization
- Token similarity heatmaps
- Representation similarity matrices (RSM)
- Gradient flow heatmaps
- t-SNE/UMAP embedding projections
- Attention difference tracking

References:
- AttnLRP (arXiv:2402.05602): Attribution methods for transformers
- GMAR (arXiv:2504.19414): Gradient-weighted Multi-head Attention Rollout
"""

import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF
from typing import Optional, List, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server use


@torch.no_grad()
def compute_pca_projection(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Project embeddings to 3 RGB channels using PCA.

    Args:
        embeddings: (N, D) tensor

    Returns:
        (N, 3) tensor
    """
    # Center the data
    mean = embeddings.mean(dim=0, keepdim=True)
    centered = embeddings - mean

    # Compute PCA via SVD
    # U, S, V = torch.svd(centered) # Deprecated
    U, S, V = torch.linalg.svd(centered, full_matrices=False)

    # Project to top 3 components
    # shape of V is (D, D) (or similar depending on implementation details, Vh is returned by linalg.svd usually)
    # torch.linalg.svd returns U, S, Vh
    # Vh is (D, D), we want top 3 rows of Vh which correspond to top 3 eigenvectors
    components = V[:3]  # (3, D)

    projected = torch.matmul(centered, components.T)  # (N, D) @ (D, 3) -> (N, 3)

    # Serialize to [0, 1] for visualization
    # Min-max normalization per channel
    low = projected.min(dim=0, keepdim=True)[0]
    high = projected.max(dim=0, keepdim=True)[0]

    projected = (projected - low) / (high - low + 1e-6)

    return projected


@torch.no_grad()
def generate_pca_visualization(
    model,
    images: torch.Tensor,
    device: torch.device,
    img_size: int = 128,
    patch_size: int = 8,
) -> Image.Image:
    """
    Generate a grid of Original vs PCA visualizations.

    Args:
        model: LeJEPA model
        images: (B, C, H, W) input images (normalized)
        device: torch device
        img_size: Input image size
        patch_size: Patch size

    Returns:
        PIL Image
    """
    model.eval()

    # Hook to capture features
    features = []

    def hook_fn(module, input, output):
        features.append(output)

    # Hook on the final norm of the encoder (before pooling)
    handle = model.encoder.norm.register_forward_hook(hook_fn)

    # Forward pass
    images = images.to(device)
    _ = model.encoder(images)

    handle.remove()

    # feats: (B, N, D)
    feats = features[0]

    # Remove CLS token if present
    if model.encoder.pool == "cls":
        feats = feats[:, 1:]  # Drop first token

    B, N, D = feats.shape
    h = w = int(N**0.5)

    # Flatten all patches from all images to compute a global PCA
    flat_feats = feats.reshape(-1, D)  # (B*N, D)

    # Compute PCA
    pca_feats = compute_pca_projection(flat_feats)  # (B*N, 3)

    # Reshape back to (B, h, w, 3)
    pca_maps = pca_feats.reshape(B, h, w, 3)

    vis_imgs = []

    for i in range(B):
        # 1. Original Image
        img = images[i].detach().cpu()  # (3, H, W)
        # Min-max scale to 0-1
        img = (img - img.min()) / (img.max() - img.min())
        img = TF.to_pil_image(img).resize((img_size, img_size))

        # 2. PCA Map
        pca = pca_maps[i].detach().cpu().numpy()  # (h, w, 3)
        pca = (pca * 255).astype(np.uint8)
        pca_img = Image.fromarray(pca).resize(
            (img_size, img_size), resample=Image.NEAREST
        )

        # Combine vertical
        combined = Image.new("RGB", (img_size, img_size * 2))
        combined.paste(img, (0, 0))
        combined.paste(pca_img, (0, img_size))

        vis_imgs.append(combined)

    # Combine all into a horizontal grid
    grid_width = img_size * B
    grid_height = img_size * 2
    grid = Image.new("RGB", (grid_width, grid_height))

    for i, img in enumerate(vis_imgs):
        grid.paste(img, (i * img_size, 0))

    return grid


@torch.no_grad()
def generate_attention_rollout(
    model,
    images: torch.Tensor,
    device: torch.device,
    img_size: int = 128,
    patch_size: int = 8,
    head_fusion: str = "mean",
    discard_ratio: float = 0.9,
) -> Image.Image:
    """
    Generate Attention Rollout visualization.

    Args:
        model: LeJEPA model
        images: (B, C, H, W) input images
        device: torch device
        img_size: Image size
        patch_size: Patch size
        head_fusion: 'mean', 'max', or 'min' for fusing attention heads
        discard_ratio: Ratio of lower attention values to discard (for clarity)

    Returns:
        PIL Image grid
    """
    model.eval()

    # Enable attention output
    for block in model.encoder.blocks:
        block.attn.output_attention = True

    # Forward pass
    images = images.to(device)
    _ = model.encoder(images)

    # Collect attention maps: List of (B, Heads, N, N)
    attentions = model.encoder.get_attention_maps()

    # Disable attention output
    for block in model.encoder.blocks:
        block.attn.output_attention = False

    if not attentions:
        return Image.new("RGB", (img_size, img_size))

    # Process Attention Rollout
    # Start with Identity matrix
    B, Heads, N, _ = attentions[0].shape
    w, h = images.shape[2] // patch_size, images.shape[3] // patch_size

    # We want to see how CLS token attends to all other tokens
    # Or if no CLS, how a central token/all tokens attend

    # If explicit CLS token exists:
    has_cls = model.encoder.pool == "cls"

    # Initialize rollout with identity matrix
    # rollout: (B, N, N)
    rollout = torch.eye(N).to(device).unsqueeze(0).repeat(B, 1, 1)

    for attn in attentions:
        # Fuse heads
        if head_fusion == "mean":
            attn_fused = attn.mean(dim=1)  # (B, N, N)
        elif head_fusion == "max":
            attn_fused = attn.max(dim=1)[0]
        elif head_fusion == "min":
            attn_fused = attn.min(dim=1)[0]

        # To avoid noise, we can discard low attention
        # flat = attn_fused.view(B, -1)
        # val, idx = torch.sort(flat)
        # val /= torch.sum(val, dim=1, keepdim=True)
        # cumsum = torch.cumsum(val, dim=1)
        # mask = cumsum < discard_ratio
        # This is complex to batchify efficiently without scatter.
        # Standard rollout usually just does matrix multiplication

        # Add residual connection (identity) and re-normalize?
        # A_hat = 0.5 * A + 0.5 * I
        attn_fused = 0.5 * attn_fused + 0.5 * torch.eye(N).to(device).unsqueeze(0)

        # Normalize rows to sum to 1? Softmax already did that.
        # But adding identity makes sum > 1. Renormalize.
        attn_fused = attn_fused / attn_fused.sum(dim=-1, keepdim=True)

        # Rollout accumulation
        rollout = torch.matmul(attn_fused, rollout)

    # Extract map
    # We generally want to see what the [CLS] token attends to (if exists).
    # If no CLS (Global Mean Pooling), we can look at the average attention of all tokens?
    # Or just fake a CLS behavior by averaging the rollout rows.

    if has_cls:
        # CLS token is at index 0 (usually?)
        # JiTEncoder with CLS: self.cls_token + x.
        # Check if CLS was prepended.
        # jit_encoder.py: `if self.pool == "cls": x = torch.cat([cls_tokens, x], dim=1)` -> Index 0 is CLS.

        # CLS attention map is row 0 of the rollout matrix: It says how much CLS attends to everyone else.
        mask = rollout[:, 0, 1:]  # (B, N-1) - skip self-attention to CLS

    else:
        # No CLS (Mean pooling).
        # We want to see "saliency".
        # Maybe average attention of all output tokens to input tokens?
        # rollout is (B, N, N).
        # We want (B, N) map.
        # Average over the output tokens (rows)?
        mask = rollout.mean(dim=1)  # (B, N)

    # Reshape to grid
    # mask: (B, num_patches)
    mask = mask.reshape(B, h, w)

    vis_imgs = []

    for i in range(B):
        # 1. Original Image
        img = images[i].detach().cpu()  # (3, H, W)
        img = (img - img.min()) / (img.max() - img.min())
        img_pil = TF.to_pil_image(img).resize((img_size, img_size))

        # 2. Attention Map
        m = mask[i].detach().cpu().numpy()
        m = (m - m.min()) / (m.max() - m.min() + 1e-6)

        # Colorize
        m = cv2.applyColorMap(np.uint8(255 * m), cv2.COLORMAP_JET)
        m = cv2.cvtColor(m, cv2.COLOR_BGR2RGB)

        # Resize to image size
        m = Image.fromarray(m).resize((img_size, img_size), resample=Image.BILINEAR)

        # Overlay
        # alpha blend
        overlay = Image.blend(img_pil, m, alpha=0.5)

        # Combine vertical: Original, Attention Overlay
        combined = Image.new("RGB", (img_size, img_size * 2))
        combined.paste(img_pil, (0, 0))
        combined.paste(overlay, (0, img_size))

        vis_imgs.append(combined)

    # Grid
    grid_width = img_size * B
    grid_height = img_size * 2
    grid = Image.new("RGB", (grid_width, grid_height))

    for i, img in enumerate(vis_imgs):
        grid.paste(img, (i * img_size, 0))

    return grid


@torch.no_grad()
def generate_attention_grid(
    model,
    images: torch.Tensor,
    device: torch.device,
    img_size: int = 128,
    patch_size: int = 8,
) -> Image.Image:
    """
    Generate grid of raw attention maps (last layer, mean head).

    Args:
        model: LeJEPA model
        images: (B, C, H, W) input images
        device: torch device
        img_size: Image size
        patch_size: Patch size

    Returns:
        PIL Image grid
    """
    model.eval()

    # Enable attention output
    for block in model.encoder.blocks:
        block.attn.output_attention = True

    # Forward pass
    images = images.to(device)
    _ = model.encoder(images)

    # Collect attention maps
    attentions = model.encoder.get_attention_maps()

    # Disable attention output
    for block in model.encoder.blocks:
        block.attn.output_attention = False

    if not attentions:
        return Image.new("RGB", (img_size, img_size))

    # Get last layer attention: (B, Heads, N, N)
    last_attn = attentions[-1]
    B, Heads, N, _ = last_attn.shape
    w, h = images.shape[2] // patch_size, images.shape[3] // patch_size

    # Mean over heads
    attn_mean = last_attn.mean(dim=1)  # (B, N, N)

    # Extract map (CLS or Mean)
    has_cls = model.encoder.pool == "cls"

    if has_cls:
        mask = attn_mean[:, 0, 1:]  # (B, N-1)
    else:
        # For mean pooling, we want to see how much each token contributes on average to others?
        # Or just average row attention.
        mask = attn_mean.mean(dim=1)  # (B, N)

    mask = mask.reshape(B, h, w)

    vis_imgs = []

    for i in range(B):
        # 1. Original Image
        img = images[i].detach().cpu()
        img = (img - img.min()) / (img.max() - img.min())
        img_pil = TF.to_pil_image(img).resize((img_size, img_size))

        # 2. Attention Map
        m = mask[i].detach().cpu().numpy()
        m = (m - m.min()) / (m.max() - m.min() + 1e-6)

        # Use Viridis for raw attention to distinguish from Rollout (Jet)?
        m = cv2.applyColorMap(np.uint8(255 * m), cv2.COLORMAP_VIRIDIS)
        m = cv2.cvtColor(m, cv2.COLOR_BGR2RGB)

        m = Image.fromarray(m).resize((img_size, img_size), resample=Image.BILINEAR)

        overlay = Image.blend(img_pil, m, alpha=0.6)

        combined = Image.new("RGB", (img_size, img_size * 2))
        combined.paste(img_pil, (0, 0))
        combined.paste(overlay, (0, img_size))
        vis_imgs.append(combined)

    grid_width = img_size * B
    grid_height = img_size * 2
    grid = Image.new("RGB", (grid_width, grid_height))

    for i, img in enumerate(vis_imgs):
        grid.paste(img, (i * img_size, 0))

    return grid


# =============================================================================
# Layer-wise Attention Evolution
# =============================================================================


@torch.no_grad()
def generate_layer_attention_evolution(
    model,
    images: torch.Tensor,
    device: torch.device,
    img_size: int = 128,
    patch_size: int = 8,
    sample_idx: int = 0,
) -> Image.Image:
    """
    Visualize how attention patterns evolve across layers.

    Creates a grid showing attention from CLS/mean token at each layer,
    demonstrating how the model builds up its understanding.

    Args:
        model: LeJEPA model
        images: (B, C, H, W) input images
        device: torch device
        img_size: Image size
        patch_size: Patch size
        sample_idx: Which sample to visualize

    Returns:
        PIL Image showing attention at each layer
    """
    model.eval()

    # Enable attention output
    for block in model.encoder.blocks:
        block.attn.output_attention = True

    images = images.to(device)
    _ = model.encoder(images)

    attentions = model.encoder.get_attention_maps()

    for block in model.encoder.blocks:
        block.attn.output_attention = False

    if not attentions:
        return Image.new("RGB", (img_size, img_size))

    num_layers = len(attentions)
    has_cls = model.encoder.pool == "cls"
    h = w = images.shape[2] // patch_size

    # Original image
    img = images[sample_idx].detach().cpu()
    img = (img - img.min()) / (img.max() - img.min())
    img_pil = TF.to_pil_image(img).resize((img_size, img_size))

    # Create grid: Original + each layer
    cols = min(6, num_layers + 1)
    rows = (num_layers + 1 + cols - 1) // cols

    grid_width = cols * img_size
    grid_height = rows * img_size
    grid = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))

    # Paste original
    grid.paste(img_pil, (0, 0))

    # Add layer attentions
    for layer_idx, attn in enumerate(attentions):
        # attn: (B, H, N, N)
        attn_sample = attn[sample_idx].mean(dim=0)  # (N, N), mean over heads

        if has_cls:
            mask = attn_sample[0, 1:].reshape(h, w)  # CLS attention to patches
        else:
            mask = attn_sample.mean(dim=0).reshape(h, w)

        mask = mask.detach().cpu().numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)

        # Colorize with layer-specific color progression
        # Early layers: blue, late layers: red
        progress = layer_idx / max(1, num_layers - 1)
        cmap = cv2.COLORMAP_JET if progress > 0.5 else cv2.COLORMAP_COOL

        m = cv2.applyColorMap(np.uint8(255 * mask), cmap)
        m = cv2.cvtColor(m, cv2.COLOR_BGR2RGB)
        m = Image.fromarray(m).resize((img_size, img_size), resample=Image.BILINEAR)

        overlay = Image.blend(img_pil, m, alpha=0.5)

        # Add layer label
        draw = ImageDraw.Draw(overlay)
        draw.text((5, 5), f"L{layer_idx}", fill=(255, 255, 255))

        pos_x = ((layer_idx + 1) % cols) * img_size
        pos_y = ((layer_idx + 1) // cols) * img_size
        grid.paste(overlay, (pos_x, pos_y))

    return grid


# =============================================================================
# Per-Head Attention Visualization
# =============================================================================


@torch.no_grad()
def generate_per_head_attention(
    model,
    images: torch.Tensor,
    device: torch.device,
    img_size: int = 96,
    patch_size: int = 8,
    sample_idx: int = 0,
    layer_idx: int = -1,
) -> Image.Image:
    """
    Visualize attention patterns for each head individually.

    Shows what different attention heads are learning - some may focus
    on edges, textures, objects, or background.

    Args:
        model: LeJEPA model
        images: (B, C, H, W) input images
        device: torch device
        img_size: Per-head image size
        patch_size: Patch size
        sample_idx: Which sample to visualize
        layer_idx: Which layer (-1 for last)

    Returns:
        PIL Image grid of per-head attention
    """
    model.eval()

    for block in model.encoder.blocks:
        block.attn.output_attention = True

    images = images.to(device)
    _ = model.encoder(images)

    attentions = model.encoder.get_attention_maps()

    for block in model.encoder.blocks:
        block.attn.output_attention = False

    if not attentions:
        return Image.new("RGB", (img_size, img_size))

    attn = attentions[layer_idx]  # (B, H, N, N)
    B, num_heads, N, _ = attn.shape
    has_cls = model.encoder.pool == "cls"
    h = w = images.shape[2] // patch_size

    # Original image
    img = images[sample_idx].detach().cpu()
    img = (img - img.min()) / (img.max() - img.min())
    img_pil = TF.to_pil_image(img).resize((img_size, img_size))

    # Grid layout for heads
    cols = min(4, num_heads)
    rows = (num_heads + cols - 1) // cols

    grid_width = (cols + 1) * img_size  # +1 for original
    grid_height = rows * img_size
    grid = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))

    # Paste original in center-left
    grid.paste(img_pil, (0, (rows * img_size - img_size) // 2))

    for head_idx in range(num_heads):
        attn_head = attn[sample_idx, head_idx]  # (N, N)

        if has_cls:
            mask = attn_head[0, 1:].reshape(h, w)
        else:
            mask = attn_head.mean(dim=0).reshape(h, w)

        mask = mask.detach().cpu().numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)

        m = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_VIRIDIS)
        m = cv2.cvtColor(m, cv2.COLOR_BGR2RGB)
        m = Image.fromarray(m).resize((img_size, img_size), resample=Image.BILINEAR)

        overlay = Image.blend(img_pil, m, alpha=0.6)

        # Add head label
        draw = ImageDraw.Draw(overlay)
        draw.text((5, 5), f"H{head_idx}", fill=(255, 255, 255))

        col = (head_idx % cols) + 1
        row = head_idx // cols
        grid.paste(overlay, (col * img_size, row * img_size))

    return grid


# =============================================================================
# Gradient-Weighted Attention (GMAR-style)
# =============================================================================


def generate_gradient_weighted_attention(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    probe: torch.nn.Module,
    device: torch.device,
    img_size: int = 128,
    patch_size: int = 8,
) -> Image.Image:
    """
    Generate GMAR-style gradient-weighted attention visualization.

    Weights attention maps by the gradient of the classification loss,
    highlighting regions that are most important for the prediction.

    Reference: GMAR (arXiv:2504.19414)

    Args:
        model: LeJEPA model
        images: (B, C, H, W) input images
        labels: (B,) class labels
        probe: Linear probe for classification
        device: torch device
        img_size: Image size
        patch_size: Patch size

    Returns:
        PIL Image comparing raw vs gradient-weighted attention
    """
    model.eval()
    probe.eval()

    # Enable attention output and gradients
    for block in model.encoder.blocks:
        block.attn.output_attention = True

    images = images.to(device).requires_grad_(True)
    labels = labels.to(device)

    # Forward pass
    emb, proj = model(images.unsqueeze(1))  # Add view dimension
    logits = probe(emb)

    # Get prediction confidence
    probs = torch.softmax(logits, dim=1)
    target_probs = probs.gather(1, labels.unsqueeze(1)).squeeze()

    # Backward to get gradients
    loss = -target_probs.sum()  # Maximize correct class probability
    loss.backward()

    attentions = model.encoder.get_attention_maps()

    for block in model.encoder.blocks:
        block.attn.output_attention = False

    if not attentions:
        return Image.new("RGB", (img_size, img_size))

    B = images.shape[0]
    has_cls = model.encoder.pool == "cls"
    h = w = images.shape[2] // patch_size

    vis_imgs = []

    for i in range(min(B, 4)):  # Limit to 4 images
        img = images[i].detach().cpu()
        img = (img - img.min()) / (img.max() - img.min())
        img_pil = TF.to_pil_image(img).resize((img_size, img_size))

        # Compute gradient-weighted attention (GMAR)
        # Weight each head by gradient magnitude
        rollout_raw = torch.eye(attentions[0].shape[2]).to(device).unsqueeze(0)
        rollout_grad = torch.eye(attentions[0].shape[2]).to(device).unsqueeze(0)

        for layer_idx, attn in enumerate(attentions):
            attn_i = attn[i]  # (H, N, N)

            # Raw attention (mean over heads)
            attn_raw = attn_i.mean(dim=0)

            # Gradient-weighted attention
            # Weight heads by their gradient norm (importance)
            if attn_i.grad is not None:
                head_weights = attn_i.grad.norm(dim=(1, 2))
                head_weights = head_weights / (head_weights.sum() + 1e-8)
                attn_grad = (attn_i * head_weights.view(-1, 1, 1)).sum(dim=0)
            else:
                attn_grad = attn_raw

            # Add residual and normalize
            attn_raw = 0.5 * attn_raw + 0.5 * torch.eye(attn_raw.shape[0]).to(device)
            attn_grad = 0.5 * attn_grad + 0.5 * torch.eye(attn_grad.shape[0]).to(device)

            attn_raw = attn_raw / attn_raw.sum(dim=-1, keepdim=True)
            attn_grad = attn_grad / attn_grad.sum(dim=-1, keepdim=True)

            rollout_raw = torch.matmul(attn_raw.unsqueeze(0), rollout_raw)
            rollout_grad = torch.matmul(attn_grad.unsqueeze(0), rollout_grad)

        # Extract masks
        if has_cls:
            mask_raw = rollout_raw[0, 0, 1:].reshape(h, w)
            mask_grad = rollout_grad[0, 0, 1:].reshape(h, w)
        else:
            mask_raw = rollout_raw[0].mean(dim=0).reshape(h, w)
            mask_grad = rollout_grad[0].mean(dim=0).reshape(h, w)

        # Normalize
        for mask, cmap, label in [
            (mask_raw, cv2.COLORMAP_JET, "Raw"),
            (mask_grad, cv2.COLORMAP_HOT, "Grad"),
        ]:
            mask_np = mask.detach().cpu().numpy()
            mask_np = (mask_np - mask_np.min()) / (mask_np.max() - mask_np.min() + 1e-6)

            m = cv2.applyColorMap(np.uint8(255 * mask_np), cmap)
            m = cv2.cvtColor(m, cv2.COLOR_BGR2RGB)
            m = Image.fromarray(m).resize((img_size, img_size), resample=Image.BILINEAR)

            overlay = Image.blend(img_pil, m, alpha=0.5)
            draw = ImageDraw.Draw(overlay)
            draw.text((5, 5), label, fill=(255, 255, 255))
            vis_imgs.append(overlay)

    # Grid: pairs of (Raw, Grad) for each image
    num_pairs = len(vis_imgs) // 2
    grid_width = 2 * img_size
    grid_height = num_pairs * img_size
    grid = Image.new("RGB", (grid_width, grid_height))

    for idx, img in enumerate(vis_imgs):
        col = idx % 2
        row = idx // 2
        grid.paste(img, (col * img_size, row * img_size))

    return grid


# =============================================================================
# Head Importance Visualization
# =============================================================================


@torch.no_grad()
def generate_head_importance_heatmap(
    model,
    images: torch.Tensor,
    device: torch.device,
    figsize: Tuple[int, int] = (10, 6),
) -> Image.Image:
    """
    Visualize head importance across layers using attention statistics.

    Creates a heatmap showing which heads are most active/important,
    useful for understanding head redundancy and specialization.

    Args:
        model: LeJEPA model
        images: (B, C, H, W) input images
        device: torch device
        figsize: Figure size

    Returns:
        PIL Image of heatmap
    """
    model.eval()

    for block in model.encoder.blocks:
        block.attn.output_attention = True

    images = images.to(device)
    _ = model.encoder(images)

    attentions = model.encoder.get_attention_maps()

    for block in model.encoder.blocks:
        block.attn.output_attention = False

    if not attentions:
        return Image.new("RGB", (400, 300))

    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]

    # Compute head importance metrics
    # 1. Entropy (uniformity of attention)
    # 2. Max attention (focus intensity)
    # 3. Variance (diversity of patterns)

    entropy_map = np.zeros((num_layers, num_heads))
    max_attn_map = np.zeros((num_layers, num_heads))
    variance_map = np.zeros((num_layers, num_heads))

    for layer_idx, attn in enumerate(attentions):
        # attn: (B, H, N, N)
        attn_np = attn.detach().cpu().numpy()

        for head_idx in range(num_heads):
            head_attn = attn_np[:, head_idx]  # (B, N, N)

            # Entropy
            eps = 1e-8
            entropy = -np.sum(head_attn * np.log(head_attn + eps), axis=-1).mean()
            entropy_map[layer_idx, head_idx] = entropy

            # Max attention
            max_attn = head_attn.max(axis=-1).mean()
            max_attn_map[layer_idx, head_idx] = max_attn

            # Variance
            variance = head_attn.var()
            variance_map[layer_idx, head_idx] = variance

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    titles = [
        "Entropy (↑=uniform)",
        "Max Attention (↑=focused)",
        "Variance (↑=diverse)",
    ]
    maps = [entropy_map, max_attn_map, variance_map]
    cmaps = ["coolwarm", "hot", "viridis"]

    for ax, title, data, cmap in zip(axes, titles, maps, cmaps):
        im = ax.imshow(data, aspect="auto", cmap=cmap)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    # Convert to PIL
    fig.canvas.draw()
    img = Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )
    plt.close(fig)

    return img


# =============================================================================
# Token Similarity Heatmap
# =============================================================================


@torch.no_grad()
def generate_token_similarity_heatmap(
    model,
    images: torch.Tensor,
    device: torch.device,
    sample_idx: int = 0,
    figsize: Tuple[int, int] = (8, 8),
) -> Image.Image:
    """
    Visualize pairwise similarity between patch tokens.

    Shows which patches have similar representations, useful for
    understanding semantic grouping learned by the model.

    Args:
        model: LeJEPA model
        images: (B, C, H, W) input images
        device: torch device
        sample_idx: Which sample to visualize
        figsize: Figure size

    Returns:
        PIL Image of similarity matrix
    """
    model.eval()

    features = []

    def hook_fn(module, input, output):
        features.append(output)

    handle = model.encoder.norm.register_forward_hook(hook_fn)

    images = images.to(device)
    _ = model.encoder(images)

    handle.remove()

    feats = features[0]  # (B, N, D)

    if model.encoder.pool == "cls":
        feats = feats[:, 1:]  # Remove CLS

    # Get features for sample
    feat = feats[sample_idx]  # (N, D)

    # Compute cosine similarity
    feat_norm = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
    sim = torch.mm(feat_norm, feat_norm.T)  # (N, N)
    sim = sim.detach().cpu().numpy()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(sim, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xlabel("Patch Token")
    ax.set_ylabel("Patch Token")
    ax.set_title("Patch Token Cosine Similarity")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    fig.canvas.draw()
    img = Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )
    plt.close(fig)

    return img


# =============================================================================
# Representation Similarity Matrix (RSM)
# =============================================================================


@torch.no_grad()
def generate_rsm_across_layers(
    model,
    images: torch.Tensor,
    device: torch.device,
    figsize: Tuple[int, int] = (12, 4),
) -> Image.Image:
    """
    Generate Representational Similarity Matrix (RSM) across layers.

    Shows how representations evolve through the network by comparing
    sample similarities at different layers.

    Args:
        model: LeJEPA model
        images: (B, C, H, W) input images
        device: torch device
        figsize: Figure size

    Returns:
        PIL Image showing RSM at different layers
    """
    model.eval()

    # Collect features from each layer
    layer_features = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            layer_features.append((layer_idx, output))

        return hook_fn

    handles = []
    for idx, block in enumerate(model.encoder.blocks):
        handle = block.register_forward_hook(make_hook(idx))
        handles.append(handle)

    images = images.to(device)
    _ = model.encoder(images)

    for handle in handles:
        handle.remove()

    if not layer_features:
        return Image.new("RGB", (400, 300))

    # Sort by layer index
    layer_features.sort(key=lambda x: x[0])

    # Select layers to visualize (first, middle, last)
    num_layers = len(layer_features)
    selected_indices = [0, num_layers // 2, num_layers - 1]

    fig, axes = plt.subplots(1, len(selected_indices), figsize=figsize)

    for ax_idx, layer_idx in enumerate(selected_indices):
        feat = layer_features[layer_idx][1]  # (B, N, D)

        # Global average pooling or CLS
        if model.encoder.pool == "cls":
            feat_pooled = feat[:, 0]  # (B, D)
        else:
            feat_pooled = feat.mean(dim=1)  # (B, D)

        # Cosine similarity between samples
        feat_norm = feat_pooled / (feat_pooled.norm(dim=-1, keepdim=True) + 1e-8)
        rsm = torch.mm(feat_norm, feat_norm.T).detach().cpu().numpy()

        im = axes[ax_idx].imshow(rsm, cmap="coolwarm", vmin=-1, vmax=1)
        axes[ax_idx].set_xlabel("Sample")
        axes[ax_idx].set_ylabel("Sample")
        axes[ax_idx].set_title(f"Layer {layer_idx}")

    plt.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)
    plt.suptitle("Representational Similarity Matrix (RSM) Across Layers")
    plt.tight_layout()

    fig.canvas.draw()
    img = Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )
    plt.close(fig)

    return img


# =============================================================================
# Gradient Flow Heatmap
# =============================================================================


def generate_gradient_flow_heatmap(
    gradient_stats: Dict[str, Dict[str, float]],
    figsize: Tuple[int, int] = (10, 6),
) -> Image.Image:
    """
    Visualize gradient flow through the network.

    Shows gradient magnitude at each layer, useful for detecting
    vanishing/exploding gradients.

    Args:
        gradient_stats: Dict from compute_layer_gradient_stats()
        figsize: Figure size

    Returns:
        PIL Image of gradient flow
    """
    if not gradient_stats:
        return Image.new("RGB", (400, 300))

    # Sort layers
    layers = sorted(gradient_stats.keys())

    # Extract metrics
    norms = [gradient_stats[layer].get("norm", 0) for layer in layers]
    abs_means = [gradient_stats[layer].get("abs_mean", 0) for layer in layers]
    stds = [gradient_stats[layer].get("std", 0) for layer in layers]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Create tick labels
    tick_labels = [name.replace("blocks.", "L") for name in layers]

    # Gradient norm
    axes[0].barh(range(len(layers)), norms, color="steelblue")
    axes[0].set_yticks(range(len(layers)))
    axes[0].set_yticklabels(tick_labels, fontsize=8)
    axes[0].set_xlabel("Gradient Norm")
    axes[0].set_title("Gradient Magnitude")

    # Absolute mean
    axes[1].barh(range(len(layers)), abs_means, color="coral")
    axes[1].set_yticks(range(len(layers)))
    axes[1].set_yticklabels(tick_labels, fontsize=8)
    axes[1].set_xlabel("|Mean Gradient|")
    axes[1].set_title("Gradient Mean")

    # Standard deviation
    axes[2].barh(range(len(layers)), stds, color="seagreen")
    axes[2].set_yticks(range(len(layers)))
    axes[2].set_yticklabels(tick_labels, fontsize=8)
    axes[2].set_xlabel("Gradient Std")
    axes[2].set_title("Gradient Variance")

    plt.suptitle("Gradient Flow Through Network")
    plt.tight_layout()

    fig.canvas.draw()
    img = Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )
    plt.close(fig)

    return img


# =============================================================================
# t-SNE/UMAP Embedding Visualization
# =============================================================================


@torch.no_grad()
def generate_embedding_projection(
    model,
    dataloader,
    device: torch.device,
    method: str = "tsne",
    max_samples: int = 500,
    figsize: Tuple[int, int] = (10, 8),
    class_names: Optional[List[str]] = None,
) -> Image.Image:
    """
    Project embeddings to 2D using t-SNE or UMAP.

    Visualizes the learned representation space, showing how well
    different classes are separated.

    Args:
        model: LeJEPA model
        dataloader: DataLoader with (images, labels)
        device: torch device
        method: 'tsne' or 'umap'
        max_samples: Maximum samples to project
        figsize: Figure size
        class_names: Optional list of class names

    Returns:
        PIL Image of embedding projection
    """
    model.eval()

    all_embeddings = []
    all_labels = []

    for views, labels in dataloader:
        if len(all_embeddings) * views.shape[0] >= max_samples:
            break

        views = views.to(device)
        emb, _ = model(views)

        all_embeddings.append(emb.detach().cpu())
        all_labels.append(labels)

    embeddings = torch.cat(all_embeddings, dim=0)[:max_samples]
    labels = torch.cat(all_labels, dim=0)[:max_samples]

    # Dimensionality reduction
    embeddings_np = embeddings.numpy()

    if method == "tsne":
        try:
            from sklearn.manifold import TSNE

            reducer = TSNE(n_components=2, perplexity=30, random_state=42)
            projected = reducer.fit_transform(embeddings_np)
        except ImportError:
            # Fallback to PCA
            from sklearn.decomposition import PCA

            reducer = PCA(n_components=2)
            projected = reducer.fit_transform(embeddings_np)
            method = "pca"
    elif method == "umap":
        try:
            import umap

            reducer = umap.UMAP(n_components=2, random_state=42)
            projected = reducer.fit_transform(embeddings_np)
        except ImportError:
            # Fallback to t-SNE
            from sklearn.manifold import TSNE

            reducer = TSNE(n_components=2, perplexity=30, random_state=42)
            projected = reducer.fit_transform(embeddings_np)
            method = "tsne"
    else:
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2)
        projected = reducer.fit_transform(embeddings_np)
        method = "pca"

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = labels.unique().numpy()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for idx, label in enumerate(unique_labels):
        mask = labels.numpy() == label
        name = class_names[label] if class_names else f"Class {label}"
        ax.scatter(
            projected[mask, 0],
            projected[mask, 1],
            c=[colors[idx]],
            label=name,
            alpha=0.7,
            s=20,
        )

    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.set_title(f"Embedding Space ({method.upper()})")

    plt.tight_layout()

    fig.canvas.draw()
    img = Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )
    plt.close(fig)

    return img


# =============================================================================
# Attention Difference Tracking
# =============================================================================


class AttentionTracker:
    """
    Track attention patterns across training for difference analysis.

    Stores attention snapshots at different epochs to visualize
    how attention patterns evolve during training.
    """

    def __init__(self, max_snapshots: int = 10):
        self.snapshots = []  # List of (epoch, attention_maps)
        self.max_snapshots = max_snapshots

    @torch.no_grad()
    def capture(self, model, images: torch.Tensor, device: torch.device, epoch: int):
        """Capture attention maps at current epoch."""
        model.eval()

        for block in model.encoder.blocks:
            block.attn.output_attention = True

        images = images.to(device)
        _ = model.encoder(images)

        attentions = model.encoder.get_attention_maps()

        for block in model.encoder.blocks:
            block.attn.output_attention = False

        # Store only first sample to save memory
        attn_snapshot = [a[0:1].detach().cpu() for a in attentions]

        self.snapshots.append((epoch, attn_snapshot))

        # Keep only max_snapshots
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)

    def generate_difference_visualization(
        self,
        img_size: int = 128,
        patch_size: int = 8,
    ) -> Optional[Image.Image]:
        """
        Generate visualization showing attention difference between epochs.

        Returns:
            PIL Image showing attention evolution, or None if < 2 snapshots
        """
        if len(self.snapshots) < 2:
            return None

        # Compare first and last snapshots
        epoch1, attn1 = self.snapshots[0]
        epoch2, attn2 = self.snapshots[-1]

        num_layers = len(attn1)

        # Create grid showing difference at each layer
        cols = min(4, num_layers)
        rows = (num_layers + cols - 1) // cols

        grid_width = cols * img_size
        grid_height = rows * img_size
        grid = Image.new("RGB", (grid_width, grid_height), (128, 128, 128))

        h = w = int(np.sqrt(attn1[0].shape[2] - 1))  # Assuming CLS token

        for layer_idx in range(num_layers):
            # Get attention for layer
            a1 = attn1[layer_idx][0].mean(dim=0)  # (N, N)
            a2 = attn2[layer_idx][0].mean(dim=0)

            # Compute difference (CLS attention to patches)
            diff = (a2[0, 1:] - a1[0, 1:]).reshape(h, w)
            diff = diff.numpy()

            # Normalize to [-1, 1] for visualization
            max_val = max(abs(diff.min()), abs(diff.max())) + 1e-6
            diff_norm = diff / max_val

            # Color: blue=decrease, red=increase
            diff_rgb = np.zeros((h, w, 3), dtype=np.uint8)
            diff_rgb[..., 0] = np.clip(diff_norm * 255, 0, 255).astype(
                np.uint8
            )  # Red for increase
            diff_rgb[..., 2] = np.clip(-diff_norm * 255, 0, 255).astype(
                np.uint8
            )  # Blue for decrease

            img = Image.fromarray(diff_rgb).resize(
                (img_size, img_size), resample=Image.BILINEAR
            )

            # Add label
            draw = ImageDraw.Draw(img)
            draw.text((5, 5), f"L{layer_idx}", fill=(255, 255, 255))
            draw.text((5, img_size - 15), f"E{epoch1}→{epoch2}", fill=(255, 255, 255))

            col = layer_idx % cols
            row = layer_idx // cols
            grid.paste(img, (col * img_size, row * img_size))

        return grid


# =============================================================================
# Feature Collapse Monitor
# =============================================================================


@torch.no_grad()
def generate_collapse_monitor(
    embeddings: torch.Tensor,
    figsize: Tuple[int, int] = (12, 4),
) -> Image.Image:
    """
    Visualize indicators of representation collapse.

    Creates dashboard showing:
    - Embedding norm distribution
    - Pairwise similarity distribution
    - Singular value spectrum

    Args:
        embeddings: (B, D) embedding vectors
        figsize: Figure size

    Returns:
        PIL Image dashboard
    """
    embeddings = embeddings.detach().cpu()
    B, D = embeddings.shape

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. Norm distribution
    norms = embeddings.norm(dim=1).numpy()
    axes[0].hist(norms, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0].axvline(
        norms.mean(), color="red", linestyle="--", label=f"Mean: {norms.mean():.2f}"
    )
    axes[0].set_xlabel("Embedding Norm")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Norm Distribution")
    axes[0].legend()

    # 2. Pairwise similarity distribution
    emb_norm = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)
    sim = torch.mm(emb_norm, emb_norm.T)
    # Get upper triangle (excluding diagonal)
    triu_indices = torch.triu_indices(B, B, offset=1)
    similarities = sim[triu_indices[0], triu_indices[1]].numpy()

    axes[1].hist(similarities, bins=30, color="coral", edgecolor="black", alpha=0.7)
    axes[1].axvline(
        similarities.mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {similarities.mean():.2f}",
    )
    axes[1].set_xlabel("Cosine Similarity")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Pairwise Similarity")
    axes[1].legend()

    # Collapse warning
    if similarities.mean() > 0.8:
        axes[1].text(
            0.5,
            0.9,
            "⚠️ COLLAPSE",
            transform=axes[1].transAxes,
            fontsize=12,
            color="red",
            ha="center",
        )

    # 3. Singular value spectrum
    try:
        s = torch.linalg.svdvals(embeddings).numpy()
        s_norm = s / s.sum()
        axes[2].bar(range(min(50, len(s))), s_norm[:50], color="seagreen")
        axes[2].set_xlabel("Singular Value Index")
        axes[2].set_ylabel("Normalized Value")
        axes[2].set_title("Singular Value Spectrum")

        # Effective rank
        effective_rank = np.exp(-np.sum(s_norm * np.log(s_norm + 1e-8)))
        axes[2].text(
            0.7,
            0.9,
            f"Eff. Rank: {effective_rank:.1f}",
            transform=axes[2].transAxes,
            fontsize=10,
        )
    except Exception:
        axes[2].text(0.5, 0.5, "SVD Failed", transform=axes[2].transAxes, ha="center")

    plt.suptitle("Representation Collapse Monitor")
    plt.tight_layout()

    fig.canvas.draw()
    img = Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )
    plt.close(fig)

    return img


# =============================================================================
# Combined Training Dynamics Dashboard
# =============================================================================


def generate_training_dashboard(
    loss_history: List[float],
    acc_history: List[float],
    entropy_history: List[float],
    gini_history: List[float],
    lr_history: List[float],
    figsize: Tuple[int, int] = (14, 8),
) -> Image.Image:
    """
    Generate comprehensive training dynamics dashboard.

    Shows:
    - Loss curve
    - Accuracy curve
    - Attention entropy evolution
    - Attention Gini (concentration) evolution
    - Learning rate schedule

    Args:
        loss_history: List of loss values per epoch
        acc_history: List of accuracy values per epoch
        entropy_history: List of attention entropy values
        gini_history: List of attention Gini values
        lr_history: List of learning rate values
        figsize: Figure size

    Returns:
        PIL Image dashboard
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    epochs = range(1, len(loss_history) + 1)

    # Loss
    axes[0, 0].plot(epochs, loss_history, "b-", linewidth=2)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(epochs, acc_history, "g-", linewidth=2)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].set_title("Validation Accuracy")
    axes[0, 1].grid(True, alpha=0.3)

    # Learning rate
    if lr_history:
        axes[0, 2].plot(range(len(lr_history)), lr_history, "orange", linewidth=2)
        axes[0, 2].set_xlabel("Step")
        axes[0, 2].set_ylabel("Learning Rate")
        axes[0, 2].set_title("Learning Rate Schedule")
        axes[0, 2].set_yscale("log")
        axes[0, 2].grid(True, alpha=0.3)

    # Attention Entropy
    if entropy_history:
        axes[1, 0].plot(
            epochs[: len(entropy_history)], entropy_history, "purple", linewidth=2
        )
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Entropy")
        axes[1, 0].set_title("Attention Entropy (↑=uniform)")
        axes[1, 0].grid(True, alpha=0.3)

    # Attention Gini
    if gini_history:
        axes[1, 1].plot(epochs[: len(gini_history)], gini_history, "red", linewidth=2)
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Gini")
        axes[1, 1].set_title("Attention Gini (↑=focused)")
        axes[1, 1].grid(True, alpha=0.3)

    # Combined view
    if entropy_history and gini_history:
        ax2 = axes[1, 2]
        ax2.plot(
            epochs[: len(entropy_history)],
            entropy_history,
            "purple",
            label="Entropy",
            linewidth=2,
        )
        ax2_twin = ax2.twinx()
        ax2_twin.plot(
            epochs[: len(gini_history)], gini_history, "red", label="Gini", linewidth=2
        )
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Entropy", color="purple")
        ax2_twin.set_ylabel("Gini", color="red")
        ax2.set_title("Attention Dynamics")
        ax2.grid(True, alpha=0.3)

    plt.suptitle("Training Dynamics Dashboard", fontsize=14)
    plt.tight_layout()

    fig.canvas.draw()
    img = Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )
    plt.close(fig)

    return img
