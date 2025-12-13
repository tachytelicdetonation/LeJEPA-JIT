"""
Metrics for monitoring training and attention dynamics.

Includes:
- Attention statistics: entropy, Gini, sparsity
- Gradient Signal to Noise Ratio (GSNR)
- Layer-wise gradient statistics
- Attention head diversity metrics
- Effective rank of representations
"""

from typing import Optional

import torch
import torch.nn as nn

# =============================================================================
# Attention Statistics
# =============================================================================


def compute_entropy(attn_probs: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Compute Shannon entropy of attention distribution.
    H(p) = -sum(p * log(p))

    Higher entropy = more uniform attention (less focused).
    Lower entropy = more peaked attention (more focused).

    Args:
        attn_probs: (B, H, N, N) attention probabilities (sum to 1 on last dim)

    Returns:
        entropy: Scalar mean entropy
    """
    log_probs = torch.log(attn_probs + epsilon)
    entropy = -torch.sum(attn_probs * log_probs, dim=-1)  # (B, H, N)
    return entropy.mean()


def compute_gini(attn_probs: torch.Tensor) -> torch.Tensor:
    """
    Compute Gini coefficient of attention distribution.
    Measures inequality (0 = perfect equality, 1 = max inequality).

    Higher Gini = attention concentrated on few tokens.
    Lower Gini = attention spread across many tokens.

    Args:
        attn_probs: (B, H, N, N)

    Returns:
        gini: Scalar mean Gini
    """
    B, H, N, M = attn_probs.shape
    x = attn_probs.reshape(-1, M)
    x_sorted, _ = torch.sort(x, dim=-1)

    # Gini formula: (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1)/n
    index = torch.arange(1, M + 1, device=x.device).float()
    numerator = torch.sum(index * x_sorted, dim=-1)
    denominator = torch.sum(x_sorted, dim=-1)

    gini = (2 * numerator) / (M * denominator) - (M + 1) / M
    return gini.mean()


def compute_sparsity(attn_probs: torch.Tensor, threshold: float = 1e-4) -> torch.Tensor:
    """
    Compute sparsity: Fraction of attention weights below threshold.

    Args:
        attn_probs: (B, H, N, N)

    Returns:
        sparsity: Scalar fraction
    """
    below_threshold = (attn_probs < threshold).float()
    return below_threshold.mean()


# =============================================================================
# Gradient Signal to Noise Ratio (GSNR)
# =============================================================================


class GSNRTracker:
    """
    Track Gradient Signal to Noise Ratio (GSNR) across training.

    GSNR = mean(gradient)^2 / variance(gradient)

    Higher GSNR indicates more consistent gradient direction across samples,
    which correlates with better generalization (Liu et al., 2020).

    Reference: "Understanding Why Neural Networks Generalize Well Through
    GSNR of Parameters" (arXiv:2001.07384)
    """

    def __init__(self, model: nn.Module, track_layers: Optional[list] = None):
        """
        Initialize GSNR tracker.

        Args:
            model: The model to track
            track_layers: Optional list of layer names to track.
                         If None, tracks all layers with parameters.
        """
        self.model = model
        self.track_layers = track_layers

        # Running statistics for each parameter
        self.grad_sum = {}  # Sum of gradients (for mean)
        self.grad_sq_sum = {}  # Sum of squared gradients (for variance)
        self.num_samples = 0

        self._init_stats()

    def _init_stats(self):
        """Initialize gradient statistics storage."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.track_layers is None or any(
                    layer in name for layer in self.track_layers
                ):
                    self.grad_sum[name] = torch.zeros_like(param)
                    self.grad_sq_sum[name] = torch.zeros_like(param)

    def reset(self):
        """Reset accumulated statistics."""
        for name in self.grad_sum:
            self.grad_sum[name].zero_()
            self.grad_sq_sum[name].zero_()
        self.num_samples = 0

    @torch.no_grad()
    def update(self):
        """
        Update running statistics with current gradients.
        Call this after backward() but before optimizer.step().
        """
        self.num_samples += 1

        for name, param in self.model.named_parameters():
            if name in self.grad_sum and param.grad is not None:
                self.grad_sum[name].add_(param.grad)
                self.grad_sq_sum[name].add_(param.grad.pow(2))

    @torch.no_grad()
    def compute_gsnr(self, epsilon: float = 1e-8) -> dict:
        """
        Compute GSNR for all tracked parameters.

        Returns:
            dict: {param_name: gsnr_value} for each parameter,
                  plus aggregated statistics
        """
        if self.num_samples < 2:
            return {"mean_gsnr": 0.0, "layer_gsnr": {}}

        n = self.num_samples
        gsnr_per_param = {}
        layer_gsnr = {}

        for name in self.grad_sum:
            # Mean gradient: E[g]
            mean_grad = self.grad_sum[name] / n

            # Variance: E[g^2] - E[g]^2
            mean_sq_grad = self.grad_sq_sum[name] / n
            variance = mean_sq_grad - mean_grad.pow(2)

            # GSNR = mean^2 / variance (element-wise, then aggregate)
            signal = mean_grad.pow(2)
            noise = variance + epsilon

            # Per-parameter GSNR (scalar)
            param_gsnr = (signal.sum() / noise.sum()).item()
            gsnr_per_param[name] = param_gsnr

            # Group by layer (e.g., "blocks.0.attn" -> "blocks.0")
            layer_name = ".".join(name.split(".")[:2])
            if layer_name not in layer_gsnr:
                layer_gsnr[layer_name] = []
            layer_gsnr[layer_name].append(param_gsnr)

        # Aggregate layer-wise GSNR
        layer_gsnr_mean = {
            layer: sum(vals) / len(vals) for layer, vals in layer_gsnr.items()
        }

        # Overall mean GSNR
        all_gsnr = list(gsnr_per_param.values())
        mean_gsnr = sum(all_gsnr) / len(all_gsnr) if all_gsnr else 0.0

        return {
            "mean_gsnr": mean_gsnr,
            "layer_gsnr": layer_gsnr_mean,
            "param_gsnr": gsnr_per_param,
        }


def compute_batch_gsnr(model: nn.Module, epsilon: float = 1e-8) -> dict:
    """
    Compute GSNR from current batch gradients only (simplified version).

    This computes per-layer gradient statistics for the current batch.
    For true GSNR, use GSNRTracker to accumulate across batches.

    Args:
        model: Model with gradients computed
        epsilon: Small value for numerical stability

    Returns:
        dict with layer-wise gradient statistics
    """
    layer_stats = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad

            # Layer grouping
            layer_name = ".".join(name.split(".")[:2])
            if layer_name not in layer_stats:
                layer_stats[layer_name] = {
                    "grad_norm": 0.0,
                    "grad_mean": 0.0,
                    "grad_std": 0.0,
                    "num_params": 0,
                }

            layer_stats[layer_name]["grad_norm"] += grad.norm().item() ** 2
            layer_stats[layer_name]["grad_mean"] += grad.mean().item()
            layer_stats[layer_name]["grad_std"] += grad.std().item()
            layer_stats[layer_name]["num_params"] += 1

    # Average within layers
    for layer in layer_stats:
        n = layer_stats[layer]["num_params"]
        if n > 0:
            layer_stats[layer]["grad_norm"] = (
                layer_stats[layer]["grad_norm"] / n
            ) ** 0.5
            layer_stats[layer]["grad_mean"] /= n
            layer_stats[layer]["grad_std"] /= n

    return layer_stats


# =============================================================================
# Layer-wise Gradient Statistics
# =============================================================================


def compute_gradient_flow_stats(grad_tensor: torch.Tensor) -> dict:
    """
    Compute basic stats of gradients.

    Args:
        grad_tensor: arbitrary gradient tensor

    Returns:
        dict: {'norm': float, 'std': float, 'max': float}
    """
    if grad_tensor is None:
        return {"norm": 0.0, "std": 0.0, "max": 0.0}

    norm = grad_tensor.norm().item()
    std = grad_tensor.std().item()
    mx = grad_tensor.abs().max().item()

    return {"norm": norm, "std": std, "max": mx}


def compute_layer_gradient_stats(model: nn.Module) -> dict:
    """
    Compute gradient statistics per layer.

    Useful for detecting vanishing/exploding gradients and
    understanding gradient flow through the network.

    Args:
        model: Model with gradients computed

    Returns:
        dict: {layer_name: {norm, mean, std, max, min}}
    """
    layer_stats = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.flatten()

            # Extract layer name (first 2 components)
            layer_name = ".".join(name.split(".")[:2])

            if layer_name not in layer_stats:
                layer_stats[layer_name] = {
                    "grads": [],
                }

            layer_stats[layer_name]["grads"].append(grad)

    # Aggregate per layer
    results = {}
    for layer_name, data in layer_stats.items():
        all_grads = torch.cat(data["grads"])
        results[layer_name] = {
            "norm": all_grads.norm().item(),
            "mean": all_grads.mean().item(),
            "std": all_grads.std().item(),
            "max": all_grads.max().item(),
            "min": all_grads.min().item(),
            "abs_mean": all_grads.abs().mean().item(),
        }

    return results


# =============================================================================
# Attention Head Diversity Metrics
# =============================================================================


def compute_head_diversity(attn_probs: torch.Tensor, epsilon: float = 1e-8) -> dict:
    """
    Compute metrics for attention head diversity.

    Diverse heads = each head learns different patterns.
    Non-diverse = redundant heads that could be pruned.

    Args:
        attn_probs: (B, H, N, N) attention probabilities

    Returns:
        dict with diversity metrics
    """
    B, H, N, M = attn_probs.shape

    # Flatten spatial dimensions for each head
    head_patterns = attn_probs.reshape(B, H, -1)  # (B, H, N*M)

    # 1. Cosine similarity between heads
    # Normalize each head's pattern
    head_norm = head_patterns / (head_patterns.norm(dim=-1, keepdim=True) + epsilon)

    # Compute pairwise cosine similarity
    # (B, H, D) @ (B, D, H) -> (B, H, H)
    sim_matrix = torch.bmm(head_norm, head_norm.transpose(1, 2))

    # Average off-diagonal similarity (excluding self-similarity)
    mask = ~torch.eye(H, device=attn_probs.device, dtype=torch.bool)
    off_diag_sim = sim_matrix[:, mask].mean().item()

    # 2. Head specialization: variance of attention patterns across heads
    head_variance = head_patterns.var(dim=1).mean().item()

    # 3. Effective number of heads (based on attention pattern diversity)
    # Using entropy of normalized head similarities
    sim_probs = sim_matrix.softmax(dim=-1)
    head_entropy = -(sim_probs * torch.log(sim_probs + epsilon)).sum(dim=-1).mean()
    effective_heads = torch.exp(head_entropy).item()

    return {
        "head_similarity": off_diag_sim,  # Lower = more diverse
        "head_variance": head_variance,  # Higher = more specialized
        "effective_heads": effective_heads,  # Higher = more utilized
    }


def compute_attention_rank(attn_probs: torch.Tensor, threshold: float = 0.01) -> dict:
    """
    Compute effective rank of attention matrices.

    Low rank = redundant attention patterns.
    High rank = utilizing full representational capacity.

    Args:
        attn_probs: (B, H, N, N) attention probabilities
        threshold: Singular value threshold for rank computation

    Returns:
        dict with rank metrics
    """
    B, H, N, M = attn_probs.shape

    # Reshape to (B*H, N, M)
    attn_flat = attn_probs.reshape(B * H, N, M)

    # Compute SVD for a sample (full SVD is expensive)
    # Use first batch element only for efficiency
    sample_attn = attn_flat[:H]  # (H, N, M) - first batch

    ranks = []
    spectral_norms = []

    for h in range(H):
        try:
            # Compute singular values
            s = torch.linalg.svdvals(sample_attn[h])

            # Effective rank (count significant singular values)
            total_energy = s.sum()
            normalized_s = s / (total_energy + 1e-8)
            effective_rank = torch.exp(
                -(normalized_s * torch.log(normalized_s + 1e-8)).sum()
            ).item()

            ranks.append(effective_rank)
            spectral_norms.append(s[0].item())
        except Exception:
            continue

    if not ranks:
        return {"effective_rank": 0.0, "numeric_rank": 0.0, "spectral_norm": 0.0}

    return {
        "effective_rank": sum(ranks) / len(ranks),
        "spectral_norm": sum(spectral_norms) / len(spectral_norms),
    }


def compute_attention_structure_metrics(attn_probs: torch.Tensor) -> dict:
    """
    Transformer attention structure diagnostics.

    These help detect "attention sinks" (CLS gathering mass),
    overly self-focused attention (high diagonal), and head imbalance.
    """
    if attn_probs is None or attn_probs.ndim != 4:
        return {
            "diag_mass": 0.0,
            "cls_to_patches": 0.0,
            "cls_self": 0.0,
            "patches_to_cls": 0.0,
            "head_entropy_std": 0.0,
        }

    B, H, N, _ = attn_probs.shape
    device = attn_probs.device

    # Diagonal mass: average attention to self (over all query tokens)
    idx = torch.arange(N, device=device)
    diag = attn_probs[:, :, idx, idx]  # (B, H, N)
    diag_mass = diag.mean().item()

    # Head entropy variability (imbalance/redundancy proxy)
    eps = 1e-8
    ent = -(attn_probs * torch.log(attn_probs + eps)).sum(dim=-1).mean(dim=-1)  # (B,H)
    head_entropy_std = ent.std(dim=1).mean().item()

    # CLS-related metrics if CLS token exists (assume index 0)
    cls_to_patches = 0.0
    cls_self = 0.0
    patches_to_cls = 0.0
    if N >= 2:
        cls_self = attn_probs[:, :, 0, 0].mean().item()
        cls_to_patches = attn_probs[:, :, 0, 1:].mean().item()
        patches_to_cls = attn_probs[:, :, 1:, 0].mean().item()

    return {
        "diag_mass": diag_mass,
        "cls_to_patches": cls_to_patches,
        "cls_self": cls_self,
        "patches_to_cls": patches_to_cls,
        "head_entropy_std": head_entropy_std,
    }


def compute_attention_distance_metrics(
    attn_probs: torch.Tensor, grid_size: int, radius: int = 1
) -> dict:
    """
    Compute ViT-style attention distance/locality metrics for patch tokens.

    Inspired by "attention distance" diagnostics commonly used with ViTs:
    expected 2D distance under attention weights.
    """
    if attn_probs is None or attn_probs.ndim != 4 or grid_size <= 0:
        return {
            "patch_attn_distance_mean": 0.0,
            "patch_attn_distance_std": 0.0,
            "patch_local_mass_r1": 0.0,
        }

    B, H, N, _ = attn_probs.shape
    has_cls = N == grid_size * grid_size + 1
    start = 1 if has_cls else 0
    num_patches = N - start
    if num_patches != grid_size * grid_size:
        # Can't infer spatial layout reliably
        return {
            "patch_attn_distance_mean": 0.0,
            "patch_attn_distance_std": 0.0,
            "patch_local_mass_r1": 0.0,
        }

    # Patch coords in (row, col)
    device = attn_probs.device
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(grid_size, device=device),
            torch.arange(grid_size, device=device),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)
    dist = torch.cdist(coords.float(), coords.float(), p=2)  # (P,P)

    a = attn_probs[:, :, start:, start:]  # (B,H,P,P)
    a = a / (a.sum(dim=-1, keepdim=True) + 1e-8)

    # Expected distance for each head (average over batch + query positions)
    # E_d = mean_{b,h,i} sum_j a[b,h,i,j] * dist[i,j]
    exp_d = (a * dist.view(1, 1, num_patches, num_patches)).sum(dim=-1).mean(dim=-1)
    # exp_d: (B,H) mean over queries; now average over batch -> (H,)
    per_head = exp_d.mean(dim=0)  # (H,)

    mean = per_head.mean().item()
    std = per_head.std().item() if H > 1 else 0.0

    # Locality mass: fraction of attention within radius (in patch grid distance)
    mask = (dist <= float(radius)).float()  # (P,P)
    local_mass_bh = (a * mask.view(1, 1, num_patches, num_patches)).sum(dim=-1).mean(dim=-1)  # (B,H)
    local_mass_per_head = local_mass_bh.mean(dim=0)  # (H,)
    local_mass = local_mass_per_head.mean().item()  # avg over heads

    return {
        "patch_attn_distance_mean": mean,
        "patch_attn_distance_std": std,
        "patch_local_mass_r1": local_mass,
        "patch_attn_distance_per_head": per_head.detach().cpu().tolist(),
        "patch_local_mass_r1_per_head": local_mass_per_head.detach().cpu().tolist(),
    }


@torch.no_grad()
def compute_encoder_block_opt_stats(
    model: nn.Module,
    lr: Optional[float] = None,
    epsilon: float = 1e-12,
) -> dict:
    """
    Compute encoder block-level parameter/gradient norms and ratios.

    Returns lists aligned with `model.encoder.blocks` index:
      - param_norm
      - grad_norm
      - grad_to_param
      - lr_scaled_grad_to_param (if lr is provided)
      - nonfinite_grad_params (count of params with non-finite grads)
    """
    encoder = getattr(model, "encoder", None)
    blocks = getattr(encoder, "blocks", None) if encoder is not None else None
    if blocks is None:
        return {}

    param_norms: list[float] = []
    grad_norms: list[float] = []
    grad_to_param: list[float] = []
    lr_scaled: list[float] = []
    nonfinite_params: list[float] = []

    for blk in blocks:
        p_sq = 0.0
        g_sq = 0.0
        nf = 0.0
        for p in blk.parameters():
            if not p.requires_grad:
                continue
            pf = p.detach().float()
            p_sq += float(pf.pow(2).sum().item())
            if p.grad is not None:
                gf = p.grad.detach().float()
                g_sq += float(gf.pow(2).sum().item())
                if not torch.isfinite(gf).all().item():
                    nf += 1.0

        pn = float((p_sq + float(epsilon)) ** 0.5)
        gn = float((g_sq + float(epsilon)) ** 0.5)
        r = gn / (pn + float(epsilon))

        param_norms.append(pn)
        grad_norms.append(gn)
        grad_to_param.append(float(r))
        nonfinite_params.append(float(nf))
        if lr is not None:
            lr_scaled.append(float(lr) * float(r))

    out = {
        "param_norm": param_norms,
        "grad_norm": grad_norms,
        "grad_to_param": grad_to_param,
        "nonfinite_grad_params": nonfinite_params,
    }
    if lr is not None:
        out["lr_scaled_grad_to_param"] = lr_scaled
    return out


@torch.no_grad()
def estimate_intrinsic_dim_twonn(embeddings: torch.Tensor, epsilon: float = 1e-12) -> float:
    """
    TwoNN intrinsic dimension estimator (Facco et al., 2017).

    Works best with a few hundred+ samples; use as a trend metric.
    """
    z = embeddings.detach().float()
    if z.ndim != 2 or z.shape[0] < 10:
        return 0.0

    # Normalize to reduce scale sensitivity
    z = z - z.mean(dim=0, keepdim=True)
    z = z / (z.norm(dim=1, keepdim=True) + 1e-8)

    # Pairwise distances
    d = torch.cdist(z, z, p=2)  # (N,N)
    n = d.shape[0]
    d[torch.arange(n), torch.arange(n)] = float("inf")

    # First and second nearest neighbor distances
    d1, _ = d.min(dim=1)
    d2, _ = d.masked_fill(d == d1.unsqueeze(1), float("inf")).min(dim=1)

    mu = (d2 / (d1 + epsilon)).clamp(min=1.0 + 1e-6)
    mu, _ = torch.sort(mu)

    # Empirical CDF and linear fit through origin:
    # y = -log(1 - F), x = log(mu), slope ~= intrinsic dimension
    i = torch.arange(1, n + 1, device=mu.device, dtype=torch.float32)
    F = (i - 0.5) / n
    x = torch.log(mu)
    y = -torch.log(1.0 - F + 1e-6)

    # Least squares slope through origin
    denom = (x * x).sum().clamp(min=epsilon)
    slope = (x * y).sum() / denom
    return slope.item()


# =============================================================================
# Representation Quality Metrics
# =============================================================================


def compute_representation_stats(embeddings: torch.Tensor) -> dict:
    """
    Compute statistics of learned representations.

    Args:
        embeddings: (B, D) embedding vectors

    Returns:
        dict with representation statistics
    """
    B, D = embeddings.shape

    # 1. Embedding norms
    norms = embeddings.norm(dim=1)
    norm_mean = norms.mean().item()
    norm_std = norms.std().item()

    # 2. Embedding variance (should be high for good representations)
    emb_var = embeddings.var(dim=0).mean().item()

    # 3. Effective dimensionality (based on covariance eigenspectrum)
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / B  # (D, D)

    eigenvalues = None
    try:
        eigenvalues = torch.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues.clamp(min=1e-8)
        normalized_eig = eigenvalues / eigenvalues.sum()
        effective_dim = torch.exp(
            -(normalized_eig * torch.log(normalized_eig)).sum()
        ).item()
    except Exception:
        effective_dim = D

    # 4. Isotropy (how uniformly embeddings use all dimensions)
    # Perfect isotropy = eigenvalues are equal
    isotropy = 0.0
    if eigenvalues is not None:
        isotropy = eigenvalues.min().item() / (eigenvalues.max().item() + 1e-8)

    return {
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "variance": emb_var,
        "effective_dim": effective_dim,
        "isotropy": isotropy,
    }


def compute_alignment_metrics(
    z1: torch.Tensor, z2: torch.Tensor, epsilon: float = 1e-8
) -> dict:
    """
    Compute simple alignment metrics between two paired views.

    These are commonly used in self-supervised learning to monitor whether
    positive pairs stay close (alignment) without collapsing.

    Args:
        z1: (B, D)
        z2: (B, D)

    Returns:
        dict with cosine similarity and normalized L2 distance
    """
    z1 = z1.float()
    z2 = z2.float()
    z1n = z1 / (z1.norm(dim=1, keepdim=True) + epsilon)
    z2n = z2 / (z2.norm(dim=1, keepdim=True) + epsilon)

    cos = (z1n * z2n).sum(dim=1).mean().item()
    l2 = (z1n - z2n).pow(2).sum(dim=1).mean().item()
    return {"cos": cos, "l2": l2}


def compute_covariance_metrics(embeddings: torch.Tensor, epsilon: float = 1e-8) -> dict:
    """
    Compute covariance-based redundancy metrics of representations.

    Off-diagonal energy is a standard proxy for feature redundancy
    (used by redundancy-reduction methods like Barlow Twins).
    """
    x = embeddings.float()
    B, D = x.shape
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / max(B - 1, 1)  # (D, D)

    diag = torch.diagonal(cov)
    offdiag = cov - torch.diag(diag)

    offdiag_l2 = offdiag.pow(2).mean().item()
    diag_mean = diag.mean().item()
    diag_min = diag.min().item()

    var = x.var(dim=0, unbiased=False)
    var_mean = var.mean().item()
    var_min = var.min().item()

    return {
        "cov_offdiag_l2": offdiag_l2,
        "cov_diag_mean": diag_mean,
        "cov_diag_min": diag_min,
        "var_mean": var_mean,
        "var_min": var_min,
    }


@torch.no_grad()
def compute_global_norms(model: nn.Module, epsilon: float = 1e-12) -> dict:
    """
    Compute global L2 norms for parameters and gradients.

    Useful for dashboards:
      - opt/param_norm
      - opt/grad_norm
      - opt/grad_to_param (scale proxy)
    """
    param_sq = 0.0
    grad_sq = 0.0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        param_sq += p.detach().float().pow(2).sum().item()
        if p.grad is not None:
            grad_sq += p.grad.detach().float().pow(2).sum().item()

    param_norm = (param_sq + epsilon) ** 0.5
    grad_norm = (grad_sq + epsilon) ** 0.5
    return {
        "param_norm": param_norm,
        "grad_norm": grad_norm,
        "grad_to_param": grad_norm / param_norm,
    }


@torch.no_grad()
def compute_linear_cka(x: torch.Tensor, y: torch.Tensor, epsilon: float = 1e-12) -> float:
    """
    Compute linear CKA between two feature matrices.

    This is a lightweight representation drift metric that is scale-invariant
    and correlates well with similarity in learned representations.

    Args:
        x: (B, D)
        y: (B, D)
    """
    x = x.detach().float()
    y = y.detach().float()

    if x.ndim != 2 or y.ndim != 2 or x.numel() == 0 or y.numel() == 0:
        return 0.0

    b = min(x.shape[0], y.shape[0])
    if b < 2:
        return 0.0

    x = x[:b]
    y = y[:b]

    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    xty = x.T @ y
    xtx = x.T @ x
    yty = y.T @ y

    hsic = (xty * xty).sum()
    norm_x = (xtx * xtx).sum().sqrt()
    norm_y = (yty * yty).sum().sqrt()

    denom = (norm_x * norm_y).clamp(min=epsilon)
    return (hsic / denom).item()

def compute_feature_collapse_metrics(embeddings: torch.Tensor) -> dict:
    """
    Detect representation collapse in self-supervised learning.

    Collapse indicators:
    - Low variance across samples
    - High cosine similarity between samples
    - Low effective rank of representation matrix

    Args:
        embeddings: (B, D) embedding vectors

    Returns:
        dict with collapse metrics
    """
    B, D = embeddings.shape
    epsilon = 1e-8

    # 1. Average pairwise cosine similarity (high = potential collapse)
    emb_norm = embeddings / (embeddings.norm(dim=1, keepdim=True) + epsilon)
    sim_matrix = emb_norm @ emb_norm.T  # (B, B)

    # Exclude diagonal (self-similarity)
    mask = ~torch.eye(B, device=embeddings.device, dtype=torch.bool)
    avg_similarity = sim_matrix[mask].mean().item()

    # 2. Standard deviation of embeddings (low = collapse)
    std = embeddings.std().item()

    # 3. Effective rank of embedding matrix
    try:
        s = torch.linalg.svdvals(embeddings)
        normalized_s = s / (s.sum() + epsilon)
        effective_rank = torch.exp(
            -(normalized_s * torch.log(normalized_s + epsilon)).sum()
        ).item()
    except Exception:
        effective_rank = min(B, D)

    # 4. Uniformity loss (from "Understanding Contrastive Representation Learning")
    # Lower = more uniform on hypersphere
    uniformity = torch.pdist(emb_norm).pow(2).mul(-2).exp().mean().log().item()

    return {
        "avg_similarity": avg_similarity,  # High = bad
        "std": std,  # Low = bad
        "effective_rank": effective_rank,  # Low = bad
        "uniformity": uniformity,  # High = bad
    }
