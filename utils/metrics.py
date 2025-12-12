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
    if eigenvalues is not None:
        isotropy = eigenvalues.min().item() / (eigenvalues.max().item() + 1e-8)
    else:
        isotropy = 0.0

    return {
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "variance": emb_var,
        "effective_dim": effective_dim,
        "isotropy": isotropy,
    }


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
