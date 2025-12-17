"""Configuration for LeJEPA-JiT training."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class Config:
    # Model selection
    encoder: Literal["jit", "vit"] = "jit"

    # Image and patch settings
    img_size: int = 224
    patch_size: int = 16  # Standard ViT/16
    in_channels: int = 3

    # Encoder architecture
    embed_dim: int = 384  # ViT-Small
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0

    # JiT-specific
    bottleneck_dim: int = 128
    use_rope: bool = True
    use_swiglu: bool = True

    # Projector
    proj_hidden_dim: int = 2048
    proj_dim: int = 64  # Paper ablations show this is a strong default

    # Training
    batch_size: int = 256
    epochs: int = 100
    lr_encoder: float = 5e-4  # Paper value
    lr_probe: float = 1e-3
    weight_decay_encoder: float = 5e-2
    weight_decay_probe: float = 1e-7
    warmup_epochs: int = 15

    # Loss
    # Paper suggests a robust default around 0.05 for ImageNet-scale runs.
    # (Keep configurable; smaller values like 0.02 can work too.)
    lambda_sigreg: float = 0.05
    num_views: int = 8  # 2 Global + 6 Local (Updated)
    num_global_views: int = 2  # Paper: prediction targets come from global views only

    # Multi-Crop Parameters
    local_crops_number: int = 6
    local_crops_size: int = 96
    local_crops_scale: tuple = (0.05, 0.3)  # Paper: (0.05, 0.3)
    global_crops_scale: tuple = (0.3, 1.0)  # Paper: (0.3, 1.0)

    # SIGReg parameters
    # Paper default: sliced Epps–Pulley (univariate test) with ~1024 slices.
    sigreg_multivariate: bool = False  # If True, use multivariate CF variant instead
    sigreg_num_slices: int = 1024  # Number of random 1D projections (slices)
    sigreg_num_frequencies: int = 256  # For multivariate mode
    sigreg_sigma: float = 1.0  # Frequency scale for multivariate mode
    sigreg_num_knots: int = 17  # For univariate mode
    sigreg_max_t: float = 3.0  # For univariate mode
    sigreg_clip_value: float | None = None  # Optional: clip tiny slice stats to 0

    # Optional: SWA/EMA teacher for global centers (paper notes small ViT gains)
    teacher_student: bool = False
    teacher_base_ema: float = 0.996
    teacher_final_ema: float = 1.0

    # Dataset
    dataset: str = "frgfm/imagenette"
    dataset_config: str = "160px"  # "160px", "320px", or "full"
    num_workers: int = 8

    # Device
    device: str = "cuda"
    mixed_precision: bool = True

    # Logging
    log_interval: int = 100
    eval_interval: int = 1  # Evaluate every N epochs
    save_interval: int = 10
    output_dir: str = "outputs"
    wandb_project: str = "lejepa-jit"
    wandb_entity: str | None = None
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    use_wandb: bool = True

    # =============================================================================
    # Diagnostics / Visualization cadence
    # Note: any interval <= 0 disables that diagnostic.
    # =============================================================================

    # Core visualizations (lightweight, most useful for qualitative progress)
    pca_vis_interval: int = 5  # patch-PCA map (spatial semantics)
    attn_rollout_interval: int = 5  # attention rollout overlay

    # Redundant/optional visuals (disabled by default; enable when investigating)
    attn_grid_interval: int = 0  # raw last-layer attention (often redundant with rollout)
    head_importance_interval: int = 0  # older entropy/max/variance heatmap (replaced by per-head entropy)
    attention_difference_interval: int = 0  # epoch-to-epoch diff tracker (assumes CLS; can be noisy)
    training_dashboard_interval: int = 0  # WandB already plots scalars; image is redundant
    embedding_projection_interval: int = 0  # t-SNE/UMAP (expensive + noisy)

    # Embedding-space visuals for benchmarking
    embedding_pca_scatter_interval: int = 10  # class separation trend (stable)

    # Attention/representation visuals (moderately expensive)
    layer_attention_interval: int = 25
    per_head_attention_interval: int = 50
    token_similarity_interval: int = 50
    rsm_interval: int = 50
    collapse_monitor_interval: int = 5
    gradient_flow_interval: int = 10
    drift_interval: int = 5

    # PCA visualization controls (output only; does not affect training)
    pca_vis_size: int = 224
    pca_resample: str = "nearest"  # "nearest" | "bilinear"
    pca_per_image: bool = False

    # Optional GIF logging (can create lots of artifacts)
    pca_gif_interval: int = 0
    attn_gif_interval: int = 0

    # Representation eval metrics (epoch-level)
    knn_interval: int = 1
    knn_k: int = 20
    knn_temperature: float = 0.07
    knn_max_samples: int = 5000
    lid_interval: int = 1
    lid_max_samples: int = 1024

    # Transformer diagnostics (epoch-level on fixed batch)
    transformer_diag_interval: int = 5
    block_diag_interval: int = 10

    # Expensive diagnostics (epoch-level; can be slow)
    diagnostic_batch_size: int = 64
    # Smaller batch for very heavy diagnostics (Hessian/landscapes/head ablation).
    heavy_diagnostic_batch_size: int = 8
    # Skip heavy diagnostics if CUDA free memory is below this threshold (MiB).
    heavy_diag_min_free_mb: int = 1024
    gns_interval: int = 25
    gns_microbatches: int = 4
    sharpness_interval: int = 25
    sharpness_power_iters: int = 10
    landscape_interval: int = 50
    landscape_radius: float = 0.05
    landscape_points: int = 11

    # Additional heavy diagnostics (epoch-level; can be slow)
    attn_distance_headmap_interval: int = 10
    attn_entropy_headmap_interval: int = 10
    pocp_interval: int = 25  # RoPE POCP (layer×head + distance curves)
    pocp_num_pairs: int = 512
    pocp_num_distance_bins: int = 6
    pocp_freq_planes: int = 32
    attn_logits_interval: int = 5
    mlp_output_stats_interval: int = 5
    head_ablation_interval: int = 0
    head_ablation_layers: int = 1  # number of last layers to ablate
    landscape2d_interval: int = 100
    landscape2d_radius: float = 0.05
    landscape2d_points: int = 11

    # Seed
    seed: int = 42


def get_config(**kwargs) -> Config:
    """Create config with optional overrides."""
    return Config(**kwargs)
