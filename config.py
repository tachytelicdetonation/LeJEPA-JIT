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
    proj_dim: int = 16  # Reference uses 16

    # Training
    batch_size: int = 256
    epochs: int = 100
    lr_encoder: float = 5e-4  # Paper value
    lr_probe: float = 1e-3
    weight_decay_encoder: float = 5e-2
    weight_decay_probe: float = 1e-7
    warmup_epochs: int = 15

    # Loss
    lambda_sigreg: float = 0.02
    num_views: int = 8  # 2 Global + 6 Local (Updated)

    # Multi-Crop Parameters
    local_crops_number: int = 6
    local_crops_size: int = 96
    local_crops_scale: tuple = (0.05, 0.4)
    global_crops_scale: tuple = (0.4, 1.0)

    # SIGReg parameters
    sigreg_multivariate: bool = True  # Use multivariate SIGReg (default)
    sigreg_num_frequencies: int = 256  # For multivariate mode
    sigreg_sigma: float = 1.0  # Frequency scale for multivariate mode
    sigreg_num_knots: int = 17  # For univariate mode
    sigreg_max_t: float = 3.0  # For univariate mode

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
    use_wandb: bool = True

    # Visualization intervals (default: every epoch)
    layer_attention_interval: int = 1
    per_head_attention_interval: int = 1
    head_importance_interval: int = 1
    token_similarity_interval: int = 1
    rsm_interval: int = 1
    collapse_monitor_interval: int = 1
    gradient_flow_interval: int = 1
    training_dashboard_interval: int = 1
    embedding_projection_interval: int = 1
    drift_interval: int = 1

    # PCA visualization controls (output only; does not affect training)
    pca_vis_size: int = 224
    pca_resample: str = "nearest"  # "nearest" | "bilinear"
    pca_per_image: bool = False

    # Representation eval metrics (epoch-level)
    knn_interval: int = 1
    knn_k: int = 20
    knn_temperature: float = 0.07
    knn_max_samples: int = 5000
    lid_interval: int = 1
    lid_max_samples: int = 1024

    # Transformer diagnostics (epoch-level on fixed batch)
    transformer_diag_interval: int = 1
    block_diag_interval: int = 1

    # Expensive diagnostics (epoch-level; can be slow)
    diagnostic_batch_size: int = 64
    gns_interval: int = 1
    gns_microbatches: int = 4
    sharpness_interval: int = 1
    sharpness_power_iters: int = 10
    landscape_interval: int = 1
    landscape_radius: float = 0.05
    landscape_points: int = 11

    # Additional heavy diagnostics (epoch-level; can be slow)
    attn_distance_headmap_interval: int = 1
    attn_logits_interval: int = 5
    mlp_output_stats_interval: int = 5
    head_ablation_interval: int = 10
    head_ablation_layers: int = 1  # number of last layers to ablate
    landscape2d_interval: int = 10
    landscape2d_radius: float = 0.05
    landscape2d_points: int = 11

    # Seed
    seed: int = 42


def get_config(**kwargs) -> Config:
    """Create config with optional overrides."""
    return Config(**kwargs)
