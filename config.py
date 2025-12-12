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
    epochs: int = 800
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
    sigreg_num_knots: int = 17
    sigreg_max_t: float = 3.0

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

    # Seed
    seed: int = 42


def get_config(**kwargs) -> Config:
    """Create config with optional overrides."""
    return Config(**kwargs)
