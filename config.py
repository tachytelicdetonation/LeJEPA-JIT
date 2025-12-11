"""Configuration for LeJEPA-JiT training."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class Config:
    # Model selection
    encoder: Literal["jit", "vit"] = "jit"

    # Image and patch settings
    img_size: int = 128
    patch_size: int = 8
    in_channels: int = 3

    # Encoder architecture
    embed_dim: int = 512
    depth: int = 12
    num_heads: int = 8
    mlp_ratio: float = 4.0

    # JiT-specific
    bottleneck_dim: int = 128
    use_rope: bool = True
    use_swiglu: bool = True

    # Projector
    proj_hidden_dim: int = 2048
    proj_dim: int = 2048

    # Training
    batch_size: int = 256
    epochs: int = 200
    lr_encoder: float = 2e-3
    lr_probe: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 10

    # Loss
    lambda_sigreg: float = 0.5  # Balance between SIGReg and invariance loss
    num_views: int = 2  # Number of augmented views per sample

    # SIGReg parameters
    sigreg_num_knots: int = 17
    sigreg_max_t: float = 3.0

    # Dataset
    dataset: str = "frgfm/imagenette"
    dataset_split: str = "full"  # or "320px", "160px"
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
    use_wandb: bool = False

    # Seed
    seed: int = 42


def get_config(**kwargs) -> Config:
    """Create config with optional overrides."""
    return Config(**kwargs)
