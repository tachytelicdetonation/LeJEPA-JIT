# LeJEPA-JiT

Self-supervised representation learning comparing **JiT** (Just image Transformer) and **ViT** (Vision Transformer) encoders using the LeJEPA framework.

## Overview

This project integrates the JiT architecture from ["Back to Basics: Let Denoising Generative Models Denoise"](https://arxiv.org/abs/2511.13720) with LeJEPA's self-supervised learning framework for benchmark comparison against standard ViT.

### Key Features

**JiT Encoder** (adapted from diffusion model):
- BottleneckPatchEmbed: Two-stage patch embedding
- RoPE: Rotary Position Embeddings (Dynamic 2D)
- SwiGLU FFN: More efficient feed-forward
- RMSNorm on Q/K: Improved attention stability

**ViT Encoder** (baseline):
- Standard patch embedding (with interpolation for mixed resolutions)
- Learned positional embeddings
- Standard GELU MLP
- LayerNorm

**LeJEPA Framework**:
- SIGReg loss (characteristic function matching)
- Invariance loss
- **Multi-Crop Augmentation**: 2 Global + 6 Local views per image
- No stop-gradients or teacher-student setup
- Online linear probe for monitoring

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/LeJEPA-JIT.git
cd LeJEPA-JIT

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
# Train with JiT encoder (default)
python train.py --encoder jit

# Train with ViT encoder for comparison
python train.py --encoder vit

# Custom configuration (Low VRAM example)
python train.py \
    --encoder jit \
    --batch_size 128 \
    --lr 5e-4
```

### Evaluation

```bash
# Evaluate a single model
python evaluate.py --checkpoint outputs/jit_xxx/best_model.pt

# Compare JiT vs ViT
python evaluate.py \
    --checkpoint_jit outputs/jit_xxx/best_model.pt \
    --checkpoint_vit outputs/vit_xxx/best_model.pt
```

## Project Structure

```
LeJEPA-JIT/
    models/
        __init__.py
        jit_encoder.py      # JiT backbone (RoPE, SwiGLU, BottleneckPatchEmbed)
        vit_encoder.py      # Standard ViT baseline
        lejepa.py           # LeJEPA wrapper with projector
    losses/
        __init__.py
        sigreg.py           # SIGReg + invariance loss
    data/
        __init__.py
        dataset.py          # ImageNette Dataset + Multi-Crop Augmentation
    train.py                # Training script
    evaluate.py             # Evaluation script
    config.py               # Configuration
    requirements.txt        # Dependencies
    README.md
```

## Configuration

Key hyperparameters (see `config.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `encoder` | `jit` | Encoder type (`jit` or `vit`) |
| `img_size` | `224` | Input image size (standard ImageNet) |
| `patch_size` | `16` | Patch size |
| `embed_dim` | `384` | Embedding dimension (ViT-Small) |
| `depth` | `12` | Number of transformer layers |
| `num_heads` | `6` | Number of attention heads |
| `batch_size` | `256` | Training batch size |
| `epochs` | `800` | Number of training epochs |
| `lr_encoder` | `5e-4` | Learning rate for encoder |
| `num_views` | `8` | 2 Global (224px) + 6 Local (96px) views |
| `lambda_sigreg` | `0.02` | Balance between SIGReg and invariance |

## Benchmark Results

| Encoder | Linear Probe Acc | Parameters |
|---------|-----------------|------------|
| ViT-Small | ~90.7% | ~22M |
| JiT-Small | TBD | ~22M |

## References

- **JiT**: [Back to Basics: Let Denoising Generative Models Denoise](https://arxiv.org/abs/2511.13720) (Li & He, 2025)
- **LeJEPA**: [Learning Joint-Embeddings with Prediction Agents](https://github.com/rbalestr-lab/lejepa)

## License

MIT License
