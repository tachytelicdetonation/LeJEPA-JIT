# Implementation Plan: Full LeJEPA Paper Specification

## Goal
Upgrade the current "Minimal" LeJEPA implementation to match the **Full Paper Specification** as closely as possible, enabling fair comparison.

## User Review Required
> [!IMPORTANT]
> **Breaking Change:** The dataset will now return mixed-resolution batches (Global views 224x224, Local views 96x96). This requires significant updates to the training loop.
> **ViT Interpolation:** ViTEncoder will be updated to support variable input sizes via positional embedding interpolation.

## Proposed Changes

### 1. Configuration (`config.py`)
#### [MODIFY] [config.py](file:///Users/tanmaydeshmukh/Projects/LeJEPA-JIT/config.py)
-   Update `img_size` to `224`.
-   Update `lr_encoder` to `5e-4`.
-   Add Multi-Crop params:
    -   `local_crops_number = 6`
    -   `local_crops_size = 96` (96 is divisible by 8/16, safer than 98)
    -   `local_crops_scale = (0.05, 0.3)`
    -   `global_crops_scale = (0.3, 1.0)`

### 2. Dataset (`data/dataset.py`)
#### [MODIFY] [dataset.py](file:///Users/tanmaydeshmukh/Projects/LeJEPA-JIT/data/dataset.py)
-   Implement `DataAugmentationLEJEPA` class (inspired by DINO).
-   Return `global_views` (List[Tensor]) and `local_views` (List[Tensor]).
-   Update `collate_fn` or usage in `train.py` to handle lists.

### 3. Encoders (`models/vit_encoder.py`, `models/jit_encoder.py`)
#### [MODIFY] [vit_encoder.py](file:///Users/tanmaydeshmukh/Projects/LeJEPA-JIT/models/vit_encoder.py)
-   Implement `interpolate_pos_encoding` to handle 96x96 inputs.
-   Update `forward` to allow returning features from last N layers.

#### [MODIFY] [jit_encoder.py](file:///Users/tanmaydeshmukh/Projects/LeJEPA-JIT/models/jit_encoder.py)
-   Update `forward` to allow returning features from last N layers. (RoPE handles size automatically).

### 4. Training Logic (`train.py`, `models/lejepa.py`)
#### [MODIFY] [lejepa.py](file:///Users/tanmaydeshmukh/Projects/LeJEPA-JIT/models/lejepa.py)
-   Update `get_embedding` to concatenate last 2 layers (avg/cls).

#### [MODIFY] [train.py](file:///Users/tanmaydeshmukh/Projects/LeJEPA-JIT/train.py)
-   Update data loading to unpack global/local views.
-   Forward global and local views separately through encoder.
-   Concatenate projections for loss calculation.
-   Update Invariance Loss: potentially match global-to-local or global-to-global. *Decision: Paper uses simple invariance on all views in LEJEPA formulation.*

## Verification Plan

### Automated Tests
1.  **Shape Check:** Verify data loader returns correct shapes (224x224 and 96x96).
2.  **Forward Pass:** Verify ViT/JiT handle 96x96 inputs without error.
3.  **Probe:** Verify linear probe receives concatenated feature dim (e.g., 512*2 = 1024).

### Manual Verification
-   Run 1 epoch of training to verify stability and loss convergence.
