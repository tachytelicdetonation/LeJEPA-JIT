# LeJEPA-JIT vs Reference Comparison Report

## 1. Overview
This report compares the current `LeJEPA-JIT` implementation against the provided implementation in `ref/lejepa/MINIMAL.md`.

## 2. Dataset and Augmentations
**Status: ✅ MATCH**
- **Dataset:** Both use `frgfm/imagenette` (160px).
- **Augmentations:** The augmentation pipeline in `data/dataset.py` exactly matches the "strong" augmentations in the reference, including:
  - `RandomResizedCrop(scale=(0.08, 1.0))`
  - `ColorJitter` & `RandomGrayscale`
  - `GaussianBlur` & `RandomSolarize`
  - `RandomHorizontalFlip`

## 3. Metrics and Evaluation
**Status: ✅ MATCH / TRAITS**
- **Metrics:**
  - **Online Linear Probe:** Implemented in `train.py`, matches standard LeJEPA monitoring.
  - **Offline Linear Probe:** Implemented in `evaluate.py`.
  - **k-NN Accuracy:** Implemented in `evaluate.py`. While the minimal reference focuses on Linear Probe, k-NN is the standard metric for JEPAs (e.g., Lightly benchmark mentioned in reference) and is correctly included here.
- **Evaluation Protocol:**
  - Training on Train set with augmentations.
  - Evaluating on Validation set with standard resize/crop.

## 4. Hyperparameters
**Status: ⚠️ MINOR DISCREPANCY**

| Hyperparameter | Reference (MINIMAL.md) | Current Config (`config.py`) | Recommendation |
| :--- | :--- | :--- | :--- |
| `epochs` | 800 | 800 | ✅ Keep |
| `lr_encoder` | 2e-3 | 2e-3 | ✅ Keep |
| `weight_decay` | 5e-2 | 5e-2 | ✅ Keep |
| `lambda_sigreg`| 0.02 | 0.02 | ✅ Keep |
| `proj_dim` | 16 (cmd examle) | 16 | ✅ Keep |
| `batch_size` | **256** | **64** | ⚠️ **Change to 256** |

**Note on Batch Size:** The configuration uses `64`, while the reference strongly suggests `256` and notes it was chosen to match benchmarks. Using `64` might affect the stability of the gradients or the specific dynamics of LeJEPA, though the reference claims LeJEPA is robust. If GPU memory allows, **increase to 256**.

## 5. Model Architecture
**Status: ✅ MATCH**
- **ViT Encoder:** The `models/vit_encoder.py` uses `embed_dim=512`, which aligns with the reference's use of `timm ... num_classes=512`.
- **JiT Encoder:** The adaptation in `models/jit_encoder.py` correctly preserves the core components of JiT (RoPE, SwiGLU, Bottleneck Patch Embed) while removing the generative heads, making it a valid encoder for this comparison.

## 6. Conclusion
The implementation is scientifically sound and correctly aligned with the reference minimal examples. The metrics are appropriate for comparing JiT vs ViT.

**To get comparable results:**
1.  **Update Batch Size:** Change `batch_size` in `config.py` to `256`.
2.  **Train JiT:** Run `python train.py --encoder jit`.
3.  **Train ViT:** Run `python train.py --encoder vit` (looks like one run already exists).
4.  **Compare:** Run `python evaluate.py --checkpoint_jit outputs/.../best_model.pt --checkpoint_vit outputs/.../best_model.pt`.
