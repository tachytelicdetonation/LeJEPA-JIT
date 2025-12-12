# Gap Analysis: Current vs Full "Paper" Implementation

You asked about comparing with "what the paper has" (Non-Minimal). The current implementation aligns with the **Minimal Example**, but differs significantly from the **Full Paper Specification** described in the `README`.

## 1. Evaluation Metric (Critical)
- **Paper:** Concatenates CLS tokens from the **last two layers** for linear probing.
  > "Feature Extraction: Concatenation of the CLS token from the last two layers"
- **Current:** Uses only the **final layer** output.
- **Impact:** The paper likely gains performance boost (1-2%) from this multi-layer fusion. To match, we must modify the Encoders to return intermediate features.

## 2. Augmentations (Multi-Crop)
- **Paper:** Uses a specific **Multi-Crop** strategy:
  - **2 Global Views:** 224x224 (scale 0.3-1.0)
  - **6 Local Views:** 98x98 (scale 0.05-0.3)
- **Current:** Uses `num_views=4` identical views (128x128).
- **Impact:** Multi-crop is a standard technique (from DINO/SwAV) to improve efficiency and performance. Local views force the model to solve "part-to-whole" problems.

## 3. Resolution & Dataset
- **Paper:** ImageNet-1k at **224x224**.
- **Current:** ImageNette at **128x128**.
- **Impact:** 128px is too small for standard ViT comparisons (typically 224px). If staying on ImageNette, we should arguably still use 224px to match the patch dynamics (14x14 grid vs 8x8 grid).

## 4. Hyperparameters
- **Learning Rate:** Paper uses `5e-4`. Current uses `2e-3`.
- **Weight Decay:** Paper uses `5e-2`. Current matches `5e-2`.
- **Batch Size:** Paper uses `512+`. Current `256` (corrected).

## Recommendation

To strictly "compare with the paper" using your current setup (ImageNette), I recommend upgrading your code to the **"Paper-lite"** specification:

1.  **Modify Encoder:** Return concatenated features from last 2 layers.
2.  **Implement Multi-Crop:** Update `dataset.py` to return 2 global + N local views.
3.  **Increase Resolution:** Switch to `224px`.
4.  **Tune Hyperparameters:** Lower LR to `5e-4`.

Shall I proceed with these upgrades?
