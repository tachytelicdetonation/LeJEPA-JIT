"""
Faithfulness Evaluation Metrics.

Measures whether attributions truly reflect what the model uses for predictions.

Key Metrics:
- Insertion AUC (iAUC): Start from baseline, progressively reveal top-attributed pixels
  Higher = Better (relevant pixels should increase confidence quickly)

- Deletion AUC (dAUC): Start from original, progressively remove top-attributed pixels
  Lower = Better (removing relevant pixels should decrease confidence quickly)

Reference:
    "RISE: Randomized Input Sampling for Explanation of Black-box Models" (BMVC 2018)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from attention_analysis.utils.data import (
    create_baseline_image,
    apply_mask_to_image,
    patches_to_image_mask,
)
from attention_analysis.utils.model_utils import get_model_info


@dataclass
class FaithfulnessResult:
    """Container for faithfulness evaluation results."""

    insertion_auc: float
    deletion_auc: float
    insertion_curve: list[float]
    deletion_curve: list[float]
    combined_score: float  # iAUC - dAUC (higher = better)


class InsertionDeletionEvaluator:
    """
    Evaluator for insertion and deletion faithfulness metrics.

    Measures how well attributions correlate with model behavior by
    progressively revealing/removing attributed regions and tracking
    confidence changes.

    Args:
        model: The classification model.
        num_steps: Number of steps for the curve (default 100).
        baseline_type: Type of baseline for insertion. Options:
            - "blur": Gaussian blur (default)
            - "black": All zeros
            - "mean": Mean pixel value
            - "noise": Random noise
        blur_sigma: Sigma for Gaussian blur baseline.
        batch_size: Batch size for efficiency.
    """

    def __init__(
        self,
        model: nn.Module,
        num_steps: int = 100,
        baseline_type: str = "blur",
        blur_sigma: float = 15.0,
        batch_size: int = 8,
    ):
        self.model = model
        self.num_steps = num_steps
        self.baseline_type = baseline_type
        self.blur_sigma = blur_sigma
        self.batch_size = batch_size
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def evaluate(
        self,
        image: torch.Tensor,
        attribution: torch.Tensor,
        target_class: int | None = None,
    ) -> FaithfulnessResult:
        """
        Compute insertion and deletion AUC for an attribution.

        Args:
            image: Original image tensor, shape (C, H, W) or (B, C, H, W).
            attribution: Attribution map, shape (H, W) or (H_patches, W_patches).
            target_class: Target class to track. If None, uses predicted class.

        Returns:
            FaithfulnessResult with AUC values and curves.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)

        # Get model info
        info = get_model_info(self.model)
        patch_size = info["patch_size"]
        img_size = image.shape[-1]

        # Determine if attribution is patch-level or pixel-level
        if attribution.shape[-1] < img_size // 2:
            # Patch-level attribution - convert to pixel-level
            attribution = patches_to_image_mask(attribution, patch_size, img_size)
        attribution = attribution.to(self.device)

        # Flatten and sort attribution
        flat_attr = attribution.flatten()
        num_pixels = len(flat_attr)
        sorted_indices = torch.argsort(flat_attr, descending=True)

        # Create baseline for insertion
        baseline = create_baseline_image(image, self.baseline_type, self.blur_sigma)

        # Get original prediction
        output = self._get_model_output(image)
        if target_class is None:
            target_class = output.argmax(dim=-1)[0].item()

        # Compute curves
        insertion_curve = []
        deletion_curve = []

        # Generate step sizes
        step_size = num_pixels // self.num_steps
        if step_size == 0:
            step_size = 1

        for step in range(self.num_steps + 1):
            k = min(step * step_size, num_pixels)

            # Create masks
            revealed_indices = sorted_indices[:k]
            insertion_mask = torch.zeros_like(flat_attr)
            if k > 0:
                insertion_mask[revealed_indices] = 1.0
            insertion_mask = insertion_mask.view_as(attribution)

            deletion_mask = torch.ones_like(flat_attr)
            if k > 0:
                deletion_mask[revealed_indices] = 0.0
            deletion_mask = deletion_mask.view_as(attribution)

            # Apply masks
            insertion_image = apply_mask_to_image(
                image, insertion_mask.unsqueeze(0), baseline
            )
            deletion_image = apply_mask_to_image(
                image, deletion_mask.unsqueeze(0), baseline
            )

            # Get confidences
            ins_output = self._get_model_output(insertion_image)
            del_output = self._get_model_output(deletion_image)

            ins_conf = F.softmax(ins_output, dim=-1)[0, target_class].item()
            del_conf = F.softmax(del_output, dim=-1)[0, target_class].item()

            insertion_curve.append(ins_conf)
            deletion_curve.append(del_conf)

        # Compute AUC (normalized to [0, 1])
        insertion_auc = sum(insertion_curve) / len(insertion_curve)
        deletion_auc = sum(deletion_curve) / len(deletion_curve)

        return FaithfulnessResult(
            insertion_auc=insertion_auc,
            deletion_auc=deletion_auc,
            insertion_curve=insertion_curve,
            deletion_curve=deletion_curve,
            combined_score=insertion_auc - deletion_auc,
        )

    def _get_model_output(self, image: torch.Tensor) -> torch.Tensor:
        """Get model output logits."""
        self.model.eval()

        if hasattr(self.model, "encoder"):
            embeddings = self.model.encoder(image)
            if hasattr(self.model, "probe"):
                return self.model.probe(embeddings)
            return embeddings
        else:
            output = self.model(image)
            if hasattr(output, "logits"):
                return output.logits
            return output


def compute_insertion_auc(
    model: nn.Module,
    image: torch.Tensor,
    attribution: torch.Tensor,
    target_class: int | None = None,
    num_steps: int = 100,
    baseline_type: str = "blur",
) -> float:
    """
    Compute Insertion AUC for a single attribution.

    Higher = Better. Measures how quickly confidence rises when
    revealing attributed regions.

    Args:
        model: The classification model.
        image: Original image tensor.
        attribution: Attribution map.
        target_class: Target class (None = predicted class).
        num_steps: Number of steps for the curve.
        baseline_type: Type of baseline ("blur", "black", "mean").

    Returns:
        Insertion AUC value in [0, 1].
    """
    evaluator = InsertionDeletionEvaluator(
        model, num_steps=num_steps, baseline_type=baseline_type
    )
    result = evaluator.evaluate(image, attribution, target_class)
    return result.insertion_auc


def compute_deletion_auc(
    model: nn.Module,
    image: torch.Tensor,
    attribution: torch.Tensor,
    target_class: int | None = None,
    num_steps: int = 100,
    baseline_type: str = "blur",
) -> float:
    """
    Compute Deletion AUC for a single attribution.

    Lower = Better. Measures how quickly confidence drops when
    removing attributed regions.

    Args:
        model: The classification model.
        image: Original image tensor.
        attribution: Attribution map.
        target_class: Target class (None = predicted class).
        num_steps: Number of steps for the curve.
        baseline_type: Type of baseline ("blur", "black", "mean").

    Returns:
        Deletion AUC value in [0, 1].
    """
    evaluator = InsertionDeletionEvaluator(
        model, num_steps=num_steps, baseline_type=baseline_type
    )
    result = evaluator.evaluate(image, attribution, target_class)
    return result.deletion_auc


def compute_faithfulness_metrics(
    model: nn.Module,
    images: torch.Tensor,
    attributions: torch.Tensor,
    target_classes: list[int] | None = None,
    num_steps: int = 100,
    baseline_type: str = "blur",
    show_progress: bool = True,
) -> dict:
    """
    Compute faithfulness metrics over a batch of images.

    Args:
        model: The classification model.
        images: Batch of images, shape (N, C, H, W).
        attributions: Batch of attribution maps, shape (N, H, W).
        target_classes: Target classes for each image (None = predicted).
        num_steps: Number of steps for curves.
        baseline_type: Type of baseline.
        show_progress: Whether to show progress bar.

    Returns:
        Dictionary with:
        - mean_insertion_auc: Mean insertion AUC
        - mean_deletion_auc: Mean deletion AUC
        - mean_combined: Mean combined score (iAUC - dAUC)
        - std_insertion_auc: Std of insertion AUC
        - std_deletion_auc: Std of deletion AUC
        - all_results: List of individual FaithfulnessResult objects
    """
    evaluator = InsertionDeletionEvaluator(
        model, num_steps=num_steps, baseline_type=baseline_type
    )

    results = []
    iterator = range(len(images))
    if show_progress:
        iterator = tqdm(iterator, desc="Computing faithfulness")

    for i in iterator:
        image = images[i]
        attr = attributions[i]
        target = target_classes[i] if target_classes else None

        result = evaluator.evaluate(image, attr, target)
        results.append(result)

    # Aggregate
    insertion_aucs = [r.insertion_auc for r in results]
    deletion_aucs = [r.deletion_auc for r in results]
    combined_scores = [r.combined_score for r in results]

    return {
        "mean_insertion_auc": sum(insertion_aucs) / len(insertion_aucs),
        "mean_deletion_auc": sum(deletion_aucs) / len(deletion_aucs),
        "mean_combined": sum(combined_scores) / len(combined_scores),
        "std_insertion_auc": torch.tensor(insertion_aucs).std().item(),
        "std_deletion_auc": torch.tensor(deletion_aucs).std().item(),
        "all_results": results,
    }


def compare_methods_faithfulness(
    model: nn.Module,
    images: torch.Tensor,
    method_attributions: dict[str, torch.Tensor],
    target_classes: list[int] | None = None,
    num_steps: int = 100,
    show_progress: bool = True,
) -> dict[str, dict]:
    """
    Compare faithfulness of multiple attribution methods.

    Args:
        model: The classification model.
        images: Batch of images.
        method_attributions: Dict mapping method names to attribution batches.
        target_classes: Target classes.
        num_steps: Number of steps for curves.
        show_progress: Whether to show progress bar.

    Returns:
        Dict mapping method names to their faithfulness metrics.
    """
    results = {}

    for method_name, attributions in method_attributions.items():
        if show_progress:
            print(f"Evaluating {method_name}...")

        results[method_name] = compute_faithfulness_metrics(
            model,
            images,
            attributions,
            target_classes,
            num_steps,
            show_progress=show_progress,
        )

    return results
