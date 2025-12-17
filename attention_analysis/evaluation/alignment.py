"""
Alignment Evaluation Metrics.

Measures how well attributions align with human-annotated ground truth regions.

Key Metrics:
- Pointing Game: Whether the max attribution is inside the GT bounding box
- Mean IoU: Intersection over Union with GT segmentation masks

Note: These require ground truth annotations (segmentation masks or bounding boxes).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


@dataclass
class AlignmentResult:
    """Container for alignment evaluation results."""

    pointing_accuracy: float
    mean_iou: float
    threshold_used: float
    num_samples: int


def compute_pointing_game(
    attribution: torch.Tensor,
    bbox: tuple[int, int, int, int] | list[tuple[int, int, int, int]],
    tolerance: int = 0,
) -> bool:
    """
    Compute Pointing Game accuracy for a single image.

    Checks if the maximum attribution point falls inside the ground truth
    bounding box.

    Args:
        attribution: Attribution map, shape (H, W).
        bbox: Bounding box as (x_min, y_min, x_max, y_max) or list of boxes.
        tolerance: Pixels of tolerance around the box.

    Returns:
        True if max attribution is inside bbox.
    """
    # Find max attribution location
    flat_idx = attribution.flatten().argmax()
    h, w = attribution.shape
    max_y = (flat_idx // w).item()
    max_x = (flat_idx % w).item()

    # Handle single box or multiple boxes
    if isinstance(bbox[0], (int, float)):
        boxes = [bbox]
    else:
        boxes = bbox

    # Check if point is inside any box
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        x_min -= tolerance
        y_min -= tolerance
        x_max += tolerance
        y_max += tolerance

        if x_min <= max_x <= x_max and y_min <= max_y <= y_max:
            return True

    return False


def compute_iou(
    attribution: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Compute Intersection over Union between attribution and ground truth mask.

    Args:
        attribution: Attribution map, shape (H, W). Will be thresholded.
        mask: Binary ground truth mask, shape (H, W).
        threshold: Threshold for binarizing attribution.

    Returns:
        IoU value in [0, 1].
    """
    # Ensure same size
    if attribution.shape != mask.shape:
        attribution = F.interpolate(
            attribution.unsqueeze(0).unsqueeze(0).float(),
            size=mask.shape,
            mode="bilinear",
            align_corners=False,
        ).squeeze()

    # Normalize attribution to [0, 1]
    attr_min = attribution.min()
    attr_max = attribution.max()
    if attr_max - attr_min > 1e-8:
        attribution = (attribution - attr_min) / (attr_max - attr_min)
    else:
        attribution = torch.zeros_like(attribution)

    # Binarize
    attr_binary = (attribution > threshold).float()
    mask_binary = (mask > 0.5).float()

    # Compute IoU
    intersection = (attr_binary * mask_binary).sum()
    union = ((attr_binary + mask_binary) > 0).float().sum()

    if union > 0:
        return (intersection / union).item()
    return 0.0


def compute_mean_iou(
    attributions: torch.Tensor,
    masks: torch.Tensor,
    threshold: float | str = "adaptive",
    show_progress: bool = False,
) -> float:
    """
    Compute Mean IoU across a batch of images.

    Args:
        attributions: Batch of attribution maps, shape (N, H, W).
        masks: Batch of ground truth masks, shape (N, H, W).
        threshold: Threshold for binarizing attributions.
            - float: Fixed threshold
            - "adaptive": Use Otsu's method per image
            - "top_k": Use top 20% of attribution values
        show_progress: Whether to show progress bar.

    Returns:
        Mean IoU value.
    """
    ious = []

    iterator = range(len(attributions))
    if show_progress:
        iterator = tqdm(iterator, desc="Computing IoU")

    for i in iterator:
        attr = attributions[i]
        mask = masks[i]

        if threshold == "adaptive":
            # Otsu's threshold
            thresh = _otsu_threshold(attr)
        elif threshold == "top_k":
            # Top 20% threshold
            flat = attr.flatten()
            k = int(0.2 * len(flat))
            thresh = (
                flat.kthvalue(len(flat) - k)[0].item()
                if k > 0
                else flat.median().item()
            )
        else:
            thresh = threshold

        iou = compute_iou(attr, mask, thresh)
        ious.append(iou)

    return sum(ious) / len(ious) if ious else 0.0


def _otsu_threshold(tensor: torch.Tensor) -> float:
    """Compute Otsu's threshold for a tensor."""
    # Convert to numpy for sklearn
    arr = tensor.flatten().cpu().numpy()

    # Normalize to [0, 255] for histogram
    arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)

    # Compute histogram
    hist, bin_edges = np.histogram(arr, bins=256, range=(0, 256))
    hist = hist.astype(float)

    # Compute cumulative sums
    total = hist.sum()
    cumsum = np.cumsum(hist)
    cumsum_val = np.cumsum(hist * np.arange(256))

    # Compute between-class variance
    mean_bg = cumsum_val / (cumsum + 1e-8)
    mean_fg = (cumsum_val[-1] - cumsum_val) / (total - cumsum + 1e-8)

    weight_bg = cumsum / total
    weight_fg = 1 - weight_bg

    variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

    # Find threshold that maximizes variance
    threshold = np.argmax(variance)

    return threshold / 255.0


def compute_pointing_game_batch(
    attributions: torch.Tensor,
    bboxes: list,
    show_progress: bool = False,
) -> float:
    """
    Compute Pointing Game accuracy across a batch.

    Args:
        attributions: Batch of attribution maps, shape (N, H, W).
        bboxes: List of bounding boxes, one per image.
            Each element is (x_min, y_min, x_max, y_max) or list of boxes.
        show_progress: Whether to show progress bar.

    Returns:
        Pointing Game accuracy in [0, 1].
    """
    hits = 0

    iterator = range(len(attributions))
    if show_progress:
        iterator = tqdm(iterator, desc="Pointing Game")

    for i in iterator:
        if compute_pointing_game(attributions[i], bboxes[i]):
            hits += 1

    return hits / len(attributions) if attributions.shape[0] > 0 else 0.0


def compute_alignment_metrics(
    attributions: torch.Tensor,
    masks: torch.Tensor | None = None,
    bboxes: list | None = None,
    threshold: float | str = "adaptive",
    show_progress: bool = False,
) -> dict:
    """
    Compute all alignment metrics.

    Args:
        attributions: Batch of attribution maps, shape (N, H, W).
        masks: Optional batch of ground truth masks, shape (N, H, W).
        bboxes: Optional list of bounding boxes.
        threshold: Threshold for IoU computation.
        show_progress: Whether to show progress bar.

    Returns:
        Dictionary with:
        - mean_iou: Mean IoU (if masks provided)
        - pointing_accuracy: Pointing game accuracy (if bboxes provided)
        - threshold_used: Threshold used for IoU
    """
    results = {"threshold_used": threshold}

    if masks is not None:
        results["mean_iou"] = compute_mean_iou(
            attributions, masks, threshold, show_progress
        )
    else:
        results["mean_iou"] = None

    if bboxes is not None:
        results["pointing_accuracy"] = compute_pointing_game_batch(
            attributions, bboxes, show_progress
        )
    else:
        results["pointing_accuracy"] = None

    return results


def evaluate_on_imagenet_seg(
    model,
    attribution_method,
    data_loader,
    num_samples: int = 500,
    show_progress: bool = True,
) -> AlignmentResult:
    """
    Evaluate attribution method on ImageNet-Segmentation dataset.

    This function assumes the data_loader yields (image, mask, label) tuples
    from the ImageNet-Segmentation dataset.

    Args:
        model: The classification model.
        attribution_method: An AttributionMethod instance.
        data_loader: DataLoader for ImageNet-Segmentation.
        num_samples: Maximum number of samples to evaluate.
        show_progress: Whether to show progress bar.

    Returns:
        AlignmentResult with mean IoU and pointing accuracy.
    """
    all_ious = []
    hits = 0
    total = 0

    iterator = data_loader
    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(data_loader, desc="Evaluating")

    for batch_idx, batch in enumerate(iterator):
        if total >= num_samples:
            break

        if len(batch) == 3:
            images, masks, labels = batch
        else:
            images, masks = batch[:2]
            labels = None

        # Compute attributions
        for i in range(len(images)):
            if total >= num_samples:
                break

            image = images[i].unsqueeze(0)
            mask = masks[i]

            # Get attribution
            target = labels[i].item() if labels is not None else None
            attribution = attribution_method(model, image, target_class=target)

            # Resize attribution to mask size if needed
            if attribution.shape != mask.shape:
                attribution = F.interpolate(
                    attribution.unsqueeze(0).unsqueeze(0).float(),
                    size=mask.shape,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()

            # Compute IoU
            iou = compute_iou(attribution, mask, threshold="adaptive")
            all_ious.append(iou)

            # Compute pointing game (use mask as bbox approximation)
            # Find bounding box from mask
            if mask.any():
                nonzero = mask.nonzero()
                y_min, x_min = nonzero.min(dim=0)[0].tolist()
                y_max, x_max = nonzero.max(dim=0)[0].tolist()
                bbox = (x_min, y_min, x_max, y_max)

                if compute_pointing_game(attribution, bbox):
                    hits += 1

            total += 1

    return AlignmentResult(
        pointing_accuracy=hits / total if total > 0 else 0.0,
        mean_iou=sum(all_ious) / len(all_ious) if all_ious else 0.0,
        threshold_used="adaptive",
        num_samples=total,
    )
