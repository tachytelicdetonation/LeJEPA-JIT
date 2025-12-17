"""
Evaluation metrics for attention attribution methods.

This module provides metrics for evaluating the quality of attention
attributions, including faithfulness and alignment metrics.

Faithfulness Metrics:
- Insertion AUC: How quickly does confidence rise when revealing attributed pixels?
- Deletion AUC: How quickly does confidence drop when removing attributed pixels?

Alignment Metrics:
- Mean IoU: Overlap with ground truth segmentation masks
- Pointing Game: Whether max attribution is inside ground truth region
"""

from attention_analysis.evaluation.faithfulness import (
    compute_insertion_auc,
    compute_deletion_auc,
    compute_faithfulness_metrics,
    InsertionDeletionEvaluator,
)
from attention_analysis.evaluation.alignment import (
    compute_pointing_game,
    compute_mean_iou,
    compute_alignment_metrics,
)

__all__ = [
    # Faithfulness
    "compute_insertion_auc",
    "compute_deletion_auc",
    "compute_faithfulness_metrics",
    "InsertionDeletionEvaluator",
    # Alignment
    "compute_pointing_game",
    "compute_mean_iou",
    "compute_alignment_metrics",
]
