"""
Attention Analysis Module for Vision Transformers.

This module provides comprehensive tools for attention visualization,
attribution methods, and evaluation metrics for Vision Transformers.

Main Components:
- methods: Attribution methods (AttnLRP, Grad-CAM, Rollout, GMAR, etc.)
- evaluation: Faithfulness and alignment metrics
- visualization: Heatmap generation and comparison tools
- training_dynamics: Training-time attention tracking
- utils: Hook management and utilities
"""

from attention_analysis.utils.hooks import HookManager
from attention_analysis.methods import (
    get_attribution,
    list_methods,
    AttnLRP,
    GradCAM,
    AttentionRollout,
    GMAR,
    RawAttention,
)
from attention_analysis.evaluation import (
    compute_insertion_auc,
    compute_deletion_auc,
    compute_faithfulness_metrics,
    compute_pointing_game,
    compute_mean_iou,
)
from attention_analysis.training_dynamics import (
    AttentionEntropyTracker,
    HeadSpecializationTracker,
    AttentionAnalysisCallback,
)

__all__ = [
    # Utils
    "HookManager",
    # Methods
    "get_attribution",
    "list_methods",
    "AttnLRP",
    "GradCAM",
    "AttentionRollout",
    "GMAR",
    "RawAttention",
    # Evaluation
    "compute_insertion_auc",
    "compute_deletion_auc",
    "compute_faithfulness_metrics",
    "compute_pointing_game",
    "compute_mean_iou",
    # Training dynamics
    "AttentionEntropyTracker",
    "HeadSpecializationTracker",
    "AttentionAnalysisCallback",
]
