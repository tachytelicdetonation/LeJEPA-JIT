"""
Training Dynamics Analysis for Attention.

Tools for tracking attention patterns during training to understand
how attention evolves and heads specialize.

Components:
- AttentionEntropyTracker: Track attention entropy over training
- HeadSpecializationTracker: Analyze head roles and specialization
- AttentionAnalysisCallback: Training callback for automated logging
"""

from attention_analysis.training_dynamics.entropy_tracking import (
    AttentionEntropyTracker,
    compute_attention_entropy,
    compute_layer_entropy_stats,
)
from attention_analysis.training_dynamics.head_specialization import (
    HeadSpecializationTracker,
    compute_head_importance,
    compute_head_similarity,
)
from attention_analysis.training_dynamics.callbacks import (
    AttentionAnalysisCallback,
)

__all__ = [
    "AttentionEntropyTracker",
    "compute_attention_entropy",
    "compute_layer_entropy_stats",
    "HeadSpecializationTracker",
    "compute_head_importance",
    "compute_head_similarity",
    "AttentionAnalysisCallback",
]
