"""Utilities for LeJEPA-JIT training and visualization."""

from utils.metrics import (
    GSNRTracker,
    compute_attention_rank,
    compute_batch_gsnr,
    compute_linear_cka,
    compute_alignment_metrics,
    compute_covariance_metrics,
    compute_global_norms,
    compute_entropy,
    compute_feature_collapse_metrics,
    compute_gini,
    compute_gradient_flow_stats,
    compute_head_diversity,
    compute_layer_gradient_stats,
    compute_representation_stats,
    compute_sparsity,
)
from utils.visualization import (
    # Original visualizations
    generate_attention_grid,
    generate_attention_rollout,
    generate_pca_visualization,
    # New: Layer-wise attention
    generate_layer_attention_evolution,
    # New: Per-head attention
    generate_per_head_attention,
    # New: Gradient-weighted attention (GMAR-style)
    generate_gradient_weighted_attention,
    # New: Head importance heatmap
    generate_head_importance_heatmap,
    # New: Token similarity
    generate_token_similarity_heatmap,
    # New: RSM across layers
    generate_rsm_across_layers,
    # New: Gradient flow
    generate_gradient_flow_heatmap,
    # New: Embedding projection (t-SNE/UMAP)
    generate_embedding_projection,
    # New: Attention tracking
    AttentionTracker,
    # New: Collapse monitor
    generate_collapse_monitor,
    # New: Training dashboard
    generate_training_dashboard,
    generate_embedding_spectrum,
)

__all__ = [
    # Metrics
    "compute_entropy",
    "compute_gini",
    "compute_sparsity",
    "compute_gradient_flow_stats",
    "compute_layer_gradient_stats",
    "compute_batch_gsnr",
    "compute_head_diversity",
    "compute_attention_rank",
    "compute_representation_stats",
    "compute_feature_collapse_metrics",
    "compute_alignment_metrics",
    "compute_covariance_metrics",
    "compute_global_norms",
    "compute_linear_cka",
    "GSNRTracker",
    # Original Visualization
    "generate_pca_visualization",
    "generate_attention_rollout",
    "generate_attention_grid",
    # New Visualizations - Training Dynamics
    "generate_layer_attention_evolution",
    "generate_per_head_attention",
    "generate_gradient_weighted_attention",
    "generate_head_importance_heatmap",
    "generate_token_similarity_heatmap",
    "generate_rsm_across_layers",
    "generate_gradient_flow_heatmap",
    "generate_embedding_projection",
    "generate_collapse_monitor",
    "generate_training_dashboard",
    "generate_embedding_spectrum",
    "AttentionTracker",
]
