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
    generate_attention_entropy_per_head_heatmap,
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
    generate_embedding_pca_scatter,
    # New: Attention tracking
    AttentionTracker,
    # New: Collapse monitor
    generate_collapse_monitor,
    # New: Training dashboard
    generate_training_dashboard,
    generate_embedding_spectrum,
    generate_xy_curves,
    generate_pocp_per_head_heatmaps,
    # SIGReg-aligned visualizations
    generate_isotropy_evolution_plot,
    generate_loss_accuracy_correlation_plot,
    generate_embedding_distribution_plot,
    generate_sigreg_loss_components_plot,
    # Loss landscape visualizations
    generate_loss_landscape_contour,
    generate_loss_landscape_3d,
    generate_loss_landscape_3d_with_contour,
    generate_loss_landscape_plotly,
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
    "generate_attention_entropy_per_head_heatmap",
    # New Visualizations - Training Dynamics
    "generate_layer_attention_evolution",
    "generate_per_head_attention",
    "generate_gradient_weighted_attention",
    "generate_head_importance_heatmap",
    "generate_token_similarity_heatmap",
    "generate_rsm_across_layers",
    "generate_gradient_flow_heatmap",
    "generate_embedding_projection",
    "generate_embedding_pca_scatter",
    "generate_collapse_monitor",
    "generate_training_dashboard",
    "generate_embedding_spectrum",
    "generate_xy_curves",
    "generate_pocp_per_head_heatmaps",
    "AttentionTracker",
    # SIGReg-aligned visualizations
    "generate_isotropy_evolution_plot",
    "generate_loss_accuracy_correlation_plot",
    "generate_embedding_distribution_plot",
    "generate_sigreg_loss_components_plot",
    # Loss landscape visualizations
    "generate_loss_landscape_contour",
    "generate_loss_landscape_3d",
    "generate_loss_landscape_3d_with_contour",
    "generate_loss_landscape_plotly",
]
