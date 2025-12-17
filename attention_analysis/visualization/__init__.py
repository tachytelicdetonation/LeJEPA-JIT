"""
Visualization tools for attention analysis.

Provides functions for generating heatmaps, overlays, and comparison plots.
"""

from attention_analysis.visualization.heatmaps import (
    attribution_to_heatmap,
    overlay_attribution,
    create_side_by_side,
    generate_attribution_grid,
)
from attention_analysis.visualization.comparison import (
    compare_methods_visual,
    plot_faithfulness_curves,
    plot_entropy_evolution,
)

__all__ = [
    # Heatmaps
    "attribution_to_heatmap",
    "overlay_attribution",
    "create_side_by_side",
    "generate_attribution_grid",
    # Comparison
    "compare_methods_visual",
    "plot_faithfulness_curves",
    "plot_entropy_evolution",
]
