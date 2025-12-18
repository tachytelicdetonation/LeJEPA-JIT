"""Multi-panel summary dashboards for WandB logging.

Provides functions for generating comprehensive, at-a-glance visualizations
that summarize training progress, health status, and key metrics.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _fig_to_pil(fig: plt.Figure) -> Image.Image:
    """Convert matplotlib figure to PIL Image."""
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return Image.fromarray(img_array[:, :, :3])  # Drop alpha channel


def generate_epoch_summary_dashboard(
    epoch: int,
    metrics: dict[str, float],
    history: dict[str, list[float]] | None = None,
    figsize: tuple[int, int] = (16, 10),
) -> Image.Image:
    """
    Generate a 6-panel summary dashboard for an epoch.

    Panels:
    1. Validation accuracy trend (ImageNette + ImageWoof)
    2. Loss components (total, sigreg, prediction)
    3. KNN accuracy trend (representation quality)
    4. Attention entropy evolution (sparsity indicator)
    5. Representation health (effective dim, isotropy)
    6. Head importance summary

    Args:
        epoch: Current epoch number
        metrics: Current epoch metrics
        history: Historical metrics {metric_name: [values]}
        figsize: Figure size

    Returns:
        PIL Image of the dashboard
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(f"Training Summary - Epoch {epoch}", fontsize=16, fontweight="bold")

    history = history or {}

    # Panel 1: Validation Accuracy
    ax = axes[0, 0]
    _plot_accuracy_panel(ax, metrics, history)

    # Panel 2: Loss Components
    ax = axes[0, 1]
    _plot_loss_panel(ax, metrics, history)

    # Panel 3: KNN Accuracy
    ax = axes[0, 2]
    _plot_knn_panel(ax, metrics, history)

    # Panel 4: Attention Entropy
    ax = axes[1, 0]
    _plot_entropy_panel(ax, metrics, history)

    # Panel 5: Representation Health
    ax = axes[1, 1]
    _plot_rep_health_panel(ax, metrics)

    # Panel 6: Head Importance Summary
    ax = axes[1, 2]
    _plot_head_summary_panel(ax, metrics)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return _fig_to_pil(fig)


def _plot_accuracy_panel(
    ax: plt.Axes,
    metrics: dict[str, float],
    history: dict[str, list[float]],
) -> None:
    """Plot validation accuracy trend."""
    ax.set_title("Validation Accuracy", fontsize=12, fontweight="bold")

    # Plot history if available
    nette_hist = history.get("val_accuracy_nette", history.get("summary/val_nette", []))
    woof_hist = history.get("val_accuracy_woof", history.get("summary/val_woof", []))

    if nette_hist:
        ax.plot(nette_hist, label="ImageNette", color="#22c55e", linewidth=2)
    if woof_hist:
        ax.plot(woof_hist, label="ImageWoof", color="#3b82f6", linewidth=2)

    # Current values
    nette_val = metrics.get("val_accuracy_nette", metrics.get("summary/val_nette", 0))
    woof_val = metrics.get("val_accuracy_woof", metrics.get("summary/val_woof", 0))

    if nette_hist:
        ax.scatter([len(nette_hist) - 1], [nette_val], color="#22c55e", s=100, zorder=5)
    if woof_hist:
        ax.scatter([len(woof_hist) - 1], [woof_val], color="#3b82f6", s=100, zorder=5)

    # Add current values as text
    ax.text(
        0.02,
        0.98,
        f"Nette: {nette_val:.1%}\nWoof: {woof_val:.1%}",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)


def _plot_loss_panel(
    ax: plt.Axes,
    metrics: dict[str, float],
    history: dict[str, list[float]],
) -> None:
    """Plot loss components."""
    ax.set_title("Loss Components", fontsize=12, fontweight="bold")

    loss_hist = history.get("train/loss", history.get("health/loss", []))
    sigreg_hist = history.get("train/sigreg", history.get("health/sigreg", []))
    pred_hist = history.get("train/prediction", history.get("health/prediction", []))

    if loss_hist:
        ax.plot(loss_hist, label="Total", color="#ef4444", linewidth=2)
    if sigreg_hist:
        ax.plot(
            sigreg_hist, label="SigReg", color="#f59e0b", linewidth=1.5, linestyle="--"
        )
    if pred_hist:
        ax.plot(
            pred_hist, label="Prediction", color="#8b5cf6", linewidth=1.5, linestyle=":"
        )

    # Current values
    loss_val = metrics.get("train/loss", metrics.get("health/loss", 0))
    sigreg_val = metrics.get("train/sigreg", metrics.get("health/sigreg", 0))

    ax.text(
        0.02,
        0.98,
        f"Loss: {loss_val:.4f}\nSigReg: {sigreg_val:.4f}",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)


def _plot_knn_panel(
    ax: plt.Axes,
    metrics: dict[str, float],
    history: dict[str, list[float]],
) -> None:
    """Plot KNN accuracy trend."""
    ax.set_title("KNN Accuracy (Rep Quality)", fontsize=12, fontweight="bold")

    knn_nette = history.get("knn/nette_top1", history.get("summary/knn_nette", []))
    knn_woof = history.get("knn/woof_top1", history.get("summary/knn_woof", []))

    if knn_nette:
        ax.plot(knn_nette, label="Nette KNN", color="#22c55e", linewidth=2)
    if knn_woof:
        ax.plot(knn_woof, label="Woof KNN", color="#3b82f6", linewidth=2)

    # Current values
    knn_nette_val = metrics.get("knn/nette_top1", metrics.get("summary/knn_nette", 0))
    knn_woof_val = metrics.get("knn/woof_top1", metrics.get("summary/knn_woof", 0))

    ax.text(
        0.02,
        0.98,
        f"Nette: {knn_nette_val:.1%}\nWoof: {knn_woof_val:.1%}",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("KNN Accuracy")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)


def _plot_entropy_panel(
    ax: plt.Axes,
    metrics: dict[str, float],
    history: dict[str, list[float]],
) -> None:
    """Plot attention entropy evolution."""
    ax.set_title("Attention Entropy", fontsize=12, fontweight="bold")

    entropy_hist = history.get("attn/entropy", history.get("epoch_attn/entropy", []))

    if entropy_hist:
        ax.plot(entropy_hist, label="Mean Entropy", color="#8b5cf6", linewidth=2)

    # Current value
    entropy_val = metrics.get("attn/entropy", metrics.get("epoch_attn/entropy", 0))

    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Collapse Risk")
    ax.axhline(y=4.0, color="orange", linestyle="--", alpha=0.5, label="Diffuse")

    ax.text(
        0.02,
        0.98,
        f"Entropy: {entropy_val:.2f}",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Entropy")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_rep_health_panel(ax: plt.Axes, metrics: dict[str, float]) -> None:
    """Plot representation health metrics as bar chart."""
    ax.set_title("Representation Health", fontsize=12, fontweight="bold")

    # Get metrics
    eff_dim = metrics.get("rep/effective_dim", metrics.get("health/effective_dim", 50))
    isotropy = metrics.get("rep/isotropy", 0.5)
    variance = metrics.get("rep/variance", 0.5)

    # Normalize for visualization
    eff_dim_norm = min(eff_dim / 100, 1.0)  # Normalize to 100
    iso_norm = isotropy
    var_norm = min(variance, 1.0)

    ax.bar(
        ["Eff. Dim", "Isotropy", "Variance"],
        [eff_dim_norm, iso_norm, var_norm],
        color=["#22c55e", "#3b82f6", "#f59e0b"],
        alpha=0.7,
    )

    # Add value labels
    ax.text(0, eff_dim_norm + 0.05, f"{eff_dim:.0f}", ha="center", fontsize=10)
    ax.text(1, iso_norm + 0.05, f"{isotropy:.2f}", ha="center", fontsize=10)
    ax.text(2, var_norm + 0.05, f"{variance:.2f}", ha="center", fontsize=10)

    # Add threshold lines
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.3, label="Warning")

    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Normalized Score")
    ax.grid(True, alpha=0.3, axis="y")


def _plot_head_summary_panel(ax: plt.Axes, metrics: dict[str, float]) -> None:
    """Plot head importance/redundancy summary."""
    ax.set_title("Head Analysis", fontsize=12, fontweight="bold")

    # Get pattern distribution if available
    patterns = ["parallel", "radioactive", "homogeneous", "xtype", "compound"]
    values = [metrics.get(f"attn/pattern_{p}", 0) for p in patterns]

    if sum(values) > 0:
        # Pie chart of pattern distribution
        colors = ["#22c55e", "#ef4444", "#3b82f6", "#f59e0b", "#8b5cf6"]
        non_zero = [(p, v, c) for p, v, c in zip(patterns, values, colors) if v > 0]

        if non_zero:
            labels, sizes, pie_colors = zip(*non_zero)
            ax.pie(
                sizes,
                labels=labels,
                colors=pie_colors,
                autopct="%1.0f%%",
                startangle=90,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No pattern data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
    else:
        # Show redundancy score if available
        redundancy = metrics.get("arch/head_redundancy", 0)
        diversity = metrics.get("arch/diversity_score", 0)

        ax.bar(
            ["Redundancy", "Diversity"],
            [redundancy, diversity],
            color=["#ef4444", "#22c55e"],
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")

        if redundancy > 0.5:
            ax.text(
                0.5,
                0.9,
                "High redundancy!",
                ha="center",
                transform=ax.transAxes,
                color="red",
            )


def generate_epoch_delta_dashboard(
    current_epoch: int,
    current_metrics: dict[str, float],
    previous_metrics: dict[str, float],
    figsize: tuple[int, int] = (12, 6),
) -> Image.Image:
    """
    Generate epoch-over-epoch delta view showing what changed.

    Args:
        current_epoch: Current epoch number
        current_metrics: Current epoch metrics
        previous_metrics: Previous epoch metrics
        figsize: Figure size

    Returns:
        PIL Image showing deltas with arrows
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(f"Epoch {current_epoch} Changes", fontsize=14, fontweight="bold")

    # Key metrics to compare
    key_metrics = [
        ("Val Accuracy (Nette)", "val_accuracy_nette", "summary/val_nette", True),
        ("Val Accuracy (Woof)", "val_accuracy_woof", "summary/val_woof", True),
        ("KNN Accuracy", "knn/nette_top1", "summary/knn_nette", True),
        ("Loss", "train/loss", "health/loss", False),
        ("SigReg", "train/sigreg", "health/sigreg", False),
        ("Entropy", "epoch_attn/entropy", "attn/entropy", None),
        ("Effective Dim", "rep/effective_dim", "health/effective_dim", True),
    ]

    y_positions = list(range(len(key_metrics)))[::-1]
    bar_height = 0.6

    for i, (name, key1, key2, higher_better) in enumerate(key_metrics):
        current = current_metrics.get(key1, current_metrics.get(key2, 0))
        previous = previous_metrics.get(key1, previous_metrics.get(key2, 0))
        delta = current - previous

        # Determine color based on improvement
        if higher_better is None:
            color = "#6b7280"  # Gray for neutral
        elif (delta > 0 and higher_better) or (delta < 0 and not higher_better):
            color = "#22c55e"  # Green for improvement
        elif delta == 0:
            color = "#6b7280"  # Gray for no change
        else:
            color = "#ef4444"  # Red for regression

        # Draw bar for delta
        ax.barh(y_positions[i], delta, height=bar_height, color=color, alpha=0.7)

        # Add labels
        arrow = "" if delta == 0 else "" if delta > 0 else ""
        delta_str = f"{delta:+.4f}" if abs(delta) < 1 else f"{delta:+.2f}"
        ax.text(
            0.02 if delta >= 0 else -0.02,
            y_positions[i],
            f"{name}: {current:.4f} ({arrow}{delta_str})",
            va="center",
            ha="left" if delta >= 0 else "right",
            fontsize=10,
            fontfamily="monospace",
        )

    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_yticks([])
    ax.set_xlabel("Change from Previous Epoch")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    return _fig_to_pil(fig)


def generate_training_radar(
    current_metrics: dict[str, float],
    target_metrics: dict[str, float] | None = None,
    figsize: tuple[int, int] = (8, 8),
) -> Image.Image:
    """
    Generate radar chart showing training progress across 6 axes.

    Args:
        current_metrics: Current metric values
        target_metrics: Optional target values for comparison
        figsize: Figure size

    Returns:
        PIL Image of radar chart
    """
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    # Define axes
    categories = [
        "Val Accuracy",
        "KNN Accuracy",
        "Eff. Dimension",
        "Attn Diversity",
        "Gradient Health",
        "Head Specialization",
    ]
    N = len(categories)

    # Get current values (normalized to 0-1)
    val_acc = current_metrics.get(
        "val_accuracy_nette", current_metrics.get("summary/val_nette", 0)
    )
    knn_acc = current_metrics.get(
        "knn/nette_top1", current_metrics.get("summary/knn_nette", 0)
    )
    eff_dim = min(current_metrics.get("rep/effective_dim", 50) / 100, 1.0)
    attn_div = current_metrics.get("arch/diversity_score", 0.5)
    grad_health = 1.0 - min(current_metrics.get("health/grad_norm", 1.0) / 10.0, 1.0)
    head_spec = 1.0 - current_metrics.get("arch/head_redundancy", 0.5)

    values = [val_acc, knn_acc, eff_dim, attn_div, grad_health, head_spec]

    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values_plot = values + values[:1]  # Close the loop
    angles += angles[:1]

    # Plot current values
    ax.plot(angles, values_plot, "o-", linewidth=2, label="Current", color="#3b82f6")
    ax.fill(angles, values_plot, alpha=0.25, color="#3b82f6")

    # Plot target if provided
    if target_metrics:
        target_values = [
            target_metrics.get("val_accuracy", 0.9),
            target_metrics.get("knn_accuracy", 0.85),
            target_metrics.get("eff_dimension", 0.8),
            target_metrics.get("attn_diversity", 0.7),
            target_metrics.get("grad_health", 0.8),
            target_metrics.get("head_specialization", 0.7),
        ]
        target_plot = target_values + target_values[:1]
        ax.plot(angles, target_plot, "--", linewidth=2, label="Target", color="#22c55e")

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)

    # Set radial limits
    ax.set_ylim(0, 1)

    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.0))
    ax.set_title("Training Progress Radar", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    return _fig_to_pil(fig)


def generate_attention_focus_summary(
    images: "np.ndarray",  # (N, H, W, 3)
    attention_scores: "np.ndarray",  # (N,) score per image
    num_best: int = 4,
    num_worst: int = 4,
    figsize: tuple[int, int] = (16, 6),
) -> Image.Image:
    """
    Generate summary showing best and worst attended images.

    Args:
        images: Array of images
        attention_scores: Attention quality score per image
        num_best: Number of best images to show
        num_worst: Number of worst images to show
        figsize: Figure size

    Returns:
        PIL Image showing best/worst attended images
    """
    n_images = len(images)
    num_best = min(num_best, n_images // 2)
    num_worst = min(num_worst, n_images // 2)

    # Sort by attention score
    sorted_indices = np.argsort(attention_scores)
    best_indices = sorted_indices[-num_best:][::-1]
    worst_indices = sorted_indices[:num_worst]

    fig, axes = plt.subplots(2, max(num_best, num_worst), figsize=figsize)
    fig.suptitle("Attention Focus: Best vs Worst", fontsize=14, fontweight="bold")

    # Plot best
    for i, idx in enumerate(best_indices):
        ax = axes[0, i] if num_best > 1 else axes[0]
        ax.imshow(images[idx])
        ax.set_title(f"Best {i + 1}\n(score: {attention_scores[idx]:.3f})", fontsize=10)
        ax.axis("off")

    # Plot worst
    for i, idx in enumerate(worst_indices):
        ax = axes[1, i] if num_worst > 1 else axes[1]
        ax.imshow(images[idx])
        ax.set_title(
            f"Worst {i + 1}\n(score: {attention_scores[idx]:.3f})", fontsize=10
        )
        ax.axis("off")

    # Hide unused axes
    for i in range(max(num_best, num_worst)):
        if i >= num_best:
            axes[0, i].axis("off")
        if i >= num_worst:
            axes[1, i].axis("off")

    plt.tight_layout()
    return _fig_to_pil(fig)


def generate_pattern_distribution_chart(
    pattern_counts: dict[str, int],
    figsize: tuple[int, int] = (8, 6),
) -> Image.Image:
    """
    Generate pie chart of attention pattern distribution.

    Args:
        pattern_counts: Count per pattern type
        figsize: Figure size

    Returns:
        PIL Image of pie chart
    """
    fig, ax = plt.subplots(figsize=figsize)

    patterns = list(pattern_counts.keys())
    counts = list(pattern_counts.values())

    colors = {
        "parallel": "#22c55e",
        "radioactive": "#ef4444",
        "homogeneous": "#3b82f6",
        "xtype": "#f59e0b",
        "compound": "#8b5cf6",
    }

    pie_colors = [colors.get(p, "#6b7280") for p in patterns]

    ax.pie(
        counts,
        labels=patterns,
        colors=pie_colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.85,
    )

    ax.set_title("Attention Pattern Distribution", fontsize=14, fontweight="bold")

    plt.tight_layout()
    return _fig_to_pil(fig)


def generate_layer_importance_curve(
    layer_importance: list[float],
    figsize: tuple[int, int] = (10, 5),
) -> Image.Image:
    """
    Generate curve showing layer importance/contribution.

    Args:
        layer_importance: Importance score per layer
        figsize: Figure size

    Returns:
        PIL Image of curve
    """
    fig, ax = plt.subplots(figsize=figsize)

    layers = list(range(len(layer_importance)))

    ax.bar(layers, layer_importance, color="#3b82f6", alpha=0.7)
    ax.plot(layers, layer_importance, "o-", color="#1e40af", linewidth=2)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Importance")
    ax.set_title("Layer Importance Distribution", fontsize=14, fontweight="bold")
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3, axis="y")

    # Highlight most important layers
    max_idx = np.argmax(layer_importance)
    ax.annotate(
        f"Max: L{max_idx}",
        xy=(max_idx, layer_importance[max_idx]),
        xytext=(max_idx + 0.5, layer_importance[max_idx] + 0.02),
        fontsize=10,
        color="red",
    )

    plt.tight_layout()
    return _fig_to_pil(fig)
