"""
Comparison visualization tools.

Functions for comparing different attribution methods and plotting
evaluation metrics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

import matplotlib.pyplot as plt


def compare_methods_visual(
    image: Image.Image | torch.Tensor,
    method_attributions: dict[str, torch.Tensor],
    cmap: str = "jet",
    figsize: tuple[int, int] | None = None,
) -> Image.Image:
    """
    Create side-by-side comparison of multiple attribution methods.

    Args:
        image: Original image.
        method_attributions: Dict mapping method names to attribution maps.
        cmap: Colormap for heatmaps.
        figsize: Figure size.

    Returns:
        Combined comparison image.
    """
    from attention_analysis.visualization.heatmaps import overlay_attribution

    n_methods = len(method_attributions)
    cols = n_methods + 1  # Original + methods

    if figsize is None:
        figsize = (cols * 3, 3)

    fig, axes = plt.subplots(1, cols, figsize=figsize)

    # Original image
    if isinstance(image, torch.Tensor):
        if image.shape[0] in (1, 3):
            image = image.permute(1, 2, 0)
        image = image.detach().cpu().numpy()
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Method overlays
    for i, (method_name, attribution) in enumerate(method_attributions.items()):
        overlay = overlay_attribution(image, attribution, cmap=cmap)
        axes[i + 1].imshow(overlay)
        axes[i + 1].set_title(method_name)
        axes[i + 1].axis("off")

    plt.tight_layout()

    # Convert to PIL
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return Image.fromarray(img_array)


def plot_faithfulness_curves(
    method_results: dict[str, Any],
    figsize: tuple[int, int] = (12, 5),
    title: str = "Faithfulness Comparison",
) -> Image.Image:
    """
    Plot insertion and deletion curves for multiple methods.

    Args:
        method_results: Dict mapping method names to FaithfulnessResult or
            dict with 'insertion_curve' and 'deletion_curve' keys.
        figsize: Figure size.
        title: Plot title.

    Returns:
        Plot as PIL Image.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    colors = plt.cm.tab10.colors

    # Insertion curves
    for i, (method_name, result) in enumerate(method_results.items()):
        if hasattr(result, "insertion_curve"):
            curve = result.insertion_curve
            auc = result.insertion_auc
        else:
            curve = result.get("insertion_curve", [])
            auc = result.get("insertion_auc", 0)

        if curve:
            x = np.linspace(0, 1, len(curve))
            axes[0].plot(
                x,
                curve,
                color=colors[i % len(colors)],
                label=f"{method_name} (AUC={auc:.3f})",
            )

    axes[0].set_xlabel("Fraction of pixels revealed")
    axes[0].set_ylabel("Confidence")
    axes[0].set_title("Insertion (Higher = Better)")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # Deletion curves
    for i, (method_name, result) in enumerate(method_results.items()):
        if hasattr(result, "deletion_curve"):
            curve = result.deletion_curve
            auc = result.deletion_auc
        else:
            curve = result.get("deletion_curve", [])
            auc = result.get("deletion_auc", 0)

        if curve:
            x = np.linspace(0, 1, len(curve))
            axes[1].plot(
                x,
                curve,
                color=colors[i % len(colors)],
                label=f"{method_name} (AUC={auc:.3f})",
            )

    axes[1].set_xlabel("Fraction of pixels removed")
    axes[1].set_ylabel("Confidence")
    axes[1].set_title("Deletion (Lower = Better)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    # Convert to PIL
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return Image.fromarray(img_array)


def plot_faithfulness_comparison_bar(
    method_results: dict[str, Any],
    figsize: tuple[int, int] = (10, 5),
    title: str = "Faithfulness Metrics Comparison",
) -> Image.Image:
    """
    Create bar chart comparing faithfulness metrics.

    Args:
        method_results: Dict mapping method names to results with
            'mean_insertion_auc', 'mean_deletion_auc' keys.
        figsize: Figure size.
        title: Plot title.

    Returns:
        Bar chart as PIL Image.
    """
    methods = list(method_results.keys())
    insertion_aucs = []
    deletion_aucs = []
    combined_scores = []

    for method in methods:
        result = method_results[method]
        if hasattr(result, "insertion_auc"):
            insertion_aucs.append(result.insertion_auc)
            deletion_aucs.append(result.deletion_auc)
            combined_scores.append(result.combined_score)
        else:
            insertion_aucs.append(result.get("mean_insertion_auc", 0))
            deletion_aucs.append(result.get("mean_deletion_auc", 0))
            combined_scores.append(
                result.get("mean_combined", insertion_aucs[-1] - deletion_aucs[-1])
            )

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(
        x - width,
        insertion_aucs,
        width,
        label="Insertion AUC (↑)",
        color="green",
        alpha=0.7,
    )
    ax.bar(x, deletion_aucs, width, label="Deletion AUC (↓)", color="red", alpha=0.7)
    ax.bar(
        x + width, combined_scores, width, label="Combined (↑)", color="blue", alpha=0.7
    )

    ax.set_xlabel("Method")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return Image.fromarray(img_array)


def plot_entropy_evolution(
    entropy_history: list[tuple[int, float]] | dict[str, list[tuple[int, float]]],
    figsize: tuple[int, int] = (10, 5),
    title: str = "Attention Entropy Evolution",
) -> Image.Image:
    """
    Plot entropy evolution during training.

    Args:
        entropy_history: Either:
            - List of (step, entropy) tuples
            - Dict mapping layer/head names to such lists
        figsize: Figure size.
        title: Plot title.

    Returns:
        Plot as PIL Image.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if isinstance(entropy_history, dict):
        for name, history in entropy_history.items():
            if history:
                steps, entropies = zip(*history)
                ax.plot(steps, entropies, label=name, alpha=0.7)
        ax.legend()
    else:
        if entropy_history:
            steps, entropies = zip(*entropy_history)
            ax.plot(steps, entropies, color="blue")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Attention Entropy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return Image.fromarray(img_array)


def plot_head_importance(
    head_importance: list,  # List of HeadImportance
    figsize: tuple[int, int] | None = None,
    title: str = "Head Importance",
) -> Image.Image:
    """
    Create heatmap of head importance across layers.

    Args:
        head_importance: List of HeadImportance objects.
        figsize: Figure size.
        title: Plot title.

    Returns:
        Heatmap as PIL Image.
    """
    if not head_importance:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
    else:
        # Organize into grid
        layers = set(h.layer_idx for h in head_importance)
        heads = set(h.head_idx for h in head_importance)
        n_layers = max(layers) + 1
        n_heads = max(heads) + 1

        importance_grid = np.zeros((n_layers, n_heads))
        for h in head_importance:
            importance_grid[h.layer_idx, h.head_idx] = h.importance_score

        if figsize is None:
            figsize = (max(6, n_heads * 0.8), max(4, n_layers * 0.5))

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(importance_grid, cmap="YlOrRd", aspect="auto")
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(title)
        ax.set_xticks(range(n_heads))
        ax.set_yticks(range(n_layers))

        plt.colorbar(im, ax=ax, label="Importance")

    plt.tight_layout()

    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return Image.fromarray(img_array)


def plot_head_similarity(
    similarity_matrix: torch.Tensor | np.ndarray,
    figsize: tuple[int, int] = (8, 8),
    title: str = "Head Similarity Matrix",
) -> Image.Image:
    """
    Plot head similarity matrix.

    Args:
        similarity_matrix: Cosine similarity matrix between heads.
        figsize: Figure size.
        title: Plot title.

    Returns:
        Heatmap as PIL Image.
    """
    if isinstance(similarity_matrix, torch.Tensor):
        similarity_matrix = similarity_matrix.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(similarity_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Head Index")
    ax.set_title(title)

    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    plt.tight_layout()

    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return Image.fromarray(img_array)
