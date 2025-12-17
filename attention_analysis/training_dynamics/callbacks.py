"""
Training Callbacks for Attention Analysis.

Provides callbacks that can be integrated into training loops to
automatically track and log attention patterns.
"""

from __future__ import annotations

from typing import Callable
from pathlib import Path

import torch
import torch.nn as nn

from attention_analysis.training_dynamics.entropy_tracking import (
    AttentionEntropyTracker,
)
from attention_analysis.training_dynamics.head_specialization import (
    HeadSpecializationTracker,
)
from attention_analysis.methods import get_attribution


class AttentionAnalysisCallback:
    """
    Training callback for attention analysis.

    Integrates with training loops to automatically compute and log
    attention metrics at specified intervals.

    Args:
        model: The model to analyze.
        probe_images: Fixed set of images for consistent measurements.
        probe_labels: Labels for probe images (optional).
        log_fn: Function to call with logged metrics. Signature:
            log_fn(metrics: dict, step: int, epoch: int)
        save_dir: Directory to save analysis artifacts.
        entropy_interval: Steps between entropy computations.
        importance_interval: Epochs between head importance computations.
        attribution_interval: Epochs between attribution visualizations.
        attribution_methods: List of attribution methods to compute.
    """

    def __init__(
        self,
        model: nn.Module,
        probe_images: torch.Tensor,
        probe_labels: torch.Tensor | None = None,
        log_fn: Callable[[dict, int, int], None] | None = None,
        save_dir: str | Path | None = None,
        entropy_interval: int = 100,
        importance_interval: int = 5,
        attribution_interval: int = 10,
        attribution_methods: list[str] | None = None,
    ):
        self.model = model
        self.probe_images = probe_images
        self.probe_labels = probe_labels
        self.log_fn = log_fn
        self.save_dir = Path(save_dir) if save_dir else None
        self.entropy_interval = entropy_interval
        self.importance_interval = importance_interval
        self.attribution_interval = attribution_interval
        self.attribution_methods = attribution_methods or ["rollout", "raw_attention"]

        # Create trackers
        self.entropy_tracker = AttentionEntropyTracker(
            model, log_interval=entropy_interval, probe_images=probe_images
        )
        self.specialization_tracker = HeadSpecializationTracker(
            model, probe_images, probe_labels
        )

        # Storage
        self._step = 0
        self._epoch = 0
        self.attribution_history: dict[str, list] = {
            m: [] for m in self.attribution_methods
        }

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_batch_end(
        self,
        batch_idx: int,
        loss: float | None = None,
        **kwargs,
    ) -> dict | None:
        """
        Called at the end of each training batch.

        Args:
            batch_idx: Current batch index.
            loss: Optional batch loss.
            **kwargs: Additional data to log.

        Returns:
            Metrics dict if logged, None otherwise.
        """
        self._step += 1

        # Track entropy
        snapshot = self.entropy_tracker.step(epoch=self._epoch)

        if snapshot is not None:
            metrics = {
                "attention/mean_entropy": snapshot.mean_entropy,
                "attention/std_entropy": snapshot.std_entropy,
            }

            # Add per-layer entropy
            for i, ent in enumerate(snapshot.layer_entropies):
                metrics[f"attention/layer_{i}_entropy"] = ent

            if self.log_fn:
                self.log_fn(metrics, self._step, self._epoch)

            return metrics

        return None

    def on_epoch_end(
        self,
        epoch: int,
        **kwargs,
    ) -> dict:
        """
        Called at the end of each training epoch.

        Args:
            epoch: Current epoch number.
            **kwargs: Additional data to log.

        Returns:
            Metrics dict with all computed metrics.
        """
        self._epoch = epoch
        metrics = {}

        # Compute head importance
        if epoch % self.importance_interval == 0:
            importance = self.specialization_tracker.compute_head_importance(
                method="activation"
            )

            # Aggregate importance by layer
            layer_importance = {}
            for imp in importance:
                key = f"attention/layer_{imp.layer_idx}_importance"
                if key not in layer_importance:
                    layer_importance[key] = []
                layer_importance[key].append(imp.importance_score)

            for key, scores in layer_importance.items():
                metrics[key] = sum(scores) / len(scores) if scores else 0.0

            # Head similarity
            similarity = self.specialization_tracker.compute_head_similarity()
            # Log mean off-diagonal similarity (redundancy measure)
            if similarity.numel() > 1:
                n = similarity.shape[0]
                off_diag = similarity.sum() - similarity.trace()
                off_diag_mean = off_diag / (n * n - n) if n > 1 else 0.0
                metrics["attention/head_redundancy"] = off_diag_mean.item()

        # Compute attributions
        if epoch % self.attribution_interval == 0:
            for method_name in self.attribution_methods:
                try:
                    # Compute attribution for first probe image
                    attr = get_attribution(
                        self.model,
                        self.probe_images[0],
                        method=method_name,
                        target_class=self.probe_labels[0].item()
                        if self.probe_labels is not None
                        else None,
                    )

                    # Store for later analysis
                    self.attribution_history[method_name].append((epoch, attr.cpu()))

                    # Log attribution statistics
                    metrics[f"attribution/{method_name}_mean"] = attr.mean().item()
                    metrics[f"attribution/{method_name}_max"] = attr.max().item()
                    metrics[f"attribution/{method_name}_entropy"] = (
                        self._compute_attr_entropy(attr)
                    )

                except Exception as e:
                    print(f"Warning: Failed to compute {method_name}: {e}")

        # Save artifacts
        if self.save_dir and epoch % self.attribution_interval == 0:
            self._save_epoch_artifacts(epoch)

        # Log all metrics
        if self.log_fn and metrics:
            self.log_fn(metrics, self._step, epoch)

        return metrics

    def _compute_attr_entropy(self, attribution: torch.Tensor) -> float:
        """Compute entropy of attribution distribution."""
        # Normalize to probability distribution
        attr = attribution.flatten()
        attr = attr - attr.min()
        if attr.sum() > 0:
            attr = attr / attr.sum()
            entropy = -(attr * torch.log(attr + 1e-10)).sum()
            return entropy.item()
        return 0.0

    def _save_epoch_artifacts(self, epoch: int) -> None:
        """Save analysis artifacts for an epoch."""
        if self.save_dir is None:
            return

        epoch_dir = self.save_dir / f"epoch_{epoch:04d}"
        epoch_dir.mkdir(exist_ok=True)

        # Save entropy history
        entropy_data = {
            "steps": [s.step for s in self.entropy_tracker.history],
            "mean_entropy": [s.mean_entropy for s in self.entropy_tracker.history],
            "layer_entropies": [
                s.layer_entropies for s in self.entropy_tracker.history
            ],
        }
        torch.save(entropy_data, epoch_dir / "entropy_history.pt")

        # Save latest attributions
        for method_name, history in self.attribution_history.items():
            if history:
                latest_epoch, latest_attr = history[-1]
                torch.save(latest_attr, epoch_dir / f"attribution_{method_name}.pt")

    def get_entropy_summary(self) -> dict:
        """Get summary of entropy evolution."""
        return self.entropy_tracker.get_summary()

    def get_attribution_evolution(self, method: str) -> list[tuple[int, torch.Tensor]]:
        """Get attribution history for a method."""
        return self.attribution_history.get(method, [])


class WandBAttentionCallback(AttentionAnalysisCallback):
    """
    Attention analysis callback with WandB integration.

    Automatically logs metrics and visualizations to Weights & Biases.
    """

    def __init__(
        self,
        model: nn.Module,
        probe_images: torch.Tensor,
        probe_labels: torch.Tensor | None = None,
        wandb_run=None,
        **kwargs,
    ):
        # Import wandb
        try:
            import wandb

            self.wandb = wandb
        except ImportError:
            raise ImportError("wandb is required for WandBAttentionCallback")

        self.wandb_run = wandb_run or wandb.run

        # Create log function that logs to wandb
        def wandb_log(metrics: dict, step: int, epoch: int):
            if self.wandb_run:
                self.wandb_run.log(metrics, step=step)

        super().__init__(
            model,
            probe_images,
            probe_labels,
            log_fn=wandb_log,
            **kwargs,
        )

    def on_epoch_end(self, epoch: int, **kwargs) -> dict:
        """Log epoch metrics and visualizations to WandB."""
        metrics = super().on_epoch_end(epoch, **kwargs)

        # Log attribution visualizations as images
        if epoch % self.attribution_interval == 0:
            for method_name in self.attribution_methods:
                if self.attribution_history[method_name]:
                    _, attr = self.attribution_history[method_name][-1]

                    # Create heatmap image
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(4, 4))
                    im = ax.imshow(attr.numpy(), cmap="jet")
                    ax.set_title(f"{method_name} (epoch {epoch})")
                    plt.colorbar(im, ax=ax)
                    plt.tight_layout()

                    self.wandb_run.log(
                        {f"attribution/{method_name}_heatmap": self.wandb.Image(fig)},
                        step=self._step,
                    )

                    plt.close(fig)

        return metrics
