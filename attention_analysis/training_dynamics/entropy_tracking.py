"""
Attention Entropy Tracking during Training.

Tracks how attention distributions evolve during training.
- Low entropy = focused, confident attention
- High entropy = diffuse, uncertain attention
- Typically decreases during training as model learns to focus
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from attention_analysis.utils.hooks import HookManager


@dataclass
class EntropySnapshot:
    """Snapshot of attention entropy at a training step."""

    step: int
    epoch: int
    layer_entropies: list[float]  # Per-layer mean entropy
    head_entropies: list[list[float]]  # Per-layer, per-head entropy
    mean_entropy: float
    std_entropy: float


class AttentionEntropyTracker:
    """
    Tracks attention entropy over training.

    Records entropy statistics at specified intervals and provides
    analysis of how attention sharpens during training.

    Args:
        model: The model to track.
        log_interval: Steps between entropy recordings.
        probe_images: Fixed set of images for consistent measurements.
    """

    def __init__(
        self,
        model: nn.Module,
        log_interval: int = 100,
        probe_images: torch.Tensor | None = None,
    ):
        self.model = model
        self.log_interval = log_interval
        self.probe_images = probe_images
        self.history: list[EntropySnapshot] = []
        self._step = 0

    def step(
        self,
        images: torch.Tensor | None = None,
        epoch: int = 0,
    ) -> EntropySnapshot | None:
        """
        Record entropy if at log interval.

        Args:
            images: Images to compute entropy on. Uses probe_images if None.
            epoch: Current epoch number.

        Returns:
            EntropySnapshot if recorded, None otherwise.
        """
        self._step += 1

        if self._step % self.log_interval != 0:
            return None

        if images is None:
            images = self.probe_images

        if images is None:
            return None

        snapshot = self.compute_snapshot(images, self._step, epoch)
        self.history.append(snapshot)
        return snapshot

    @torch.no_grad()
    def compute_snapshot(
        self,
        images: torch.Tensor,
        step: int = 0,
        epoch: int = 0,
    ) -> EntropySnapshot:
        """
        Compute entropy snapshot for given images.

        Args:
            images: Images to compute entropy on.
            step: Current training step.
            epoch: Current epoch.

        Returns:
            EntropySnapshot with all entropy statistics.
        """
        device = next(self.model.parameters()).device
        images = images.to(device)

        with HookManager(self.model) as hooks:
            hooks.enable_attention_capture()
            hooks.register_attention_hooks()

            self.model.eval()
            if hasattr(self.model, "encoder"):
                _ = self.model.encoder(images)
            else:
                _ = self.model(images)

            attentions = hooks.get_attention_maps()

        if not attentions:
            return EntropySnapshot(
                step=step,
                epoch=epoch,
                layer_entropies=[],
                head_entropies=[],
                mean_entropy=0.0,
                std_entropy=0.0,
            )

        layer_entropies = []
        head_entropies = []
        all_entropies = []

        for attn in attentions:
            attn = attn.to(device)
            # attn shape: (B, H, N, N)

            # Compute entropy per head
            # Entropy = -sum(p * log(p)) along attention dimension
            entropy = -(attn * torch.log(attn + 1e-10)).sum(dim=-1)  # (B, H, N)
            entropy = entropy.mean(dim=(0, 2))  # (H,) - mean over batch and tokens

            head_entropies.append(entropy.cpu().tolist())
            layer_entropies.append(entropy.mean().item())
            all_entropies.extend(entropy.cpu().tolist())

        return EntropySnapshot(
            step=step,
            epoch=epoch,
            layer_entropies=layer_entropies,
            head_entropies=head_entropies,
            mean_entropy=sum(all_entropies) / len(all_entropies)
            if all_entropies
            else 0.0,
            std_entropy=torch.tensor(all_entropies).std().item()
            if all_entropies
            else 0.0,
        )

    def get_entropy_curve(self) -> tuple[list[int], list[float]]:
        """
        Get mean entropy over training steps.

        Returns:
            (steps, entropies) tuple for plotting.
        """
        steps = [s.step for s in self.history]
        entropies = [s.mean_entropy for s in self.history]
        return steps, entropies

    def get_layer_curves(self) -> dict[int, tuple[list[int], list[float]]]:
        """
        Get per-layer entropy curves.

        Returns:
            Dict mapping layer index to (steps, entropies) tuples.
        """
        if not self.history:
            return {}

        num_layers = len(self.history[0].layer_entropies)
        curves = {}

        for layer_idx in range(num_layers):
            steps = [s.step for s in self.history]
            entropies = [s.layer_entropies[layer_idx] for s in self.history]
            curves[layer_idx] = (steps, entropies)

        return curves

    def get_summary(self) -> dict:
        """Get summary statistics of entropy evolution."""
        if not self.history:
            return {}

        initial = self.history[0]
        final = self.history[-1]

        return {
            "initial_mean_entropy": initial.mean_entropy,
            "final_mean_entropy": final.mean_entropy,
            "entropy_change": final.mean_entropy - initial.mean_entropy,
            "entropy_change_pct": (
                (final.mean_entropy - initial.mean_entropy) / initial.mean_entropy * 100
                if initial.mean_entropy > 0
                else 0.0
            ),
            "num_snapshots": len(self.history),
            "total_steps": final.step,
        }


def compute_attention_entropy(
    attention: torch.Tensor,
    reduce: str = "mean",
) -> torch.Tensor:
    """
    Compute Shannon entropy of attention distributions.

    Args:
        attention: Attention weights, shape (B, H, N, N) or (B, N, N).
        reduce: How to reduce. Options:
            - "none": Return entropy per position
            - "mean": Mean entropy (default)
            - "head": Mean per head

    Returns:
        Entropy tensor. Shape depends on reduce parameter.
    """
    # Ensure 4D
    if attention.dim() == 3:
        attention = attention.unsqueeze(1)  # Add head dimension

    # Compute entropy: -sum(p * log(p))
    entropy = -(attention * torch.log(attention + 1e-10)).sum(dim=-1)  # (B, H, N)

    if reduce == "none":
        return entropy
    elif reduce == "mean":
        return entropy.mean()
    elif reduce == "head":
        return entropy.mean(dim=(0, 2))  # (H,)
    else:
        raise ValueError(f"Unknown reduce: {reduce}")


def compute_layer_entropy_stats(
    model: nn.Module,
    images: torch.Tensor,
) -> dict:
    """
    Compute entropy statistics across all layers.

    Args:
        model: The model.
        images: Input images.

    Returns:
        Dictionary with per-layer and aggregate entropy stats.
    """
    device = next(model.parameters()).device
    images = images.to(device)

    with HookManager(model) as hooks:
        hooks.enable_attention_capture()
        hooks.register_attention_hooks()

        model.eval()
        with torch.no_grad():
            if hasattr(model, "encoder"):
                _ = model.encoder(images)
            else:
                _ = model(images)

        attentions = hooks.get_attention_maps()

    if not attentions:
        return {}

    stats = {
        "per_layer_mean": [],
        "per_layer_std": [],
        "per_layer_head_means": [],
    }

    all_entropies = []

    for layer_idx, attn in enumerate(attentions):
        attn = attn.to(device)
        entropy = compute_attention_entropy(attn, reduce="none")  # (B, H, N)

        layer_mean = entropy.mean().item()
        layer_std = entropy.std().item()
        head_means = entropy.mean(dim=(0, 2)).cpu().tolist()  # (H,)

        stats["per_layer_mean"].append(layer_mean)
        stats["per_layer_std"].append(layer_std)
        stats["per_layer_head_means"].append(head_means)

        all_entropies.append(layer_mean)

    stats["global_mean"] = sum(all_entropies) / len(all_entropies)
    stats["num_layers"] = len(attentions)

    return stats
