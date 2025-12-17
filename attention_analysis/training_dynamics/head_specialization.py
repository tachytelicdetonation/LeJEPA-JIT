"""
Head Specialization Tracking.

Analyzes how attention heads specialize during training and what roles
they develop.

Concepts:
- Monosemantic: Head fires for ONE type of input (e.g., edges, specific color)
- Polysemantic: Head fires for MULTIPLE unrelated inputs

Methods:
- Ablation-based importance: Zero out heads and measure accuracy drop
- Activation similarity: Cluster heads by their activation patterns
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_analysis.utils.hooks import HookManager


@dataclass
class HeadImportance:
    """Importance scores for each attention head."""

    layer_idx: int
    head_idx: int
    importance_score: float
    ablation_loss_delta: float | None = None


class HeadSpecializationTracker:
    """
    Tracks head specialization patterns during training.

    Computes head importance via ablation and tracks how heads
    specialize over time.

    Args:
        model: The model to analyze.
        probe_images: Fixed set of images for consistent measurements.
        probe_labels: Labels for probe images (for ablation).
    """

    def __init__(
        self,
        model: nn.Module,
        probe_images: torch.Tensor | None = None,
        probe_labels: torch.Tensor | None = None,
    ):
        self.model = model
        self.probe_images = probe_images
        self.probe_labels = probe_labels
        self.importance_history: list[list[HeadImportance]] = []
        self.similarity_history: list[torch.Tensor] = []

    @torch.no_grad()
    def compute_head_importance(
        self,
        images: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        method: str = "ablation",
    ) -> list[HeadImportance]:
        """
        Compute importance scores for all attention heads.

        Args:
            images: Images to test on. Uses probe_images if None.
            labels: Labels for accuracy computation.
            method: Method for importance. Options:
                - "ablation": Zero out heads, measure loss change
                - "gradient": Gradient-based importance
                - "activation": Activation magnitude

        Returns:
            List of HeadImportance for each head.
        """
        if images is None:
            images = self.probe_images
        if labels is None:
            labels = self.probe_labels

        if images is None:
            return []

        device = next(self.model.parameters()).device
        images = images.to(device)
        if labels is not None:
            labels = labels.to(device)

        if method == "ablation":
            return self._ablation_importance(images, labels)
        elif method == "gradient":
            return self._gradient_importance(images, labels)
        elif method == "activation":
            return self._activation_importance(images)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _ablation_importance(
        self,
        images: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> list[HeadImportance]:
        """Compute importance via head ablation."""
        device = next(self.model.parameters()).device
        results = []

        # Get baseline output
        self.model.eval()
        if hasattr(self.model, "encoder"):
            base_embeddings = self.model.encoder(images)
            if hasattr(self.model, "probe"):
                base_output = self.model.probe(base_embeddings)
            else:
                base_output = base_embeddings
        else:
            base_output = self.model(images)

        if labels is not None:
            base_loss = F.cross_entropy(base_output, labels).item()
        else:
            base_loss = 0.0

        # Find attention layers
        attention_layers = self._find_attention_layers()

        for layer_idx, (name, layer) in enumerate(attention_layers):
            num_heads = getattr(layer, "num_heads", 6)

            for head_idx in range(num_heads):
                # Enable head masking
                if hasattr(layer, "head_mask"):
                    original_mask = layer.head_mask
                    mask = torch.ones(num_heads, device=device)
                    mask[head_idx] = 0
                    layer.head_mask = mask

                    # Forward with masked head
                    if hasattr(self.model, "encoder"):
                        ablated_embeddings = self.model.encoder(images)
                        if hasattr(self.model, "probe"):
                            ablated_output = self.model.probe(ablated_embeddings)
                        else:
                            ablated_output = ablated_embeddings
                    else:
                        ablated_output = self.model(images)

                    # Compute loss change
                    if labels is not None:
                        ablated_loss = F.cross_entropy(ablated_output, labels).item()
                        loss_delta = ablated_loss - base_loss
                    else:
                        loss_delta = None

                    # Restore mask
                    layer.head_mask = original_mask

                    results.append(
                        HeadImportance(
                            layer_idx=layer_idx,
                            head_idx=head_idx,
                            importance_score=loss_delta if loss_delta else 0.0,
                            ablation_loss_delta=loss_delta,
                        )
                    )
                else:
                    # Can't ablate without mask support
                    results.append(
                        HeadImportance(
                            layer_idx=layer_idx,
                            head_idx=head_idx,
                            importance_score=0.0,
                            ablation_loss_delta=None,
                        )
                    )

        return results

    def _gradient_importance(
        self,
        images: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> list[HeadImportance]:
        """Compute importance via gradient magnitude."""
        results = []

        # Enable attention capture
        attention_layers = self._find_attention_layers()
        for name, layer in attention_layers:
            layer.output_attention = True

        try:
            # Forward
            if hasattr(self.model, "encoder"):
                embeddings = self.model.encoder(images)
                if hasattr(self.model, "probe"):
                    output = self.model.probe(embeddings)
                else:
                    output = embeddings
            else:
                output = self.model(images)

            # Backward
            if labels is not None:
                loss = F.cross_entropy(output, labels)
            else:
                loss = output.sum()  # Fallback

            self.model.zero_grad()
            loss.backward()

            # Collect importance from attention gradients
            for layer_idx, (name, layer) in enumerate(attention_layers):
                if layer.attn_map is not None:
                    attn = layer.attn_map  # (B, H, N, N)
                    num_heads = attn.shape[1]

                    for head_idx in range(num_heads):
                        # Importance = magnitude of attention for this head
                        head_attn = attn[:, head_idx]  # (B, N, N)
                        importance = head_attn.abs().mean().item()

                        results.append(
                            HeadImportance(
                                layer_idx=layer_idx,
                                head_idx=head_idx,
                                importance_score=importance,
                            )
                        )
                else:
                    num_heads = getattr(layer, "num_heads", 6)
                    for head_idx in range(num_heads):
                        results.append(
                            HeadImportance(
                                layer_idx=layer_idx,
                                head_idx=head_idx,
                                importance_score=0.0,
                            )
                        )
        finally:
            for name, layer in attention_layers:
                layer.output_attention = False
                layer.attn_map = None

        return results

    def _activation_importance(
        self,
        images: torch.Tensor,
    ) -> list[HeadImportance]:
        """Compute importance via activation magnitude."""
        device = next(self.model.parameters()).device
        results = []

        with HookManager(self.model) as hooks:
            hooks.enable_attention_capture()
            hooks.register_attention_hooks()

            self.model.eval()
            if hasattr(self.model, "encoder"):
                _ = self.model.encoder(images)
            else:
                _ = self.model(images)

            attentions = hooks.get_attention_maps()

        for layer_idx, attn in enumerate(attentions):
            attn = attn.to(device)
            num_heads = attn.shape[1]

            for head_idx in range(num_heads):
                head_attn = attn[:, head_idx]  # (B, N, N)

                # Importance = variance of attention (focused heads have low variance)
                # Or could use entropy, max value, etc.
                importance = head_attn.var().item()

                results.append(
                    HeadImportance(
                        layer_idx=layer_idx,
                        head_idx=head_idx,
                        importance_score=importance,
                    )
                )

        return results

    def _find_attention_layers(self) -> list[tuple[str, nn.Module]]:
        """Find all attention layers."""
        layers = []

        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "blocks"):
            for i, block in enumerate(self.model.encoder.blocks):
                if hasattr(block, "attn"):
                    layers.append((f"encoder.blocks.{i}.attn", block.attn))
        elif hasattr(self.model, "blocks"):
            for i, block in enumerate(self.model.blocks):
                if hasattr(block, "attn"):
                    layers.append((f"blocks.{i}.attn", block.attn))

        return layers

    @torch.no_grad()
    def compute_head_similarity(
        self,
        images: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute similarity matrix between all attention heads.

        Similar heads may be redundant. Uses cosine similarity of
        flattened attention patterns.

        Args:
            images: Images to compute similarity on.

        Returns:
            Similarity matrix of shape (num_heads_total, num_heads_total).
        """
        if images is None:
            images = self.probe_images

        if images is None:
            return torch.zeros(1, 1)

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

        # Flatten all heads into vectors
        head_vectors = []

        for attn in attentions:
            attn = attn.to(device)
            B, H, N, _ = attn.shape

            for h in range(H):
                # Flatten attention pattern: (B, N, N) -> (B * N * N,)
                vec = attn[:, h].flatten()
                head_vectors.append(vec)

        if not head_vectors:
            return torch.zeros(1, 1)

        # Stack and compute cosine similarity
        vectors = torch.stack(head_vectors)  # (num_heads, vec_dim)
        vectors = F.normalize(vectors, dim=-1)
        similarity = torch.mm(vectors, vectors.t())  # (num_heads, num_heads)

        return similarity


def compute_head_importance(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor | None = None,
    method: str = "activation",
) -> list[HeadImportance]:
    """
    Compute importance scores for all attention heads.

    Convenience function that creates a tracker and computes importance.

    Args:
        model: The model.
        images: Images to evaluate on.
        labels: Optional labels for ablation method.
        method: Importance method ("ablation", "gradient", "activation").

    Returns:
        List of HeadImportance scores.
    """
    tracker = HeadSpecializationTracker(model, images, labels)
    return tracker.compute_head_importance(images, labels, method)


def compute_head_similarity(
    model: nn.Module,
    images: torch.Tensor,
) -> torch.Tensor:
    """
    Compute similarity matrix between all attention heads.

    Args:
        model: The model.
        images: Images to compute on.

    Returns:
        Cosine similarity matrix.
    """
    tracker = HeadSpecializationTracker(model, images)
    return tracker.compute_head_similarity(images)
