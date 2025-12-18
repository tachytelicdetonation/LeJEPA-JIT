"""Health status indicators and recommendations for training monitoring.

Provides functions that analyze training metrics and return actionable
status indicators with recommendations for improving training.

Based on research from:
- Transformer Interpretability Beyond Attention (CVPR 2021)
- Head Pursuit: Attention Specialization (2024)
- SSL Representation Quality benchmarking
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class HealthStatus(IntEnum):
    """Health status levels for WandB logging."""

    CRITICAL = 0
    WARNING = 1
    GOOD = 2


@dataclass
class HealthResult:
    """Result of a health check with status and recommendation."""

    status: HealthStatus
    score: float  # 0-1 normalized score
    message: str  # Short description
    recommendation: str  # Actionable suggestion
    details: dict[str, Any]  # Additional metrics


def check_representation_health(
    effective_dim: float | None = None,
    isotropy: float | None = None,
    norm_mean: float | None = None,
    norm_std: float | None = None,
    variance: float | None = None,
    cov_diag_min: float | None = None,
) -> HealthResult:
    """
    Check representation quality and collapse risk.

    Args:
        effective_dim: Effective dimensionality of embeddings
        isotropy: Isotropy measure (spread in directions)
        norm_mean: Mean embedding norm
        norm_std: Embedding norm standard deviation
        variance: Representation variance
        cov_diag_min: Minimum diagonal of covariance matrix

    Returns:
        HealthResult with status, score, and recommendations
    """
    issues = []
    score = 1.0
    details = {}

    # Check effective dimension
    if effective_dim is not None:
        details["effective_dim"] = effective_dim
        if effective_dim < 10:
            issues.append(("critical", "Severe collapse detected (dim < 10)"))
            score = min(score, 0.1)
        elif effective_dim < 50:
            issues.append(
                ("warning", f"Low dimensional embeddings (dim={effective_dim:.0f})")
            )
            score = min(score, 0.5)
        else:
            score = min(score, min(1.0, effective_dim / 100))

    # Check isotropy
    if isotropy is not None:
        details["isotropy"] = isotropy
        if isotropy < 0.1:
            issues.append(("critical", "Embeddings collapsed to single direction"))
            score = min(score, 0.1)
        elif isotropy < 0.3:
            issues.append(("warning", f"Low isotropy ({isotropy:.2f})"))
            score = min(score, 0.4)

    # Check norm stability
    if norm_mean is not None and norm_std is not None and norm_mean > 0:
        norm_cv = norm_std / norm_mean
        details["norm_cv"] = norm_cv
        if norm_cv > 0.8:
            issues.append(("warning", f"High norm variance (CV={norm_cv:.2f})"))
            score = min(score, 0.6)
        elif norm_cv > 0.5:
            issues.append(("info", f"Moderate norm variance (CV={norm_cv:.2f})"))

    # Check variance
    if variance is not None:
        details["variance"] = variance
        if variance < 0.01:
            issues.append(("critical", "Near-zero variance (collapse)"))
            score = min(score, 0.1)
        elif variance < 0.1:
            issues.append(("warning", "Low variance"))
            score = min(score, 0.5)

    # Check covariance diagonal
    if cov_diag_min is not None:
        details["cov_diag_min"] = cov_diag_min
        if cov_diag_min < 1e-6:
            issues.append(("warning", "Some dimensions have near-zero variance"))
            score = min(score, 0.6)

    # Determine overall status
    if any(level == "critical" for level, _ in issues):
        status = HealthStatus.CRITICAL
        recommendation = (
            "Severe collapse detected. Try: increase sigreg_lambda, "
            "reduce LR, check data augmentation, verify loss is decreasing."
        )
    elif any(level == "warning" for level, _ in issues):
        status = HealthStatus.WARNING
        recommendation = (
            "Representation quality concerns. Consider: stronger augmentation, "
            "more training epochs, or adjusting sigreg weight."
        )
    else:
        status = HealthStatus.GOOD
        recommendation = "Representation health is good."

    message = "; ".join(msg for _, msg in issues) if issues else "Healthy embeddings"

    return HealthResult(
        status=status,
        score=score,
        message=message,
        recommendation=recommendation,
        details=details,
    )


def check_attention_health(
    entropy_mean: float | None = None,
    entropy_std: float | None = None,
    head_redundancy: float | None = None,
    pattern_distribution: dict[str, float] | None = None,
    effective_rank: float | None = None,
    head_entropy_min: float | None = None,
) -> HealthResult:
    """
    Check attention pattern health and head diversity.

    Args:
        entropy_mean: Mean attention entropy across heads
        entropy_std: Standard deviation of attention entropy
        head_redundancy: Fraction of redundant heads (>0.9 similarity)
        pattern_distribution: Distribution of attention pattern types
        effective_rank: Effective rank of attention matrix
        head_entropy_min: Minimum head entropy (dead head indicator)

    Returns:
        HealthResult with status, score, and recommendations
    """
    issues = []
    score = 1.0
    details = {}

    # Check entropy (attention spread)
    if entropy_mean is not None:
        details["entropy_mean"] = entropy_mean
        if entropy_mean < 0.5:
            issues.append(("critical", "Attention collapsed to few tokens"))
            score = min(score, 0.2)
        elif entropy_mean < 1.0:
            issues.append(("warning", f"Low attention entropy ({entropy_mean:.2f})"))
            score = min(score, 0.5)
        elif entropy_mean > 4.0:
            issues.append(("info", "Very diffuse attention (may be unfocused)"))

    # Check for dead heads
    if head_entropy_min is not None:
        details["head_entropy_min"] = head_entropy_min
        if head_entropy_min < 0.1:
            issues.append(("warning", "Some heads have near-zero entropy (dead heads)"))
            score = min(score, 0.6)

    # Check head redundancy
    if head_redundancy is not None:
        details["head_redundancy"] = head_redundancy
        if head_redundancy > 0.7:
            issues.append(
                ("warning", f"{head_redundancy * 100:.0f}% heads are redundant")
            )
            score = min(score, 0.4)
        elif head_redundancy > 0.5:
            issues.append(("info", f"{head_redundancy * 100:.0f}% head redundancy"))
            score = min(score, 0.7)

    # Check pattern distribution
    if pattern_distribution is not None:
        details["pattern_distribution"] = pattern_distribution
        radioactive = pattern_distribution.get("radioactive", 0)
        parallel = pattern_distribution.get("parallel", 0)

        if radioactive > 0.7:
            issues.append(
                ("warning", f"{radioactive * 100:.0f}% heads attend to single token")
            )
            score = min(score, 0.5)
        if parallel > 0.6:
            issues.append(
                ("info", f"{parallel * 100:.0f}% heads have uniform attention")
            )

    # Check effective rank
    if effective_rank is not None:
        details["effective_rank"] = effective_rank
        if effective_rank < 2:
            issues.append(("warning", "Very low attention rank"))
            score = min(score, 0.5)

    # Determine overall status
    if any(level == "critical" for level, _ in issues):
        status = HealthStatus.CRITICAL
        recommendation = (
            "Attention collapsed to few tokens. Check for dead heads, "
            "consider longer warmup, or reduce learning rate."
        )
    elif any(level == "warning" for level, _ in issues):
        status = HealthStatus.WARNING
        recommendation = (
            "Attention patterns show concerns. Model may benefit from "
            "attention regularization or head pruning experiments."
        )
    else:
        status = HealthStatus.GOOD
        recommendation = "Attention patterns are healthy."

    message = (
        "; ".join(msg for _, msg in issues) if issues else "Diverse attention patterns"
    )

    return HealthResult(
        status=status,
        score=score,
        message=message,
        recommendation=recommendation,
        details=details,
    )


def check_gradient_health(
    grad_norm: float | None = None,
    param_norm: float | None = None,
    nan_count: int = 0,
    inf_count: int = 0,
    layer_grad_norms: list[float] | None = None,
) -> HealthResult:
    """
    Check gradient health and training stability.

    Args:
        grad_norm: Overall gradient norm
        param_norm: Overall parameter norm
        nan_count: Number of NaN gradients
        inf_count: Number of Inf gradients
        layer_grad_norms: Per-layer gradient norms for flow analysis

    Returns:
        HealthResult with status, score, and recommendations
    """
    issues = []
    score = 1.0
    details = {}

    # Check for NaN/Inf
    if nan_count > 0 or inf_count > 0:
        details["nan_count"] = nan_count
        details["inf_count"] = inf_count
        issues.append(
            (
                "critical",
                f"NaN/Inf gradients detected ({nan_count} NaN, {inf_count} Inf)",
            )
        )
        score = 0.0

    # Check gradient norm
    if grad_norm is not None:
        details["grad_norm"] = grad_norm
        if grad_norm > 100.0:
            issues.append(("critical", f"Exploding gradients (norm={grad_norm:.1f})"))
            score = min(score, 0.1)
        elif grad_norm > 10.0:
            issues.append(("warning", f"High gradient norm ({grad_norm:.2f})"))
            score = min(score, 0.5)
        elif grad_norm < 1e-7:
            issues.append(("critical", "Vanishing gradients"))
            score = min(score, 0.1)
        elif grad_norm < 1e-4:
            issues.append(("warning", "Very small gradients"))
            score = min(score, 0.5)

    # Check gradient-to-parameter ratio
    if grad_norm is not None and param_norm is not None and param_norm > 0:
        grad_to_param = grad_norm / param_norm
        details["grad_to_param"] = grad_to_param
        if grad_to_param > 0.1:
            issues.append(
                ("warning", f"Large update steps (ratio={grad_to_param:.3f})")
            )
            score = min(score, 0.6)

    # Check layer gradient flow
    if layer_grad_norms is not None and len(layer_grad_norms) > 1:
        details["layer_grad_norms"] = layer_grad_norms
        # Check for gradient flow issues
        max_norm = max(layer_grad_norms)
        min_norm = min(layer_grad_norms)
        if max_norm > 0 and min_norm / max_norm < 0.01:
            issues.append(
                (
                    "warning",
                    "Poor gradient flow (some layers have very small gradients)",
                )
            )
            score = min(score, 0.6)

    # Determine overall status
    if any(level == "critical" for level, _ in issues):
        status = HealthStatus.CRITICAL
        if nan_count > 0 or inf_count > 0:
            recommendation = (
                "NaN/Inf gradients detected! Reduce LR, enable gradient clipping, "
                "check loss function for numerical issues."
            )
        elif grad_norm is not None and grad_norm > 100:
            recommendation = (
                "Exploding gradients. Enable gradient clipping, reduce LR, "
                "or check for unstable loss computation."
            )
        else:
            recommendation = (
                "Vanishing gradients. Increase LR, check for dead layers, "
                "or consider different initialization."
            )
    elif any(level == "warning" for level, _ in issues):
        status = HealthStatus.WARNING
        recommendation = (
            "Gradient health concerns. Consider gradient clipping or "
            "adjusting learning rate schedule."
        )
    else:
        status = HealthStatus.GOOD
        recommendation = "Gradient flow is healthy."

    message = "; ".join(msg for _, msg in issues) if issues else "Stable gradients"

    return HealthResult(
        status=status,
        score=score,
        message=message,
        recommendation=recommendation,
        details=details,
    )


def check_training_health(
    loss: float | None = None,
    loss_delta: float | None = None,
    accuracy: float | None = None,
    accuracy_delta: float | None = None,
    epoch: int | None = None,
) -> HealthResult:
    """
    Check overall training progress and convergence.

    Args:
        loss: Current loss value
        loss_delta: Change in loss from previous epoch
        accuracy: Current validation accuracy
        accuracy_delta: Change in accuracy from previous epoch
        epoch: Current epoch number

    Returns:
        HealthResult with status, score, and recommendations
    """
    issues = []
    score = 1.0
    details = {}

    # Check loss
    if loss is not None:
        details["loss"] = loss
        if loss > 100 or not (loss == loss):  # NaN check
            issues.append(("critical", "Loss is NaN or exploded"))
            score = 0.0
        elif loss > 10 and epoch is not None and epoch > 10:
            issues.append(("warning", "Loss remains high after warmup"))
            score = min(score, 0.5)

    # Check loss trend
    if loss_delta is not None:
        details["loss_delta"] = loss_delta
        if loss_delta > 0.5 and epoch is not None and epoch > 5:
            issues.append(("warning", f"Loss increased by {loss_delta:.3f}"))
            score = min(score, 0.6)

    # Check accuracy
    if accuracy is not None:
        details["accuracy"] = accuracy
        if epoch is not None and epoch > 50 and accuracy < 0.3:
            issues.append(("warning", "Low accuracy after 50 epochs"))
            score = min(score, 0.5)

    # Check accuracy trend
    if accuracy_delta is not None:
        details["accuracy_delta"] = accuracy_delta
        if accuracy_delta < -0.1:
            issues.append(("warning", f"Accuracy dropped by {-accuracy_delta:.1%}"))
            score = min(score, 0.7)

    # Determine overall status
    if any(level == "critical" for level, _ in issues):
        status = HealthStatus.CRITICAL
        recommendation = (
            "Training is unstable. Check for NaN loss, reduce LR, "
            "or verify data pipeline."
        )
    elif any(level == "warning" for level, _ in issues):
        status = HealthStatus.WARNING
        recommendation = (
            "Training progress concerns. Consider adjusting hyperparameters "
            "or extending training."
        )
    else:
        status = HealthStatus.GOOD
        recommendation = "Training is progressing well."

    message = "; ".join(msg for _, msg in issues) if issues else "Training on track"

    return HealthResult(
        status=status,
        score=score,
        message=message,
        recommendation=recommendation,
        details=details,
    )


def generate_health_summary_html(
    rep_health: HealthResult,
    attn_health: HealthResult,
    grad_health: HealthResult,
    training_health: HealthResult | None = None,
    epoch: int | None = None,
    metrics: dict[str, Any] | None = None,
) -> str:
    """
    Generate comprehensive HTML summary with automatic analysis for WandB logging.

    Args:
        rep_health: Representation health result
        attn_health: Attention health result
        grad_health: Gradient health result
        training_health: Optional training health result
        epoch: Current epoch number
        metrics: Current epoch metrics dict for deeper analysis

    Returns:
        HTML string for wandb.Html logging
    """
    status_colors = {
        HealthStatus.GOOD: "#22c55e",  # green
        HealthStatus.WARNING: "#eab308",  # yellow
        HealthStatus.CRITICAL: "#ef4444",  # red
    }

    status_icons = {
        HealthStatus.GOOD: "&#x2713;",  # checkmark
        HealthStatus.WARNING: "&#x26A0;",  # warning
        HealthStatus.CRITICAL: "&#x2717;",  # X
    }

    status_labels = {
        HealthStatus.GOOD: "GOOD",
        HealthStatus.WARNING: "WARN",
        HealthStatus.CRITICAL: "CRIT",
    }

    def format_health_line(name: str, health: HealthResult) -> str:
        color = status_colors[health.status]
        icon = status_icons[health.status]
        label = status_labels[health.status]
        return (
            f'<div style="color: {color}; margin: 6px 0; padding: 4px; '
            f'background: rgba(255,255,255,0.05); border-radius: 4px;">'
            f"{icon} <strong>[{label}] {name}</strong>: {health.message}"
            f' <span style="opacity: 0.7;">(score: {health.score:.2f})</span></div>'
        )

    def format_metric(
        name: str, value: float | None, unit: str = "", good_range: str = ""
    ) -> str:
        if value is None:
            return ""
        range_html = (
            f" <span style='opacity: 0.6;'>(good: {good_range})</span>"
            if good_range
            else ""
        )
        return (
            f'<div style="margin: 2px 0; color: #b5b5b5;">'
            f"  &bull; {name}: <strong style='color: #e5e5e5;'>{value:.3f}</strong>{unit}"
            f"{range_html}"
            f"</div>"
        )

    # Build header with epoch
    epoch_str = f" - Epoch {epoch}" if epoch is not None else ""
    lines = [
        '<div style="font-family: -apple-system, BlinkMacSystemFont, monospace; '
        'padding: 12px; background: #1a1a2e; border-radius: 8px; max-width: 600px;">',
        f'<div style="font-size: 16px; margin-bottom: 10px; color: #ffffff; '
        f'border-bottom: 1px solid #333; padding-bottom: 8px;">'
        f"<strong>&#x1F4CA; Training Health Report{epoch_str}</strong></div>",
    ]

    # Status summary section
    lines.append('<div style="margin-bottom: 12px;">')
    lines.append(format_health_line("Representation", rep_health))
    lines.append(format_health_line("Attention", attn_health))
    lines.append(format_health_line("Gradients", grad_health))
    if training_health is not None:
        lines.append(format_health_line("Training", training_health))
    lines.append("</div>")

    # Key metrics section with interpretation
    if metrics:
        lines.append(
            '<div style="margin: 12px 0; padding: 8px; background: rgba(255,255,255,0.03); '
            'border-radius: 4px;">'
        )
        lines.append(
            '<div style="color: #a5a5a5; font-size: 12px; margin-bottom: 6px;">'
            "<strong>Key Metrics & Interpretation</strong></div>"
        )

        # Representation metrics
        eff_dim = metrics.get("rep/effective_dim")
        isotropy = metrics.get("rep/isotropy")
        if eff_dim is not None or isotropy is not None:
            lines.append(
                '<div style="margin: 4px 0; color: #8b8bcd;"><em>Representation:</em></div>'
            )
            if eff_dim is not None:
                lines.append(format_metric("Effective Dim", eff_dim, "", ">50"))
            if isotropy is not None:
                lines.append(format_metric("Isotropy", isotropy, "", ">0.3"))

        # Attention metrics
        entropy = metrics.get("train/attn_entropy") or metrics.get(
            "attn/head_entropy_mean"
        )
        if entropy is not None:
            lines.append(
                '<div style="margin: 4px 0; color: #8bcd8b;"><em>Attention:</em></div>'
            )
            lines.append(format_metric("Entropy", entropy, " bits", "2-5"))

        # Training metrics
        loss = metrics.get("train/loss") or metrics.get("epoch_loss")
        acc = metrics.get("val_accuracy_nette") or metrics.get("train/accuracy")
        if loss is not None or acc is not None:
            lines.append(
                '<div style="margin: 4px 0; color: #cd8b8b;"><em>Training:</em></div>'
            )
            if loss is not None:
                lines.append(format_metric("Loss", loss, "", "<1.0"))
            if acc is not None:
                lines.append(format_metric("Accuracy", acc, "%", ">70"))

        # KNN metrics (representation quality indicator)
        knn = metrics.get("knn/nette_top1")
        if knn is not None:
            lines.append(
                '<div style="margin: 4px 0; color: #cdcd8b;"><em>Feature Quality:</em></div>'
            )
            lines.append(format_metric("KNN Accuracy", knn, "%", ">60"))

        lines.append("</div>")

    # Automatic interpretation section
    all_health = [rep_health, attn_health, grad_health]
    if training_health:
        all_health.append(training_health)

    worst_health = min(all_health, key=lambda h: h.status)

    # Generate automatic analysis
    lines.append(
        '<div style="margin: 12px 0; padding: 8px; background: rgba(255,255,255,0.03); '
        'border-radius: 4px;">'
    )
    lines.append(
        '<div style="color: #a5a5a5; font-size: 12px; margin-bottom: 6px;">'
        "<strong>&#x1F50D; Automatic Analysis</strong></div>"
    )

    analysis_lines = []

    # Representation analysis
    if rep_health.status == HealthStatus.CRITICAL:
        analysis_lines.append(
            "&#x1F534; <strong>COLLAPSE DETECTED</strong>: Model is producing identical "
            "embeddings for different inputs. This is a critical failure mode."
        )
    elif rep_health.status == HealthStatus.WARNING:
        analysis_lines.append(
            "&#x1F7E1; Representation quality is degraded. Features may lack "
            "discriminative power for downstream tasks."
        )
    else:
        analysis_lines.append(
            "&#x1F7E2; Representations are diverse and well-distributed."
        )

    # Attention analysis
    if attn_health.status == HealthStatus.CRITICAL:
        analysis_lines.append(
            "&#x1F534; <strong>ATTENTION COLLAPSED</strong>: All heads attending to same tokens. "
            "Model may not be learning meaningful patterns."
        )
    elif attn_health.status == HealthStatus.WARNING:
        analysis_lines.append(
            "&#x1F7E1; Attention patterns show redundancy or limited diversity."
        )
    else:
        analysis_lines.append(
            "&#x1F7E2; Attention patterns are healthy with good head diversity."
        )

    # Gradient analysis
    if grad_health.status == HealthStatus.CRITICAL:
        analysis_lines.append(
            "&#x1F534; <strong>GRADIENT ISSUE</strong>: Training may be unstable. "
            "Check for NaN/Inf values."
        )
    elif grad_health.status == HealthStatus.WARNING:
        analysis_lines.append(
            "&#x1F7E1; Gradient flow has minor concerns. Monitor for instability."
        )
    else:
        analysis_lines.append(
            "&#x1F7E2; Gradient flow is stable and training normally."
        )

    for line in analysis_lines:
        lines.append(
            f'<div style="color: #d5d5d5; margin: 4px 0; font-size: 12px;">{line}</div>'
        )

    lines.append("</div>")

    # Recommendations section
    if worst_health.status != HealthStatus.GOOD:
        rec_color = status_colors[worst_health.status]
        lines.extend(
            [
                f'<div style="margin-top: 12px; padding: 8px; background: {rec_color}22; '
                f'border-left: 3px solid {rec_color}; border-radius: 0 4px 4px 0;">',
                f'<div style="color: {rec_color}; font-size: 12px; font-weight: bold;">'
                f"&#x1F4A1; Recommended Action</div>",
                f'<div style="color: #d5d5d5; font-size: 12px; margin-top: 4px;">'
                f"{worst_health.recommendation}</div>",
                "</div>",
            ]
        )
    else:
        lines.extend(
            [
                '<div style="margin-top: 12px; padding: 8px; background: #22c55e22; '
                'border-left: 3px solid #22c55e; border-radius: 0 4px 4px 0;">',
                '<div style="color: #22c55e; font-size: 12px; font-weight: bold;">'
                "&#x2713; All Systems Healthy</div>",
                '<div style="color: #d5d5d5; font-size: 12px; margin-top: 4px;">'
                "Training is progressing well. Continue monitoring.</div>",
                "</div>",
            ]
        )

    # Quick reference footer
    lines.extend(
        [
            '<div style="margin-top: 12px; padding-top: 8px; border-top: 1px solid #333; '
            'color: #666; font-size: 10px;">',
            "<strong>Quick Reference:</strong> "
            "Effective Dim &gt;50 = diverse features | "
            "Isotropy &gt;0.3 = spread in all directions | "
            "Entropy 2-5 = balanced attention",
            "</div>",
        ]
    )

    lines.append("</div>")

    return "\n".join(lines)


def compute_overall_health_score(
    rep_health: HealthResult,
    attn_health: HealthResult,
    grad_health: HealthResult,
) -> float:
    """
    Compute overall health score (0-1) from individual health checks.

    Uses weighted average with critical issues dominating.

    Args:
        rep_health: Representation health result
        attn_health: Attention health result
        grad_health: Gradient health result

    Returns:
        Overall health score (0-1)
    """
    # Weights for each health aspect
    weights = {
        "rep": 0.4,  # Representation quality is most important for SSL
        "attn": 0.3,  # Attention patterns matter for interpretability
        "grad": 0.3,  # Gradient health for training stability
    }

    # If any critical, overall score is very low
    if any(
        h.status == HealthStatus.CRITICAL
        for h in [rep_health, attn_health, grad_health]
    ):
        return min(rep_health.score, attn_health.score, grad_health.score) * 0.5

    # Weighted average
    return (
        weights["rep"] * rep_health.score
        + weights["attn"] * attn_health.score
        + weights["grad"] * grad_health.score
    )
