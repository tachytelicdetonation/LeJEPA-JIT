"""
Training script for LeJEPA with JiT or ViT encoder.

This script trains the LeJEPA model on ImageNette with:
- SIGReg + Invariance loss
- Online linear probe for monitoring
- Mixed precision training
- Learning rate scheduling (warmup + cosine decay)

Usage:
    # Train with JiT encoder (default)
    python train.py --encoder jit

    # Train with ViT encoder for comparison
    python train.py --encoder vit

    # Custom settings
    python train.py --encoder jit --epochs 200 --batch_size 256 --lr 2e-3
"""

import argparse
import copy
import time
from contextlib import nullcontext
from pathlib import Path
import os

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

import loss_landscapes
import loss_landscapes.metrics

from config import Config, get_config
from data import get_dataloaders
from losses import LeJEPALoss
from models.lejepa import LinearProbe, create_lejepa
from utils.metrics import (
    compute_entropy,
    compute_gini,
    compute_sparsity,
    compute_layer_gradient_stats,
    compute_head_diversity,
    compute_feature_collapse_metrics,
    compute_representation_stats,
    compute_alignment_metrics,
    compute_covariance_metrics,
    compute_global_norms,
    compute_attention_rank,
    compute_linear_cka,
    compute_attention_structure_metrics,
    compute_attention_distance_metrics,
    compute_encoder_block_opt_stats,
    estimate_intrinsic_dim_twonn,
    # New research-backed metrics
    compute_head_specialization,
    classify_attention_patterns,
    compute_representation_qscore,
    compute_rope_position_correlation,
)
from utils.health_checks import (
    check_representation_health,
    check_attention_health,
    check_gradient_health,
    check_training_health,
    generate_health_summary_html,
    compute_overall_health_score,
)
from utils.dashboards import (
    generate_epoch_summary_dashboard,
    generate_epoch_delta_dashboard,
    generate_training_radar,
    generate_pattern_distribution_chart,
    generate_layer_importance_curve,
)

# Attention Analysis Module
from attention_analysis import (
    get_attribution,
    compute_faithfulness_metrics,
    AttentionEntropyTracker,
    HeadSpecializationTracker,
)
from attention_analysis.visualization import (
    compare_methods_visual,
    plot_head_importance,
)


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a CHW or HWC tensor to PIL Image.

    Handles normalization and dtype conversion properly to avoid
    'Cannot handle this data type' errors from PIL.
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # Remove batch dim
    arr = tensor.detach().cpu().numpy()
    # Handle CHW -> HWC
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    # Squeeze single channel
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr.squeeze(2)
    # Normalize to [0, 255]
    arr = np.clip(arr, 0.0, None)  # Ensure non-negative
    if arr.max() <= 1.0 + 1e-6:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _cleanup_gpu_memory(device: torch.device) -> None:
    """Force garbage collection and CUDA cache cleanup."""
    import gc

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _cosine_ema_coeff(
    base: float, final: float, epoch: int, total_epochs: int
) -> float:
    """Cosine schedule for EMA coefficient (common in teacher-student / SWA-EMA setups)."""
    if total_epochs <= 0:
        return float(final)
    # epoch is 1-based in this training script
    t = float(epoch) / float(max(total_epochs, 1))
    return float(
        final
        - 0.5 * (final - base) * (1.0 + torch.cos(torch.tensor(t * torch.pi)).item())
    )


@torch.no_grad()
def _ema_update_(teacher: nn.Module, student: nn.Module, ema_coeff: float) -> None:
    """teacher = ema_coeff*teacher + (1-ema_coeff)*student (params + buffers)."""
    m = float(ema_coeff)
    if m <= 0.0:
        teacher.load_state_dict(student.state_dict(), strict=True)
        return
    if m >= 1.0:
        return

    for teacher_group, student_group in [
        (teacher.parameters(), student.parameters()),
        (teacher.buffers(), student.buffers()),
    ]:
        for t, s in zip(teacher_group, student_group):
            if t.dtype.is_floating_point:
                t.mul_(m)
                t.add_((1.0 - m) * s.to(dtype=t.dtype))
            else:
                # Non-floating buffers are copied directly.
                t.copy_(s)


def _amp_dtype_for_device(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    # CPU autocast supports bfloat16
    return torch.bfloat16


def _autocast_ctx(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=_amp_dtype_for_device(device))


def get_schedulers(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> SequentialLR:
    """Create learning rate scheduler with warmup and cosine decay."""
    if warmup_steps <= 0:
        cosine = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
        return SequentialLR(optimizer, schedulers=[cosine], milestones=[])

    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=1e-5
    )
    return SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )


def train_one_epoch(
    model: nn.Module,
    teacher_model: nn.Module | None,
    probe: LinearProbe,
    loss_fn: LeJEPALoss,
    optimizer: torch.optim.Optimizer,
    scheduler: SequentialLR,
    train_loader,
    device: torch.device,
    epoch: int,
    config: Config,
    scaler: GradScaler,
    attn_grads: list,
    entropy_tracker: AttentionEntropyTracker | None = None,
) -> dict:
    """Train for one epoch."""
    model.train()
    probe.train()
    if teacher_model is not None:
        teacher_model.eval()

    total_loss = 0
    total_sigreg = 0
    total_pred = 0
    total_probe_loss = 0
    total_correct = 0
    total_samples = 0
    total_entropy = 0
    total_gini = 0
    total_sparsity = 0
    total_grad_norm = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    prev_iter_end = time.perf_counter()

    for batch_idx, (crops, labels) in enumerate(pbar):
        iter_start = time.perf_counter()
        data_time = iter_start - prev_iter_end
        # crops is a list of [global1, global2, local1, ..., local6]
        # Move all crops to device
        crops = [c.to(device, non_blocking=True) for c in crops]
        labels = labels.to(device, non_blocking=True)

        # Split into global and local
        global_crops = crops[:2]
        local_crops = crops[2:]

        B = labels.shape[0]

        # Forward pass with mixed precision
        with _autocast_ctx(device, enabled=config.mixed_precision):
            # 1. Forward Global Views
            global_crops_tensor = torch.stack(
                global_crops, dim=1
            )  # (B, 2, C, 224, 224)
            emb_global, proj_global = model(
                global_crops_tensor
            )  # emb: (B*2, D), proj: (2, B, D)

            proj_global_teacher = None
            if teacher_model is not None and config.teacher_student:
                _, proj_global_teacher = teacher_model(global_crops_tensor)

            # 2. Forward Local Views
            if local_crops:
                local_crops_tensor = torch.stack(
                    local_crops, dim=1
                )  # (B, 6, C, 96, 96)
                emb_local, proj_local = model(
                    local_crops_tensor
                )  # emb: (B*6, D), proj: (6, B, D)

                # Concatenate projections (Views, B, D)
                proj = torch.cat([proj_global, proj_local], dim=0)  # (8, B, D)

                # Combine embeddings for probe??
                # Usually probe only on global views or specific view.
                # Let's keep probe on all views IF we want strict monitoring, but paper only monitors global?
                # Minimal example monitored (B*V). Let's monitor Global only to be safe/stable.
                emb_for_probe = emb_global
                labels_for_probe = labels.repeat_interleave(2)
            else:
                proj = proj_global
                emb_for_probe = emb_global
                labels_for_probe = labels.repeat_interleave(2)

            # LeJEPA loss on all projections
            loss_dict = loss_fn(proj, global_proj=proj_global_teacher)
            lejepa_loss = loss_dict["loss"]

            # Linear probe on Global embeddings (detached)
            # Labels need to be repeated for global views
            probe_logits = probe(emb_for_probe.detach())
            probe_loss = F.cross_entropy(probe_logits, labels_for_probe)

            # Combined loss
            loss = lejepa_loss + probe_loss

        # Compute Attention Metrics (Forward pass stats)
        # We need raw attention maps.
        # CAUTION: We didn't ask model to output attention in forward pass above.
        # Computing it now would require another forward or modifying model call.
        # Efficient way: Enable output_attention temporarily or just do it on a sample?
        # Doing it for every batch is expensive if model doesn't return it by default.
        # But user wants to track "over training".
        # Let's request attention maps on the Global crops forward pass if cheap,
        # OR just do it on the last batch of the interval to save time?
        # Let's do it every log_interval batch to avoid slowing down training too much.

        current_entropy = 0.0
        current_gini = 0.0
        current_sparsity = 0.0
        attn_rank = {"effective_rank": 0.0, "spectral_norm": 0.0}
        attn_struct = {
            "diag_mass": 0.0,
            "cls_to_patches": 0.0,
            "cls_self": 0.0,
            "patches_to_cls": 0.0,
            "head_entropy_std": 0.0,
        }
        attn_dist = {
            "patch_attn_distance_mean": 0.0,
            "patch_attn_distance_std": 0.0,
            "patch_local_mass_r1": 0.0,
        }
        attn_layer_metrics = {}

        if batch_idx % config.log_interval == 0:
            with torch.no_grad():
                # Enable attention output for this mini-check
                for blk in model.encoder.blocks:
                    blk.attn.output_attention = True
                # Encoder expects (B, C, H, W); flatten the view dimension.
                _ = model.encoder(global_crops_tensor.flatten(0, 1))  # (B*2, C, H, W)
                attns = model.encoder.get_attention_maps()  # List of (B, H, N, N)
                for blk in model.encoder.blocks:
                    blk.attn.output_attention = False

                if attns:
                    # Compute metrics on last layer
                    last_attn = attns[-1]
                    current_entropy = compute_entropy(last_attn).item()
                    current_gini = compute_gini(last_attn).item()
                    current_sparsity = compute_sparsity(last_attn).item()
                    attn_rank = compute_attention_rank(last_attn)
                    attn_struct = compute_attention_structure_metrics(last_attn)
                    attn_dist = compute_attention_distance_metrics(
                        last_attn,
                        grid_size=config.img_size // config.patch_size,
                        radius=1,
                    )

                    # Per-layer CLS sink / diagonal / head-entropy collapse
                    eps = 1e-8
                    for li, a in enumerate(attns):
                        m = compute_attention_structure_metrics(a)
                        for k, v in m.items():
                            attn_layer_metrics[f"attn_layer/{k}/l{li}"] = v

                        d = compute_attention_distance_metrics(
                            a, grid_size=config.img_size // config.patch_size, radius=1
                        )
                        attn_layer_metrics[
                            f"attn_layer/patch_attn_distance_mean/l{li}"
                        ] = d.get("patch_attn_distance_mean", 0.0)
                        attn_layer_metrics[f"attn_layer/patch_local_mass_r1/l{li}"] = (
                            d.get("patch_local_mass_r1", 0.0)
                        )

                        # Head entropy stats (collapse indicator)
                        head_ent = (
                            -(a * torch.log(a + eps)).sum(dim=-1).mean(dim=-1)
                        )  # (B,H)
                        attn_layer_metrics[f"attn_layer/head_entropy_mean/l{li}"] = (
                            head_ent.mean().item()
                        )
                        attn_layer_metrics[f"attn_layer/head_entropy_min/l{li}"] = (
                            head_ent.min(dim=1).values.mean().item()
                        )
                        attn_layer_metrics[f"attn_layer/head_entropy_max/l{li}"] = (
                            head_ent.max(dim=1).values.mean().item()
                        )
                        attn_layer_metrics[f"attn_layer/head_entropy_std/l{li}"] = (
                            head_ent.std(dim=1).mean().item()
                        )

        nonfinite_loss = float((~torch.isfinite(loss.detach())).item())

        # Backward pass
        if nonfinite_loss > 0.0:
            optimizer.zero_grad(set_to_none=True)
            if config.use_wandb:
                try:
                    import wandb

                    global_step = (epoch - 1) * len(train_loader) + batch_idx
                    wandb.log(
                        {"step": global_step, "sys/nonfinite_loss": nonfinite_loss},
                        commit=True,
                    )
                except Exception:
                    pass
            raise RuntimeError("Non-finite loss encountered during training.")

        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # Unscale grads on logging steps so any gradient-based metrics are comparable.
        # (On non-logging steps, GradScaler will unscale inside scaler.step()).
        opt_norms_model = None
        opt_norms_probe = None
        block_opt_metrics = {}
        if batch_idx % config.log_interval == 0 and scaler.is_enabled():
            scaler.unscale_(optimizer)

        if batch_idx % config.log_interval == 0:
            opt_norms_model = compute_global_norms(model)
            opt_norms_probe = compute_global_norms(probe)
            lr_cur = float(optimizer.param_groups[0].get("lr", 0.0))
            blk = compute_encoder_block_opt_stats(model, lr=lr_cur)
            if blk:
                depth = len(blk.get("param_norm", []))
                for li in range(depth):
                    block_opt_metrics[f"opt_block/param_norm/l{li}"] = blk[
                        "param_norm"
                    ][li]
                    block_opt_metrics[f"opt_block/grad_norm/l{li}"] = blk["grad_norm"][
                        li
                    ]
                    block_opt_metrics[f"opt_block/grad_to_param/l{li}"] = blk[
                        "grad_to_param"
                    ][li]
                    if "lr_scaled_grad_to_param" in blk:
                        block_opt_metrics[
                            f"opt_block/lr_scaled_grad_to_param/l{li}"
                        ] = blk["lr_scaled_grad_to_param"][li]
                    block_opt_metrics[f"opt_block/nonfinite_grad_params/l{li}"] = blk[
                        "nonfinite_grad_params"
                    ][li]
                block_opt_metrics["sys/nonfinite_grad_params"] = float(
                    sum(blk.get("nonfinite_grad_params", []))
                )

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Optional SWA/EMA teacher update (paper notes small ViT gains when using SWA for centers)
        if teacher_model is not None and config.teacher_student:
            ema_coeff = _cosine_ema_coeff(
                base=config.teacher_base_ema,
                final=config.teacher_final_ema,
                epoch=epoch,
                total_epochs=config.epochs,
            )
            _ema_update_(teacher_model, model, ema_coeff)

        # Capture Grad Stats from hooks
        # attn_grads is a list populated by hooks
        current_grad_norm = 0.0
        if attn_grads:
            # Average norm over layers
            norms = [g.norm().item() for g in attn_grads]
            current_grad_norm = sum(norms) / len(norms)
            if scaler.is_enabled():
                current_grad_norm /= float(scaler.get_scale())
            attn_grads.clear()  # Reset for next batch

        # Compute accuracy (on first global view only)
        with torch.no_grad():
            emb_view0 = emb_global.view(B, 2, -1)[:, 0]  # (B, D)
            logits_view0 = probe(emb_view0)
            pred = logits_view0.argmax(dim=1)
            correct = (pred == labels).sum().item()
            total_correct += correct
            total_samples += B
            batch_acc = 100.0 * correct / max(B, 1)

        # Representation health metrics from the already-computed global embeddings
        rep_stats = {}
        align_stats = {}
        cov_stats = {}
        if batch_idx % config.log_interval == 0:
            with torch.no_grad():
                z = emb_global.view(B, 2, -1).float()
                z_mean = z.mean(dim=1)
                rep_stats = compute_representation_stats(z_mean)
                align_stats = compute_alignment_metrics(z[:, 0], z[:, 1])
                cov_stats = compute_covariance_metrics(z_mean)

        # Accumulate losses
        total_loss += lejepa_loss.item()
        total_sigreg += loss_dict["sigreg_loss"].item()
        total_pred += loss_dict["prediction_loss"].item()
        total_probe_loss += probe_loss.item()
        if batch_idx % config.log_interval == 0:
            total_entropy += current_entropy
            total_gini += current_gini
            total_sparsity += current_sparsity
            total_grad_norm += current_grad_norm

        # Update progress bar and log to wandb
        if batch_idx % config.log_interval == 0:
            lrs = scheduler.get_last_lr()
            lr_encoder = lrs[0] if lrs else 0.0
            lr_probe = lrs[1] if len(lrs) > 1 else lr_encoder
            pbar.set_postfix(
                {
                    "loss": f"{lejepa_loss.item():.4f}",
                    "sigreg": f"{loss_dict['sigreg_loss'].item():.4f}",
                    "pred": f"{loss_dict['prediction_loss'].item():.4f}",
                    "acc": f"{100 * total_correct / total_samples:.1f}%",
                    "lr_e": f"{lr_encoder:.2e}",
                    "lr_p": f"{lr_probe:.2e}",
                }
            )
            # Log batch metrics to wandb
            iter_end = time.perf_counter()
            batch_time = iter_end - iter_start
            prev_iter_end = iter_end

            if config.use_wandb:
                import wandb

                global_step = (epoch - 1) * len(train_loader) + batch_idx
                num_views = 2 + len(local_crops)
                images_per_sec = (B * num_views) / max(batch_time, 1e-8)
                samples_per_sec = B / max(batch_time, 1e-8)
                amp_scale = float(scaler.get_scale()) if scaler.is_enabled() else 1.0

                sys_metrics = {
                    "sys/data_time_s": data_time,
                    "sys/batch_time_s": batch_time,
                    "sys/samples_per_sec": samples_per_sec,
                    "sys/images_per_sec": images_per_sec,
                }
                if device.type == "cuda":
                    sys_metrics.update(
                        {
                            "sys/cuda_mem_allocated_mb": torch.cuda.memory_allocated()
                            / 1024**2,
                            "sys/cuda_mem_reserved_mb": torch.cuda.memory_reserved()
                            / 1024**2,
                            "sys/cuda_max_mem_allocated_mb": torch.cuda.max_memory_allocated()
                            / 1024**2,
                        }
                    )

                wandb.log(
                    {
                        "step": global_step,
                        "train/total_loss": loss.item(),
                        "train/loss": lejepa_loss.item(),
                        "train/sigreg": loss_dict["sigreg_loss"].item(),
                        "train/prediction": loss_dict["prediction_loss"].item(),
                        "train/probe_loss": probe_loss.item(),
                        "train/accuracy": 100 * total_correct / total_samples,
                        "train/batch_accuracy": batch_acc,
                        "train/lr_encoder": lr_encoder,
                        "train/lr_probe": lr_probe,
                        "train/amp_scale": amp_scale,
                        "train/attn_entropy": current_entropy,
                        "train/attn_gini": current_gini,
                        "train/attn_sparsity": current_sparsity,
                        "train/attn_grad_norm": current_grad_norm,
                        "attn/last_layer_effective_rank": attn_rank.get(
                            "effective_rank", 0.0
                        ),
                        "attn/last_layer_spectral_norm": attn_rank.get(
                            "spectral_norm", 0.0
                        ),
                        "attn/diag_mass": attn_struct.get("diag_mass", 0.0),
                        "attn/cls_to_patches": attn_struct.get("cls_to_patches", 0.0),
                        "attn/cls_self": attn_struct.get("cls_self", 0.0),
                        "attn/patches_to_cls": attn_struct.get("patches_to_cls", 0.0),
                        "attn/head_entropy_std": attn_struct.get(
                            "head_entropy_std", 0.0
                        ),
                        "attn/patch_attn_distance_mean": attn_dist.get(
                            "patch_attn_distance_mean", 0.0
                        ),
                        "attn/patch_attn_distance_std": attn_dist.get(
                            "patch_attn_distance_std", 0.0
                        ),
                        "attn/patch_local_mass_r1": attn_dist.get(
                            "patch_local_mass_r1", 0.0
                        ),
                        **attn_layer_metrics,
                        "rep/norm_mean": rep_stats.get("norm_mean", 0.0),
                        "rep/norm_std": rep_stats.get("norm_std", 0.0),
                        "rep/variance": rep_stats.get("variance", 0.0),
                        "rep/effective_dim": rep_stats.get("effective_dim", 0.0),
                        "rep/isotropy": rep_stats.get("isotropy", 0.0),
                        "rep/align_cos": align_stats.get("cos", 0.0),
                        "rep/align_l2": align_stats.get("l2", 0.0),
                        "rep/cov_offdiag_l2": cov_stats.get("cov_offdiag_l2", 0.0),
                        "rep/cov_diag_mean": cov_stats.get("cov_diag_mean", 0.0),
                        "rep/cov_diag_min": cov_stats.get("cov_diag_min", 0.0),
                        "rep/var_mean": cov_stats.get("var_mean", 0.0),
                        "rep/var_min": cov_stats.get("var_min", 0.0),
                        "opt/encoder_param_norm": (
                            opt_norms_model.get("param_norm", 0.0)
                            if opt_norms_model
                            else 0.0
                        ),
                        "opt/encoder_grad_norm": (
                            opt_norms_model.get("grad_norm", 0.0)
                            if opt_norms_model
                            else 0.0
                        ),
                        "opt/encoder_grad_to_param": (
                            opt_norms_model.get("grad_to_param", 0.0)
                            if opt_norms_model
                            else 0.0
                        ),
                        "opt/probe_param_norm": (
                            opt_norms_probe.get("param_norm", 0.0)
                            if opt_norms_probe
                            else 0.0
                        ),
                        "opt/probe_grad_norm": (
                            opt_norms_probe.get("grad_norm", 0.0)
                            if opt_norms_probe
                            else 0.0
                        ),
                        "opt/probe_grad_to_param": (
                            opt_norms_probe.get("grad_to_param", 0.0)
                            if opt_norms_probe
                            else 0.0
                        ),
                        **sys_metrics,
                        "sys/nonfinite_loss": nonfinite_loss,
                        **block_opt_metrics,
                    }
                )

                # Attention Analysis Module - Batch-level entropy tracking
                if entropy_tracker is not None:
                    entropy_snap = entropy_tracker.step(
                        images=global_crops_tensor[:, 0],  # First global view
                        epoch=epoch,
                    )
                    if entropy_snap is not None:
                        entropy_metrics = {
                            "attn_analysis/entropy_mean": entropy_snap.mean_entropy,
                            "attn_analysis/entropy_std": entropy_snap.std_entropy,
                        }
                        # Per-layer entropy
                        for layer_idx, layer_ent in enumerate(
                            entropy_snap.layer_entropies
                        ):
                            entropy_metrics[
                                f"attn_analysis/entropy_layer_{layer_idx}"
                            ] = layer_ent
                        wandb.log(entropy_metrics, commit=False)
        else:
            prev_iter_end = time.perf_counter()

    num_batches = len(train_loader)
    return {
        "loss": total_loss / num_batches,
        "sigreg_loss": total_sigreg / num_batches,
        "prediction_loss": total_pred / num_batches,
        "probe_loss": total_probe_loss / num_batches,
        "accuracy": 100 * total_correct / total_samples,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    probe: LinearProbe,
    val_loader,
    device: torch.device,
    config: Config,
) -> dict:
    """Evaluate on validation set."""
    model.eval()
    probe.eval()

    total_correct = 0
    total_samples = 0
    total_top5 = 0
    total_loss = 0.0

    for views, labels in tqdm(val_loader, desc="Evaluating"):
        views = views.to(device, non_blocking=True)  # (B, 1, C, H, W)
        labels = labels.to(device, non_blocking=True)

        # Get embeddings
        with _autocast_ctx(device, enabled=config.mixed_precision):
            emb, _ = model(views)
            logits = probe(emb)

        loss = F.cross_entropy(logits, labels, reduction="sum")
        total_loss += loss.item()

        pred = logits.argmax(dim=1)
        total_correct += (pred == labels).sum().item()
        total_samples += labels.size(0)

        k = min(5, logits.shape[1])
        topk = logits.topk(k, dim=1).indices
        total_top5 += topk.eq(labels.unsqueeze(1)).any(dim=1).sum().item()

    accuracy = 100 * total_correct / max(total_samples, 1)
    top5 = 100 * total_top5 / max(total_samples, 1)
    avg_loss = total_loss / max(total_samples, 1)
    return {"val_accuracy": accuracy, "val_top5": top5, "val_loss": avg_loss}


@torch.no_grad()
def evaluate_knn(
    model: nn.Module,
    val_loader,
    device: torch.device,
    num_classes: int = 20,
    k: int = 20,
    temperature: float = 0.07,
    max_samples: int = 5000,
    mixed_precision: bool = True,
) -> dict:
    """
    kNN evaluation on embeddings (DINO-style weighted voting).

    Uses leave-one-out kNN within the validation set (up to max_samples).
    """
    model.eval()

    feats = []
    labels_all = []
    for views, labels in tqdm(val_loader, desc="Collecting kNN bank"):
        views = views.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with _autocast_ctx(device, enabled=mixed_precision):
            emb, _ = model(views)
        feats.append(emb.detach().float())
        labels_all.append(labels.detach())
        if sum(x.shape[0] for x in feats) >= max_samples:
            break

    if not feats:
        return {"knn_top1": 0.0, "knn_top5": 0.0}

    bank = torch.cat(feats, dim=0)[:max_samples]
    bank_labels = torch.cat(labels_all, dim=0)[:max_samples]
    bank = bank / (bank.norm(dim=1, keepdim=True) + 1e-8)

    n = bank.shape[0]
    if n < 2:
        return {"knn_top1": 0.0, "knn_top5": 0.0}

    k = min(k, n - 1)
    correct1 = 0
    correct5 = 0

    # Query in chunks to limit peak memory
    chunk = min(512, n)
    for start in tqdm(range(0, n, chunk), desc="kNN eval"):
        end = min(start + chunk, n)
        q = bank[start:end]  # (B,D)
        sims = q @ bank.T  # (B,N)

        # exclude self
        row = torch.arange(end - start, device=device)
        sims[row, start + row] = float("-inf")

        vals, idx = sims.topk(k, dim=1, largest=True, sorted=False)  # (B,k)
        nn_labels = bank_labels[idx]  # (B,k)

        # weighted vote
        weights = (vals / max(temperature, 1e-6)).exp()  # (B,k)
        scores = torch.zeros((end - start, num_classes), device=device)
        scores.scatter_add_(1, nn_labels, weights)

        pred = scores.argmax(dim=1)
        true = bank_labels[start:end]
        correct1 += (pred == true).sum().item()

        topk5 = scores.topk(min(5, num_classes), dim=1).indices
        correct5 += topk5.eq(true.unsqueeze(1)).any(dim=1).sum().item()

    return {
        "knn_top1": 100.0 * correct1 / n,
        "knn_top5": 100.0 * correct5 / n,
        "knn_samples": float(n),
    }


@torch.no_grad()
def evaluate_intrinsic_dim(
    model: nn.Module,
    val_loader,
    device: torch.device,
    max_samples: int = 1024,
    mixed_precision: bool = True,
) -> dict:
    """
    Estimate intrinsic dimension of embeddings (TwoNN) on a subset of val set.
    """
    model.eval()
    feats = []
    for views, _labels in tqdm(val_loader, desc="Collecting for LID"):
        views = views.to(device, non_blocking=True)
        with _autocast_ctx(device, enabled=mixed_precision):
            emb, _ = model(views)
        feats.append(emb.detach().float())
        if sum(x.shape[0] for x in feats) >= max_samples:
            break
    if not feats:
        return {"intrinsic_dim_twonn": 0.0}
    z = torch.cat(feats, dim=0)[:max_samples]
    return {"intrinsic_dim_twonn": estimate_intrinsic_dim_twonn(z)}


def _slice_crops(crops: list[torch.Tensor], start: int, end: int) -> list[torch.Tensor]:
    return [c[start:end] for c in crops]


def _compute_training_loss_from_crops(
    model: nn.Module,
    probe: LinearProbe,
    loss_fn: LeJEPALoss,
    crops: list[torch.Tensor],
    labels: torch.Tensor,
    device: torch.device,
    mixed_precision: bool,
) -> tuple[torch.Tensor, dict]:
    """
    Compute the same combined loss used during training on a batch of crops.
    Returns (loss, aux_dict).
    """
    crops = [c.to(device, non_blocking=True) for c in crops]
    labels = labels.to(device, non_blocking=True)

    global_crops = crops[:2]
    local_crops = crops[2:]
    _B = labels.shape[0]  # noqa: F841 (unused, kept for reference)

    with _autocast_ctx(device, enabled=mixed_precision):
        global_crops_tensor = torch.stack(global_crops, dim=1)  # (B,2,C,H,W)
        emb_global, proj_global = model(global_crops_tensor)

        if local_crops:
            local_crops_tensor = torch.stack(local_crops, dim=1)
            _emb_local, proj_local = model(local_crops_tensor)
            proj = torch.cat([proj_global, proj_local], dim=0)
        else:
            proj = proj_global

        loss_dict = loss_fn(proj)
        lejepa_loss = loss_dict["loss"]

        probe_logits = probe(emb_global.detach())
        probe_loss = F.cross_entropy(probe_logits, labels.repeat_interleave(2))
        total_loss = lejepa_loss + probe_loss

    return total_loss, {
        "lejepa_loss": lejepa_loss.detach(),
        "probe_loss": probe_loss.detach(),
        "sigreg_loss": loss_dict.get("sigreg_loss", torch.tensor(0.0)).detach(),
        "prediction_loss": loss_dict.get("prediction_loss", torch.tensor(0.0)).detach(),
    }


class LeJEPALossMetric(loss_landscapes.metrics.Metric):
    """Custom metric for loss-landscapes library to evaluate LeJEPA loss."""

    def __init__(
        self,
        probe: LinearProbe,
        loss_fn: LeJEPALoss,
        crops: list[torch.Tensor],
        labels: torch.Tensor,
        device: torch.device,
        mixed_precision: bool,
    ):
        super().__init__()
        self.probe = probe
        self.loss_fn = loss_fn
        self.crops = crops
        self.labels = labels
        self.device = device
        self.mixed_precision = mixed_precision

    def __call__(self, model_wrapper) -> float:
        """Evaluate loss at current parameter point."""
        model = model_wrapper.get_model()
        # loss_landscapes deep-copies model to CPU; move it back to our device
        model = model.to(self.device)
        # Ensure loss function and its buffers are on the same device
        loss_fn = self.loss_fn.to(self.device)
        # Ensure other components are also on the same device
        probe = self.probe.to(self.device)
        crops_dev = [c.to(self.device) for c in self.crops]
        labels_dev = self.labels.to(self.device)
        
        with torch.no_grad():
            loss, _ = _compute_training_loss_from_crops(
                model,
                probe,
                loss_fn,
                crops_dev,
                labels_dev,
                self.device,
                self.mixed_precision,
            )
        return float(loss)


def _select_diag_params(model: nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    """
    Pick a manageable but informative parameter subset for expensive diagnostics.
    Focus on normalization params (LayerNorm/RMSNorm/q_norm/k_norm) + final norms.
    """
    selected = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        if any(
            k in lname for k in ["norm", "rmsnorm", "layernorm", "q_norm", "k_norm"]
        ):
            selected.append((name, p))
    return selected


def _flatten_grads(params: list[torch.nn.Parameter]) -> torch.Tensor:
    vecs = []
    for p in params:
        if p.grad is None:
            continue
        vecs.append(p.grad.detach().float().reshape(-1))
    if not vecs:
        return torch.zeros(1)
    return torch.cat(vecs, dim=0)


def _gns_microbatch(
    model: nn.Module,
    probe: LinearProbe,
    loss_fn: LeJEPALoss,
    crops: list[torch.Tensor],
    labels: torch.Tensor,
    device: torch.device,
    mixed_precision: bool,
    microbatches: int,
) -> dict:
    """
    Microbatch gradient noise proxy on a parameter subset.
    Computes noise-to-signal ratio and cosine agreement across microbatches.
    """
    named = _select_diag_params(model)
    params = [p for _n, p in named]
    if not params:
        return {"gns/noise_to_signal": 0.0, "gns/cos_to_mean": 0.0}

    B = labels.shape[0]
    mb = max(1, min(microbatches, B))
    splits = torch.linspace(0, B, steps=mb + 1).long().tolist()

    grads = []
    norms = []
    for i in range(mb):
        s, e = splits[i], splits[i + 1]
        if e - s < 1:
            continue
        model.zero_grad(set_to_none=True)
        probe.zero_grad(set_to_none=True)
        loss, _aux = _compute_training_loss_from_crops(
            model,
            probe,
            loss_fn,
            _slice_crops(crops, s, e),
            labels[s:e],
            device,
            mixed_precision,
        )
        loss.backward()
        g = _flatten_grads(params)
        grads.append(g)
        norms.append(g.norm().item())

    if len(grads) < 2:
        return {"gns/noise_to_signal": 0.0, "gns/cos_to_mean": 0.0}

    G = torch.stack(grads, dim=0)  # (M, P)
    g_bar = G.mean(dim=0)
    mean_sq = (G.pow(2).sum(dim=1)).mean()
    sq_mean = g_bar.pow(2).sum()
    noise = (mean_sq - sq_mean).clamp(min=0.0)
    n2s = (noise / (sq_mean + 1e-12)).item()

    gbar_n = g_bar / (g_bar.norm() + 1e-12)
    cos = (G / (G.norm(dim=1, keepdim=True) + 1e-12)) @ gbar_n
    cos_mean = cos.mean().item()
    cos_std = cos.std().item()

    return {
        "gns/noise_to_signal": n2s,
        "gns/cos_to_mean": cos_mean,
        "gns/cos_to_mean_std": cos_std,
        "gns/micro_grad_norm_mean": float(sum(norms) / len(norms)),
        "gns/micro_grad_norm_std": float(torch.tensor(norms).std().item()),
        "gns/num_params": float(G.shape[1]),
        "gns/microbatches": float(len(grads)),
    }


def _hessian_top_eig_power_iter(
    model: nn.Module,
    probe: LinearProbe,
    loss_fn: LeJEPALoss,
    crops: list[torch.Tensor],
    labels: torch.Tensor,
    device: torch.device,
    mixed_precision: bool,
    iters: int,
) -> dict:
    """
    Estimate top Hessian eigenvalue via power iteration on a parameter subset.
    """
    with torch.enable_grad():
        named = _select_diag_params(model)
        params = [p for _n, p in named]
        if not params:
            return {"sharpness/top_hessian_eig": 0.0}

        model.zero_grad(set_to_none=True)
        probe.zero_grad(set_to_none=True)

        loss, _aux = _compute_training_loss_from_crops(
            model, probe, loss_fn, crops, labels, device, mixed_precision
        )

        grads = torch.autograd.grad(
            loss, params, create_graph=True, retain_graph=True, allow_unused=True
        )
        vec = [torch.randn_like(p, dtype=torch.float32) for p in params]

        def _normalize(vs: list[torch.Tensor]) -> list[torch.Tensor]:
            nrm = torch.sqrt(
                sum([(v.detach().float().pow(2).sum()) for v in vs]) + 1e-12
            )
            return [v / nrm for v in vs]

        vec = _normalize(vec)

        lam = 0.0
        for _ in range(max(1, iters)):
            dot = 0.0
            for g, v in zip(grads, vec):
                if g is None:
                    continue
                # NOTE: do NOT detach `g` here; we need its graph for Hessian-vector products.
                dot = dot + (g * v.to(device=g.device, dtype=g.dtype)).sum()
            hv = torch.autograd.grad(dot, params, retain_graph=True, allow_unused=True)
            hv = [
                torch.zeros_like(p, dtype=torch.float32)
                if h is None
                else h.detach().float()
                for p, h in zip(params, hv)
            ]

            # Rayleigh quotient
            num = 0.0
            den = 0.0
            for v, h in zip(vec, hv):
                num = num + (v.detach().float() * h).sum()
                den = den + (v.detach().float().pow(2).sum())
            lam = (num / (den + 1e-12)).item()
            vec = _normalize(hv)

        return {
            "sharpness/top_hessian_eig": float(lam),
            "sharpness/params_tracked": float(len(params)),
        }


def _loss_landscape_slice(
    model: nn.Module,
    probe: LinearProbe,
    loss_fn: LeJEPALoss,
    crops: list[torch.Tensor],
    labels: torch.Tensor,
    device: torch.device,
    mixed_precision: bool,
    radius: float,
    points: int,
) -> dict:
    """
    Evaluate loss along two 1D slices in parameter space:
      - gradient direction
      - random direction
    Uses the same training loss on the provided batch.
    """
    named = _select_diag_params(model)
    params = [p for _n, p in named]
    if not params:
        return {"alphas": [], "losses_grad": [], "losses_rand": []}

    # Compute base loss and gradient direction (on subset) with grad enabled.
    with torch.enable_grad():
        model.zero_grad(set_to_none=True)
        probe.zero_grad(set_to_none=True)

        base_loss, _aux = _compute_training_loss_from_crops(
            model, probe, loss_fn, crops, labels, device, mixed_precision
        )
        grads = torch.autograd.grad(
            base_loss, params, retain_graph=False, allow_unused=True
        )
        gdir = [
            torch.zeros_like(p) if g is None else g.detach().float()
            for p, g in zip(params, grads)
        ]

    def _norm(vs: list[torch.Tensor]) -> torch.Tensor:
        return torch.sqrt(sum([v.float().pow(2).sum() for v in vs]) + 1e-12)

    gnorm = _norm(gdir).item()
    if gnorm > 0:
        gdir = [v / gnorm for v in gdir]

    rdir = [torch.randn_like(p, dtype=torch.float32) for p in params]
    rnorm = _norm(rdir).item()
    rdir = [v / (rnorm + 1e-12) for v in rdir]

    # Scale by radius * param_norm to be somewhat comparable across runs
    pnorm = _norm([p.detach().float() for p in params]).item()
    scale = float(radius) * float(pnorm)

    alphas = torch.linspace(-1.0, 1.0, steps=max(3, points)).tolist()
    losses_grad = []
    losses_rand = []

    # Save originals
    originals = [p.detach().clone() for p in params]

    def _set_params(direction: list[torch.Tensor], alpha: float):
        for p, p0, d in zip(params, originals, direction):
            p.copy_(p0 + (alpha * scale) * d.to(p.device, dtype=p.dtype))

    with torch.no_grad():
        for a in alphas:
            _set_params(gdir, float(a))
            loss_val, _ = _compute_training_loss_from_crops(
                model, probe, loss_fn, crops, labels, device, mixed_precision
            )
            losses_grad.append(float(loss_val.detach().item()))

        for a in alphas:
            _set_params(rdir, float(a))
            loss_val, _ = _compute_training_loss_from_crops(
                model, probe, loss_fn, crops, labels, device, mixed_precision
            )
            losses_rand.append(float(loss_val.detach().item()))

    # Restore
    with torch.no_grad():
        for p, p0 in zip(params, originals):
            p.copy_(p0)

    return {"alphas": alphas, "losses_grad": losses_grad, "losses_rand": losses_rand}


@torch.no_grad()
def _head_ablation_sensitivity(
    model: nn.Module,
    probe: LinearProbe,
    loss_fn: LeJEPALoss,
    crops: list[torch.Tensor],
    labels: torch.Tensor,
    device: torch.device,
    mixed_precision: bool,
    last_layers: int,
) -> dict:
    """
    Compute Δloss when ablating individual attention heads.

    Returns a (L,H) matrix over the last `last_layers` encoder blocks.
    """
    if last_layers <= 0:
        return {"base_loss": 0.0, "layers": [], "delta_loss": []}

    base_loss, _ = _compute_training_loss_from_crops(
        model, probe, loss_fn, crops, labels, device, mixed_precision
    )
    base = float(base_loss.detach().item())

    blocks = list(getattr(model.encoder, "blocks", []))
    if not blocks:
        return {"base_loss": base, "layers": [], "delta_loss": []}

    num_layers = len(blocks)
    start = max(0, num_layers - last_layers)
    layer_ids = list(range(start, num_layers))

    # infer num_heads from the block's attention module
    num_heads = int(getattr(blocks[-1].attn, "num_heads", 0))
    if num_heads <= 0:
        return {"base_loss": base, "layers": layer_ids, "delta_loss": []}

    delta: list[list[float]] = []

    for li in layer_ids:
        attn = blocks[li].attn
        orig_mask = getattr(attn, "head_mask", None)
        row: list[float] = []
        for hi in range(num_heads):
            m = torch.ones(num_heads, device=device, dtype=torch.float32)
            m[hi] = 0.0
            attn.head_mask = m
            loss_val, _ = _compute_training_loss_from_crops(
                model, probe, loss_fn, crops, labels, device, mixed_precision
            )
            row.append(float(loss_val.detach().item()) - base)
        attn.head_mask = orig_mask
        delta.append(row)

    return {"base_loss": base, "layers": layer_ids, "delta_loss": delta}


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description="Train LeJEPA with JiT or ViT")
    parser.add_argument("--encoder", type=str, default="jit", choices=["jit", "vit"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr_probe", type=float, default=None)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--lambda_sigreg", type=float, default=None)
    parser.add_argument("--num_views", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true", default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument(
        "--wandb_mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default=None,
    )
    parser.add_argument("--attn_distance_headmap_interval", type=int, default=None)
    parser.add_argument("--attn_logits_interval", type=int, default=None)
    parser.add_argument("--mlp_output_stats_interval", type=int, default=None)
    parser.add_argument("--head_ablation_interval", type=int, default=None)
    parser.add_argument("--head_ablation_layers", type=int, default=None)
    parser.add_argument("--landscape2d_interval", type=int, default=None)
    parser.add_argument("--landscape2d_radius", type=float, default=None)
    parser.add_argument("--landscape2d_points", type=int, default=None)
    # Quick verification / testing
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Limit training samples (for quick verification)",
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=None,
        help="Limit validation samples (for quick verification)",
    )
    args = parser.parse_args()

    # Create config with overrides
    overrides = {
        k: v for k, v in vars(args).items() if v is not None and k != "no_wandb"
    }
    if args.no_wandb:
        overrides["use_wandb"] = False
    if "lr" in overrides:
        overrides["lr_encoder"] = overrides.pop("lr")
    config = get_config(**overrides)

    # Set seed
    torch.manual_seed(config.seed)

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = (
        Path(config.output_dir) / f"{config.encoder}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Initialize wandb
    if config.use_wandb:
        try:
            import wandb

            if config.wandb_mode == "disabled":
                config.use_wandb = False
            if os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
                config.use_wandb = False

            # Prefer explicit entity (CLI/config), otherwise fall back to env.
            if not config.wandb_entity:
                env_entity = os.environ.get("WANDB_ENTITY")
                if env_entity:
                    config.wandb_entity = env_entity

            run_name = f"{config.encoder}-{time.strftime('%Y%m%d_%H%M%S')}"
            init_kwargs = {
                "project": config.wandb_project,
                "config": vars(config),
                "name": run_name,
            }
            if config.wandb_entity:
                init_kwargs["entity"] = config.wandb_entity

            if config.use_wandb:
                try:
                    if config.wandb_mode == "offline":
                        wandb.init(**init_kwargs, mode="offline")
                    else:
                        wandb.init(**init_kwargs)
                except Exception as e:
                    msg = (
                        "wandb.init failed. This is usually due to an invalid/expired API key "
                        "or missing permissions to write to the selected entity/project.\n"
                        f"  entity={init_kwargs.get('entity', '<default>')}\n"
                        f"  project={init_kwargs.get('project')}\n"
                        "Fix:\n"
                        "  - Run `wandb login --relogin` with a key that has access, OR\n"
                        "  - Pass the correct entity via `--wandb_entity <your_entity>`.\n"
                        "If you want to proceed without logging, add `--no_wandb`."
                    )
                    raise RuntimeError(msg) from e

            if config.use_wandb:
                # Define x-axes for different metric types
                wandb.define_metric("step")
                wandb.define_metric("epoch")
                wandb.define_metric("train/*", step_metric="step")
                wandb.define_metric("opt/*", step_metric="step")
                wandb.define_metric("opt_block/*", step_metric="step")
                wandb.define_metric("rep/*", step_metric="step")
                wandb.define_metric("sys/*", step_metric="step")
                wandb.define_metric("attn/*", step_metric="step")
                wandb.define_metric("attn_layer/*", step_metric="step")
                wandb.define_metric("epoch_*", step_metric="epoch")
                wandb.define_metric("val_*", step_metric="epoch")
                wandb.define_metric("attention/*", step_metric="epoch")
                wandb.define_metric("collapse/*", step_metric="epoch")
                wandb.define_metric("epoch_attn_logit/*", step_metric="epoch")
                wandb.define_metric("epoch_mlp/*", step_metric="epoch")
                wandb.define_metric("epoch_rep/*", step_metric="epoch")
                wandb.define_metric("knn/*", step_metric="epoch")
                wandb.define_metric("lid/*", step_metric="epoch")
                wandb.define_metric("epoch_attn_layer/*", step_metric="epoch")
                wandb.define_metric("epoch_block/*", step_metric="epoch")
                wandb.define_metric("epoch_opt/*", step_metric="epoch")
                wandb.define_metric("gns/*", step_metric="epoch")
                wandb.define_metric("sharpness/*", step_metric="epoch")
                # Attention Analysis Module metrics
                wandb.define_metric("attribution/*", step_metric="epoch")
                wandb.define_metric("faithfulness/*", step_metric="epoch")
                wandb.define_metric("entropy/*", step_metric="step")
                # Production Dashboard metrics
                wandb.define_metric("summary/*", step_metric="epoch")
                wandb.define_metric("health/*", step_metric="epoch")
                wandb.define_metric("arch/*", step_metric="epoch")
        except ImportError:
            print("wandb not installed, skipping logging")
            config.use_wandb = False

    # Create dataloaders
    print("Loading data (ImageNette + ImageWoof)...")
    train_loader, val_loader_nette, val_loader_woof = get_dataloaders(
        batch_size=config.batch_size,
        img_size=config.img_size,
        num_workers=config.num_workers,
        local_crops_number=config.local_crops_number,
        local_crops_size=config.local_crops_size,
        local_crops_scale=config.local_crops_scale,
        global_crops_scale=config.global_crops_scale,
        max_train_samples=getattr(config, "max_train_samples", None),
        max_val_samples=getattr(config, "max_val_samples", None),
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples (ImageNette): {len(val_loader_nette.dataset)}")
    print(f"Val samples (ImageWoof): {len(val_loader_woof.dataset)}")

    # Create model
    print(f"Creating LeJEPA model with {config.encoder.upper()} encoder...")
    jit_kwargs = {}
    if config.encoder == "jit":
        jit_kwargs["bottleneck_dim"] = config.bottleneck_dim

    model = create_lejepa(
        encoder_type=config.encoder,
        img_size=config.img_size,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        proj_hidden_dim=config.proj_hidden_dim,
        proj_dim=config.proj_dim,
        **jit_kwargs,
    ).to(device)

    teacher_model = None
    if config.teacher_student:
        # Paper notes that using SWA/teacher targets for the global-view centers can help ViTs.
        # Keep optional since it doubles model memory.
        teacher_model = copy.deepcopy(model).to(device)
        teacher_model.requires_grad_(False)
        teacher_model.eval()

    # Create linear probe
    # Online probe monitors training convergence using standard embeddings (last layer)
    # Full evaluation uses concatenated features (last 2 layers)
    # We update online probe to also use concatenated features to match paper protocol
    # Combined ImageNette (10) + ImageWoof (10) = 20 classes
    probe = LinearProbe(config.embed_dim * 2, num_classes=20).to(device)

    # Print model info
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Encoder parameters: {count_parameters(model.encoder):,}")
    print(f"Projector parameters: {count_parameters(model.proj):,}")
    print(f"Probe parameters: {count_parameters(probe):,}")

    # Create loss function
    loss_fn = LeJEPALoss(
        lambda_sigreg=config.lambda_sigreg,
        knots=config.sigreg_num_knots,
        t_max=config.sigreg_max_t,
        num_slices=config.sigreg_num_slices,
        clip_value=config.sigreg_clip_value,
        multivariate=config.sigreg_multivariate,
        num_frequencies=config.sigreg_num_frequencies,
        sigma=config.sigreg_sigma,
        num_global_views=config.num_global_views,
    )

    # Create optimizers with separate weight decay (matching reference)
    param_groups = [
        {
            "params": model.parameters(),
            "lr": config.lr_encoder,
            "weight_decay": config.weight_decay_encoder,
        },
        {
            "params": probe.parameters(),
            "lr": config.lr_probe,
            "weight_decay": config.weight_decay_probe,
        },
    ]
    optimizer = torch.optim.AdamW(param_groups)

    # Create scheduler
    total_steps = len(train_loader) * config.epochs
    warmup_steps = min(
        len(train_loader) * config.warmup_epochs,
        max(total_steps - 1, 1),
    )
    scheduler = get_schedulers(optimizer, warmup_steps, total_steps)

    # Create gradient scaler for mixed precision
    amp_dtype = _amp_dtype_for_device(device)
    use_grad_scaler = (
        config.mixed_precision and device.type == "cuda" and amp_dtype == torch.float16
    )
    scaler = GradScaler("cuda", enabled=use_grad_scaler)

    # Training loop
    print("\nStarting training...")
    print(
        f"Config: epochs={config.epochs}, batch_size={config.batch_size}, "
        f"num_views={config.num_views}, lambda={config.lambda_sigreg}"
    )
    best_val_acc = 0

    # Fixed batch for visualization
    vis_images_nette = None
    vis_images_woof = None
    vis_frames_nette = []
    vis_frames_woof = []
    attn_frames_nette = []
    attn_frames_woof = []

    # History tracking for SIGReg-aligned visualizations
    isotropy_history = []
    effective_rank_history = []
    uniformity_history = []
    sigreg_loss_history = []
    invariance_loss_history = []
    total_loss_history = []
    accuracy_history = []

    # Get a batch from val_loader_nette
    try:
        iter_nette = iter(val_loader_nette)
        views, _ = next(iter_nette)
        vis_images_nette = views[: config.diagnostic_batch_size, 0].to(device)
    except StopIteration:
        print("Warning: ImageNette Validation loader is empty.")

    # Get a batch from val_loader_woof
    try:
        iter_woof = iter(val_loader_woof)
        views, _ = next(iter_woof)
        vis_images_woof = views[: config.diagnostic_batch_size, 0].to(device)
    except StopIteration:
        print("Warning: ImageWoof Validation loader is empty.")

    # Import visualization utils
    from utils.visualization import (
        generate_attention_entropy_per_head_heatmap_from_maps,
        generate_pca_visualization,
        generate_attention_rollout,
        generate_layer_attention_evolution,
        generate_per_head_attention,
        generate_attention_distance_per_head_heatmap_from_maps,
        generate_token_similarity_heatmap,
        generate_rsm_across_layers,
        generate_gradient_flow_heatmap,
        generate_embedding_pca_scatter,
        generate_embedding_projection,
        generate_collapse_monitor,
        generate_layerwise_curves,
        generate_xy_curves,
        generate_loss_landscape_slice,
        generate_loss_landscape_2d,
        generate_loss_landscape_contour,
        generate_loss_landscape_3d,
        generate_loss_landscape_3d_with_contour,
        generate_loss_landscape_plotly,
        generate_head_ablation_heatmap,
        generate_pocp_per_head_heatmaps,
        AttentionTracker,
        # SIGReg-aligned visualizations
        generate_isotropy_evolution_plot,
        generate_loss_accuracy_correlation_plot,
        generate_embedding_distribution_plot,
        generate_sigreg_loss_components_plot,
    )

    # ==========================================================================
    # Initialize Attention Analysis Module
    # ==========================================================================
    entropy_tracker = None
    head_tracker = None
    attribution_probe_images = None
    attribution_probe_labels = None

    if config.attention_analysis_enabled:
        print("\nInitializing Attention Analysis Module...")

        # Get probe images for consistent attribution tracking
        try:
            iter_probe = iter(val_loader_nette)
            probe_batch, probe_labels_batch = next(iter_probe)
            attribution_probe_images = probe_batch[
                : config.attribution_probe_size, 0
            ].to(device)
            attribution_probe_labels = probe_labels_batch[
                : config.attribution_probe_size
            ].to(device)
            print(f"  - Probe images: {attribution_probe_images.shape}")
        except StopIteration:
            print("  - Warning: Could not get probe images for attribution")

        # Initialize entropy tracker
        if config.entropy_tracking_enabled:
            entropy_tracker = AttentionEntropyTracker(
                model,
                log_interval=config.entropy_log_interval,
                probe_images=attribution_probe_images,
            )
            print(
                f"  - Entropy tracker: log every {config.entropy_log_interval} batches"
            )

        # Initialize head specialization tracker
        if config.head_specialization_enabled:
            head_tracker = HeadSpecializationTracker(
                model,
                probe_images=attribution_probe_images,
                probe_labels=attribution_probe_labels,
            )
            print(
                f"  - Head tracker: analyze every {config.head_specialization_interval} epochs"
            )

        print(f"  - Attribution methods: {config.attribution_methods}")
        print(f"  - Attribution vis interval: {config.attribution_vis_interval} epochs")
        if config.faithfulness_eval_enabled:
            print(
                f"  - Faithfulness eval: every {config.faithfulness_eval_interval} epochs"
            )

    # Register Backward Hooks for Gradient Flow
    attn_grads = []

    def grad_hook(grad):
        if grad is not None:
            attn_grads.append(grad)

    # Hook into attention map gradients
    for blk in model.encoder.blocks:
        blk.attn.attn_drop.register_full_backward_hook(lambda m, i, o: grad_hook(o[0]))

    # Initialize attention tracker for epoch-to-epoch comparison
    attn_tracker = (
        AttentionTracker(max_snapshots=10)
        if int(getattr(config, "attention_difference_interval", 0)) > 0
        else None
    )

    prev_rep_nette = None
    prev_rep_woof = None
    prev_block_reps_nette = None
    prev_block_reps_woof = None

    # Initialize metric history for dashboard generation
    metric_history: dict[str, list[float]] = {
        "val_accuracy_nette": [],
        "val_accuracy_woof": [],
        "knn/nette_top1": [],
        "knn/woof_top1": [],
        "train/loss": [],
        "train/sigreg": [],
        "train/prediction": [],
        "epoch_attn/entropy": [],
        "rep/effective_dim": [],
        "rep/isotropy": [],
    }
    prev_epoch_metrics: dict[str, float] = {}

    def _every(ep: int, interval: int) -> bool:
        """Return True if a diagnostic should run this epoch (interval<=0 disables)."""
        return int(interval) > 0 and (ep % int(interval) == 0)

    def _every_or_first(ep: int, interval: int) -> bool:
        """Like _every(), but always run on epoch 1 if enabled (interval>0)."""
        return int(interval) > 0 and (ep == 1 or ep % int(interval) == 0)

    for epoch in range(1, config.epochs + 1):
        start_time = time.time()

        # Train
        train_metrics = train_one_epoch(
            model=model,
            teacher_model=teacher_model,
            probe=probe,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            device=device,
            epoch=epoch,
            config=config,
            scaler=scaler,
            attn_grads=attn_grads,
            entropy_tracker=entropy_tracker,
        )

        # Evaluate
        val_metrics_nette = {}
        val_metrics_woof = {}
        if epoch % config.eval_interval == 0:
            # Eval ImageNette
            print("Evaluating on ImageNette...")
            val_metrics_nette = evaluate(model, probe, val_loader_nette, device, config)
            acc_nette = val_metrics_nette["val_accuracy"]

            # Eval ImageWoof
            print("Evaluating on ImageWoof...")
            val_metrics_woof = evaluate(model, probe, val_loader_woof, device, config)
            acc_woof = val_metrics_woof["val_accuracy"]

            # Track best based on ImageNette (Easy) or Average?
            # Let's track based on Average to be fair, or just ImageNette since user said "observe"
            # I'll update best_val_acc = acc_nette for compatibility with saving "best_model.pt" logic
            # But let's log both.

            if acc_nette > best_val_acc:
                best_val_acc = acc_nette
                # Save best model
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "probe_state_dict": probe.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_accuracy": acc_nette,
                        "val_accuracy_woof": acc_woof,
                        "val_top5": val_metrics_nette.get("val_top5", 0.0),
                        "val_top5_woof": val_metrics_woof.get("val_top5", 0.0),
                        "val_loss": val_metrics_nette.get("val_loss", 0.0),
                        "val_loss_woof": val_metrics_woof.get("val_loss", 0.0),
                        "config": vars(config),
                    },
                    output_dir / "best_model.pt",
                )

        knn_metrics_nette = {}
        knn_metrics_woof = {}
        lid_metrics_nette = {}
        lid_metrics_woof = {}
        if config.use_wandb and _every_or_first(epoch, config.knn_interval):
            print("kNN eval on ImageNette embeddings...")
            knn_metrics_nette = evaluate_knn(
                model,
                val_loader_nette,
                device,
                num_classes=20,
                k=config.knn_k,
                temperature=config.knn_temperature,
                max_samples=config.knn_max_samples,
                mixed_precision=config.mixed_precision,
            )
            print("kNN eval on ImageWoof embeddings...")
            knn_metrics_woof = evaluate_knn(
                model,
                val_loader_woof,
                device,
                num_classes=20,
                k=config.knn_k,
                temperature=config.knn_temperature,
                max_samples=config.knn_max_samples,
                mixed_precision=config.mixed_precision,
            )

        if config.use_wandb and _every_or_first(epoch, config.lid_interval):
            print("Intrinsic dimension (TwoNN) on ImageNette embeddings...")
            lid_metrics_nette = evaluate_intrinsic_dim(
                model,
                val_loader_nette,
                device,
                max_samples=config.lid_max_samples,
                mixed_precision=config.mixed_precision,
            )
            print("Intrinsic dimension (TwoNN) on ImageWoof embeddings...")
            lid_metrics_woof = evaluate_intrinsic_dim(
                model,
                val_loader_woof,
                device,
                max_samples=config.lid_max_samples,
                mixed_precision=config.mixed_precision,
            )

        # Log metrics
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch}/{config.epochs}")
        print(
            f"  Loss: {train_metrics['loss']:.4f} (SIGReg: {train_metrics['sigreg_loss']:.4f}, Pred: {train_metrics['prediction_loss']:.4f})"
        )
        print(f"  Train Acc: {train_metrics['accuracy']:.2f}%")
        if val_metrics_nette:
            print(
                f"  Val Acc (Nette): {val_metrics_nette['val_accuracy']:.2f}% (Top-5: {val_metrics_nette.get('val_top5', 0):.2f}%, Loss: {val_metrics_nette.get('val_loss', 0):.4f}, Best: {best_val_acc:.2f}%)"
            )
            print(
                f"  Val Acc (Woof): {val_metrics_woof['val_accuracy']:.2f}% (Top-5: {val_metrics_woof.get('val_top5', 0):.2f}%, Loss: {val_metrics_woof.get('val_loss', 0):.4f})"
            )
        print(f"  Time: {epoch_time:.1f}s")

        # Track histories for SIGReg-aligned visualizations
        sigreg_loss_history.append(train_metrics["sigreg_loss"])
        invariance_loss_history.append(train_metrics["prediction_loss"])
        total_loss_history.append(train_metrics["loss"])
        if val_metrics_nette:
            accuracy_history.append(val_metrics_nette["val_accuracy"])

        if config.use_wandb:
            import wandb

            wandb.log(
                {
                    "epoch": epoch,
                    "epoch_loss": train_metrics["loss"],
                    "epoch_sigreg_loss": train_metrics["sigreg_loss"],
                    "epoch_prediction_loss": train_metrics["prediction_loss"],
                    "epoch_probe_loss": train_metrics["probe_loss"],
                    "epoch_train_accuracy": train_metrics["accuracy"],
                    "val_accuracy_nette": val_metrics_nette.get("val_accuracy", 0),
                    "val_accuracy_woof": val_metrics_woof.get("val_accuracy", 0),
                    "val_top5_nette": val_metrics_nette.get("val_top5", 0),
                    "val_top5_woof": val_metrics_woof.get("val_top5", 0),
                    "val_loss_nette": val_metrics_nette.get("val_loss", 0),
                    "val_loss_woof": val_metrics_woof.get("val_loss", 0),
                    "knn/nette_top1": knn_metrics_nette.get("knn_top1", 0),
                    "knn/nette_top5": knn_metrics_nette.get("knn_top5", 0),
                    "knn/woof_top1": knn_metrics_woof.get("knn_top1", 0),
                    "knn/woof_top5": knn_metrics_woof.get("knn_top5", 0),
                    "lid/nette_twonn": lid_metrics_nette.get("intrinsic_dim_twonn", 0),
                    "lid/woof_twonn": lid_metrics_woof.get("intrinsic_dim_twonn", 0),
                    "best_val_accuracy": best_val_acc,
                }
            )

        # Generate Visualizations
        if vis_images_nette is not None or vis_images_woof is not None:
            try:
                # Helper for GIF logging
                def log_gif(frames, key, tag):
                    if not frames:
                        return
                    import tempfile

                    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
                        gif_path = f.name
                    frames[0].save(
                        gif_path,
                        save_all=True,
                        append_images=frames[1:],
                        optimize=False,
                        duration=500,
                        loop=0,
                    )
                    wandb.log(
                        {
                            f"pca_progression_{tag}": wandb.Video(
                                gif_path, fps=2, format="gif"
                            )
                        },
                        commit=False,
                    )

                # === Core qualitative visuals (interval-gated) ===
                # 1. ImageNette: Patch PCA visualization
                if vis_images_nette is not None and _every_or_first(
                    epoch, config.pca_vis_interval
                ):
                    vis_grid_nette = generate_pca_visualization(
                        model,
                        vis_images_nette,
                        device,
                        img_size=config.pca_vis_size,
                        patch_size=config.patch_size,
                        pca_resample=config.pca_resample,
                        per_image_pca=config.pca_per_image,
                    )
                    vis_frames_nette.append(vis_grid_nette)
                    if config.use_wandb:
                        wandb.log(
                            {
                                "vis/nette_patch_pca": wandb.Image(
                                    vis_grid_nette, caption=f"Epoch {epoch}"
                                )
                            },
                            commit=False,
                        )
                        if (
                            _every(epoch, getattr(config, "pca_gif_interval", 0))
                            and len(vis_frames_nette) > 1
                        ):
                            log_gif(vis_frames_nette, "pca_progression_nette", "nette")

                # 2. ImageNette: Attention rollout overlay
                if vis_images_nette is not None and _every_or_first(
                    epoch, config.attn_rollout_interval
                ):
                    attn_grid_nette = generate_attention_rollout(
                        model,
                        vis_images_nette,
                        device,
                        img_size=config.img_size,
                        patch_size=config.patch_size,
                    )
                    attn_frames_nette.append(attn_grid_nette)
                    if config.use_wandb:
                        wandb.log(
                            {
                                "vis/nette_attn_rollout": wandb.Image(
                                    attn_grid_nette, caption=f"Epoch {epoch}"
                                )
                            },
                            commit=False,
                        )
                        if (
                            _every(epoch, getattr(config, "attn_gif_interval", 0))
                            and len(attn_frames_nette) > 1
                        ):
                            log_gif(
                                attn_frames_nette,
                                "attn_progression_nette",
                                "attn_nette",
                            )

                # 2. ImageWoof Vis
                if vis_images_woof is not None and _every_or_first(
                    epoch, config.pca_vis_interval
                ):
                    vis_grid_woof = generate_pca_visualization(
                        model,
                        vis_images_woof,
                        device,
                        img_size=config.pca_vis_size,
                        patch_size=config.patch_size,
                        pca_resample=config.pca_resample,
                        per_image_pca=config.pca_per_image,
                    )
                    vis_frames_woof.append(vis_grid_woof)
                    if config.use_wandb:
                        wandb.log(
                            {
                                "vis/woof_patch_pca": wandb.Image(
                                    vis_grid_woof, caption=f"Epoch {epoch}"
                                )
                            },
                            commit=False,
                        )
                        if (
                            _every(epoch, getattr(config, "pca_gif_interval", 0))
                            and len(vis_frames_woof) > 1
                        ):
                            log_gif(vis_frames_woof, "pca_progression_woof", "woof")

                if vis_images_woof is not None and _every_or_first(
                    epoch, config.attn_rollout_interval
                ):
                    attn_grid_woof = generate_attention_rollout(
                        model,
                        vis_images_woof,
                        device,
                        img_size=config.img_size,
                        patch_size=config.patch_size,
                    )
                    attn_frames_woof.append(attn_grid_woof)
                    if config.use_wandb:
                        wandb.log(
                            {
                                "vis/woof_attn_rollout": wandb.Image(
                                    attn_grid_woof, caption=f"Epoch {epoch}"
                                )
                            },
                            commit=False,
                        )
                        if (
                            _every(epoch, getattr(config, "attn_gif_interval", 0))
                            and len(attn_frames_woof) > 1
                        ):
                            log_gif(
                                attn_frames_woof, "attn_progression_woof", "attn_woof"
                            )

                # === NEW VISUALIZATIONS: Training & Attention Dynamics ===
                vis_images = (
                    vis_images_nette
                    if vis_images_nette is not None
                    else vis_images_woof
                )

                # 3. Layer-wise Attention Evolution
                if _every_or_first(epoch, config.layer_attention_interval):
                    layer_attn_vis = generate_layer_attention_evolution(
                        model,
                        vis_images,
                        device,
                        img_size=config.img_size,
                        patch_size=config.patch_size,
                    )
                    if config.use_wandb:
                        wandb.log(
                            {
                                "vis/layer_attention_evolution": wandb.Image(
                                    layer_attn_vis, caption=f"Epoch {epoch}"
                                )
                            },
                            commit=False,
                        )

                # 4. Per-Head Attention
                if _every_or_first(epoch, config.per_head_attention_interval):
                    per_head_vis = generate_per_head_attention(
                        model,
                        vis_images,
                        device,
                        img_size=96,
                        patch_size=config.patch_size,
                        layer_idx=-1,
                    )
                    if config.use_wandb:
                        wandb.log(
                            {
                                "vis/per_head_attention": wandb.Image(
                                    per_head_vis, caption=f"Epoch {epoch} Last Layer"
                                )
                            },
                            commit=False,
                        )

                # 5. Token Similarity Heatmap
                if _every_or_first(epoch, config.token_similarity_interval):
                    token_sim_vis = generate_token_similarity_heatmap(
                        model, vis_images, device, sample_idx=0
                    )
                    if config.use_wandb:
                        wandb.log(
                            {
                                "vis/token_similarity": wandb.Image(
                                    token_sim_vis, caption=f"Epoch {epoch}"
                                )
                            },
                            commit=False,
                        )

                # 6. RSM Across Layers
                if _every_or_first(epoch, config.rsm_interval):
                    rsm_vis = generate_rsm_across_layers(model, vis_images, device)
                    if config.use_wandb:
                        wandb.log(
                            {
                                "vis/rsm_layers": wandb.Image(
                                    rsm_vis, caption=f"Epoch {epoch}"
                                )
                            },
                            commit=False,
                        )

                # 7. Attention Difference Tracking (optional; assumes CLS and can be noisy)
                if attn_tracker is not None and _every_or_first(
                    epoch, getattr(config, "attention_difference_interval", 0)
                ):
                    attn_tracker.capture(model, vis_images, device, epoch)
                    if epoch > 1:
                        attn_diff_vis = attn_tracker.generate_difference_visualization(
                            img_size=config.img_size, patch_size=config.patch_size
                        )
                        if attn_diff_vis is not None and config.use_wandb:
                            wandb.log(
                                {
                                    "vis/attention_difference": wandb.Image(
                                        attn_diff_vis, caption=f"Epoch 1 → {epoch}"
                                    )
                                },
                                commit=False,
                            )

                # 8. Embedding PCA scatter (stable qualitative benchmark)
                if config.use_wandb and _every_or_first(
                    epoch, getattr(config, "embedding_pca_scatter_interval", 0)
                ):
                    try:
                        pca_scatter = generate_embedding_pca_scatter(
                            model,
                            val_loader_nette,
                            device,
                            max_samples=400,
                            view_idx=0,
                            use_last_layer_only=True,
                        )
                        wandb.log(
                            {
                                "vis/embedding_pca_scatter_nette": wandb.Image(
                                    pca_scatter, caption=f"Epoch {epoch}"
                                )
                            },
                            commit=False,
                        )
                    except Exception as e:
                        print(f"Embedding PCA scatter failed: {e}")

                # 9b. Representation drift (lightweight): cosine + linear CKA
                if config.use_wandb and _every_or_first(epoch, config.drift_interval):
                    import wandb

                    def _drift_metrics(
                        tag: str, prev: torch.Tensor, cur: torch.Tensor
                    ) -> dict:
                        if prev is None or cur is None:
                            return {}
                        b = min(prev.shape[0], cur.shape[0])
                        if b < 2:
                            return {}
                        p = prev[:b].float()
                        c = cur[:b].float()

                        p_n = p / (p.norm(dim=1, keepdim=True) + 1e-8)
                        c_n = c / (c.norm(dim=1, keepdim=True) + 1e-8)
                        cos = (p_n * c_n).sum(dim=1).mean().item()
                        l2 = (p_n - c_n).pow(2).sum(dim=1).mean().item()
                        cka = compute_linear_cka(p, c)
                        return {
                            f"epoch_rep/drift_cos_{tag}": cos,
                            f"epoch_rep/drift_l2_{tag}": l2,
                            f"epoch_rep/drift_cka_{tag}": cka,
                        }

                    # Nette drift
                    cur_rep_nette = None
                    if vis_images_nette is not None:
                        with torch.no_grad():
                            cur_rep_nette, _ = model(vis_images_nette.unsqueeze(1))
                        wandb.log(
                            _drift_metrics("nette", prev_rep_nette, cur_rep_nette),
                            commit=False,
                        )
                        prev_rep_nette = cur_rep_nette.detach()

                    # Woof drift
                    cur_rep_woof = None
                    if vis_images_woof is not None:
                        with torch.no_grad():
                            cur_rep_woof, _ = model(vis_images_woof.unsqueeze(1))
                        wandb.log(
                            _drift_metrics("woof", prev_rep_woof, cur_rep_woof),
                            commit=False,
                        )
                        prev_rep_woof = cur_rep_woof.detach()

                # 10. Representation / collapse monitoring (fixed batch)
                if _every_or_first(epoch, config.collapse_monitor_interval):
                    with torch.no_grad():
                        emb_sample, _ = model(vis_images.unsqueeze(1))

                    if config.use_wandb:
                        collapse_vis = generate_collapse_monitor(emb_sample)
                        wandb.log(
                            {
                                "vis/collapse_monitor": wandb.Image(
                                    collapse_vis, caption=f"Epoch {epoch}"
                                )
                            },
                            commit=False,
                        )

                        # Scalar collapse + representation diagnostics (benchmark-friendly)
                        collapse_metrics = compute_feature_collapse_metrics(emb_sample)
                        rep_stats = compute_representation_stats(emb_sample)
                        cov_stats = compute_covariance_metrics(emb_sample)

                        # Track SIGReg-aligned histories
                        isotropy_history.append(rep_stats.get("isotropy", 0.0))
                        effective_rank_history.append(
                            collapse_metrics["effective_rank"]
                        )
                        uniformity_history.append(collapse_metrics["uniformity"])

                        wandb.log(
                            {
                                "collapse/avg_similarity": collapse_metrics[
                                    "avg_similarity"
                                ],
                                "collapse/std": collapse_metrics["std"],
                                "collapse/effective_rank": collapse_metrics[
                                    "effective_rank"
                                ],
                                "collapse/uniformity": collapse_metrics["uniformity"],
                                "rep/norm_mean": rep_stats.get("norm_mean", 0.0),
                                "rep/norm_std": rep_stats.get("norm_std", 0.0),
                                "rep/variance": rep_stats.get("variance", 0.0),
                                "rep/effective_dim": rep_stats.get(
                                    "effective_dim", 0.0
                                ),
                                "rep/isotropy": rep_stats.get("isotropy", 0.0),
                                "rep/cov_offdiag_l2": cov_stats.get(
                                    "cov_offdiag_l2", 0.0
                                ),
                                "rep/var_mean": cov_stats.get("var_mean", 0.0),
                                "rep/var_min": cov_stats.get("var_min", 0.0),
                            },
                            commit=False,
                        )

                        # SIGReg-aligned visualizations (every collapse_monitor_interval)
                        # 1. Isotropy evolution plot (only if we have enough data points)
                        if len(isotropy_history) >= 2:
                            iso_plot = generate_isotropy_evolution_plot(
                                isotropy_history,
                                effective_rank_history,
                                uniformity_history,
                            )
                            wandb.log(
                                {
                                    "vis/isotropy_evolution": wandb.Image(
                                        iso_plot, caption=f"Epoch {epoch}"
                                    )
                                },
                                commit=False,
                            )

                        # 2. Embedding distribution plot (shows Gaussian fit quality)
                        emb_dist_plot = generate_embedding_distribution_plot(emb_sample)
                        wandb.log(
                            {
                                "vis/embedding_distribution": wandb.Image(
                                    emb_dist_plot, caption=f"Epoch {epoch}"
                                )
                            },
                            commit=False,
                        )

                        # 3. Loss-accuracy correlation (every 10 epochs, need enough data)
                        if (
                            epoch % 10 == 0
                            and len(total_loss_history) >= 2
                            and len(accuracy_history) >= 2
                        ):
                            corr_plot = generate_loss_accuracy_correlation_plot(
                                total_loss_history,
                                accuracy_history,
                            )
                            wandb.log(
                                {
                                    "vis/loss_accuracy_correlation": wandb.Image(
                                        corr_plot, caption=f"Epoch {epoch}"
                                    )
                                },
                                commit=False,
                            )

                        # 4. SIGReg loss components plot (every collapse_monitor_interval)
                        if len(sigreg_loss_history) >= 2:
                            loss_comp_plot = generate_sigreg_loss_components_plot(
                                sigreg_loss_history,
                                invariance_loss_history,
                                total_loss_history,
                                lambda_sigreg=config.lambda_sigreg,
                            )
                            wandb.log(
                                {
                                    "vis/sigreg_loss_components": wandb.Image(
                                        loss_comp_plot, caption=f"Epoch {epoch}"
                                    )
                                },
                                commit=False,
                            )

                # 10b. Block-wise activation diagnostics (token norms + residual ratios + drift)
                if config.use_wandb and _every_or_first(
                    epoch, config.block_diag_interval
                ):
                    import wandb

                    def _collect_block_stats(images: torch.Tensor) -> tuple[dict, list]:
                        block_stats = {}
                        block_reps = []

                        has_cls = getattr(model.encoder, "pool", "") == "cls"
                        handles = []

                        def _make_hook(li: int):
                            def _hook(mod, inp, out):
                                x_in = inp[0].detach()
                                x_out = out.detach()
                                x_in_f = x_in.float()
                                x_out_f = x_out.float()

                                token_norm = x_out_f.norm(dim=-1)  # (B,N)
                                block_stats[f"epoch_block/token_norm_mean/l{li}"] = (
                                    token_norm.mean().item()
                                )
                                if has_cls and x_out_f.shape[1] > 1:
                                    block_stats[f"epoch_block/cls_norm_mean/l{li}"] = (
                                        token_norm[:, 0].mean().item()
                                    )
                                    block_stats[
                                        f"epoch_block/patch_norm_mean/l{li}"
                                    ] = token_norm[:, 1:].mean().item()

                                delta = (x_out_f - x_in_f).norm(dim=-1)
                                base = x_in_f.norm(dim=-1).clamp(min=1e-8)
                                block_stats[f"epoch_block/residual_ratio/l{li}"] = (
                                    (delta / base).mean().item()
                                )
                                if has_cls and x_out_f.shape[1] > 1:
                                    block_stats[
                                        f"epoch_block/residual_ratio_cls/l{li}"
                                    ] = (delta[:, 0] / base[:, 0]).mean().item()
                                    block_stats[
                                        f"epoch_block/residual_ratio_patch/l{li}"
                                    ] = (delta[:, 1:] / base[:, 1:]).mean().item()

                                # pooled rep for drift (CLS if present else mean tokens)
                                rep = x_out_f[:, 0] if has_cls else x_out_f.mean(dim=1)
                                block_reps.append(rep.detach())

                            return _hook

                        for li, blk in enumerate(model.encoder.blocks):
                            handles.append(blk.register_forward_hook(_make_hook(li)))

                        _ = model.encoder(images)

                        for h in handles:
                            h.remove()

                        # parameter norm per block (cheap sanity)
                        for li, blk in enumerate(model.encoder.blocks):
                            sq = 0.0
                            for p in blk.parameters():
                                sq += p.detach().float().pow(2).sum().item()
                            block_stats[f"epoch_block/param_norm/l{li}"] = (
                                sq + 1e-12
                            ) ** 0.5

                        return block_stats, block_reps

                    def _log_block(
                        tag: str, images: torch.Tensor, prev_reps: list | None
                    ):
                        stats, reps = _collect_block_stats(images)
                        drift = {}
                        if prev_reps is not None and len(prev_reps) == len(reps):
                            for li, (p, c) in enumerate(zip(prev_reps, reps)):
                                b = min(p.shape[0], c.shape[0])
                                if b < 2:
                                    continue
                                p_n = p[:b] / (p[:b].norm(dim=1, keepdim=True) + 1e-8)
                                c_n = c[:b] / (c[:b].norm(dim=1, keepdim=True) + 1e-8)
                                drift[f"epoch_block/{tag}_drift_cos/l{li}"] = (
                                    (p_n * c_n).sum(dim=1).mean().item()
                                )
                        wandb.log({**stats, **drift}, commit=False)

                        # Curves for quick scanning
                        depth = len(model.encoder.blocks)
                        tok = [
                            stats.get(f"epoch_block/token_norm_mean/l{i}", 0.0)
                            for i in range(depth)
                        ]
                        res = [
                            stats.get(f"epoch_block/residual_ratio/l{i}", 0.0)
                            for i in range(depth)
                        ]
                        plot = generate_layerwise_curves(
                            {"token_norm_mean": tok, "residual_ratio": res},
                            title=f"Block Diagnostics ({tag}) Epoch {epoch}",
                            ylabel="value",
                        )
                        wandb.log(
                            {
                                f"epoch_block/{tag}_diagnostics_plot": wandb.Image(
                                    plot, caption=f"Epoch {epoch}"
                                )
                            },
                            commit=False,
                        )
                        return reps

                    if vis_images_nette is not None:
                        prev_block_reps_nette = _log_block(
                            "nette", vis_images_nette, prev_block_reps_nette
                        )
                    if vis_images_woof is not None:
                        prev_block_reps_woof = _log_block(
                            "woof", vis_images_woof, prev_block_reps_woof
                        )

                # 11. Gradient Flow Heatmap
                if config.use_wandb and _every_or_first(
                    epoch, config.gradient_flow_interval
                ):
                    grad_stats = compute_layer_gradient_stats(model)
                    grad_flow_vis = generate_gradient_flow_heatmap(grad_stats)
                    wandb.log(
                        {
                            "vis/gradient_flow": wandb.Image(
                                grad_flow_vis, caption=f"Epoch {epoch}"
                            )
                        },
                        commit=False,
                    )

                # 12. Transformer attention diagnostics (fixed batch; interval-gated)
                need_attn_maps = config.use_wandb and (
                    _every_or_first(epoch, config.transformer_diag_interval)
                    or _every_or_first(epoch, config.attn_distance_headmap_interval)
                    or _every_or_first(
                        epoch, getattr(config, "attn_entropy_headmap_interval", 0)
                    )
                )
                need_pocp = config.use_wandb and _every_or_first(
                    epoch, getattr(config, "pocp_interval", 0)
                )
                need_forward = need_attn_maps or need_pocp

                attns = []
                pocp_layer_stats = None
                if need_forward:
                    with torch.no_grad():
                        grid_size = config.img_size // config.patch_size
                        for blk in model.encoder.blocks:
                            blk.attn.output_attention = bool(need_attn_maps)
                            blk.attn.output_pocp = bool(need_pocp)
                            if need_pocp:
                                blk.attn.pocp_grid_size = grid_size
                                blk.attn.pocp_num_pairs = int(
                                    getattr(config, "pocp_num_pairs", 512)
                                )
                                blk.attn.pocp_num_distance_bins = int(
                                    getattr(config, "pocp_num_distance_bins", 6)
                                )
                                blk.attn.pocp_freq_planes = int(
                                    getattr(config, "pocp_freq_planes", 32)
                                )
                        _ = model.encoder(vis_images)
                        if need_attn_maps:
                            attns = model.encoder.get_attention_maps()
                        if need_pocp:
                            pocp_layer_stats = [
                                getattr(blk.attn, "pocp_stats", None)
                                for blk in model.encoder.blocks
                            ]
                        for blk in model.encoder.blocks:
                            blk.attn.output_attention = False
                            blk.attn.output_pocp = False
                            if hasattr(blk.attn, "attn_map"):
                                blk.attn.attn_map = None
                            if hasattr(blk.attn, "pocp_stats"):
                                blk.attn.pocp_stats = None

                    if need_pocp and pocp_layer_stats:
                        import math
                        import numpy as np

                        num_layers = len(pocp_layer_stats)
                        num_heads = int(getattr(config, "num_heads", 0)) or 1
                        pocp_map = np.full(
                            (num_layers, num_heads), np.nan, dtype=np.float32
                        )
                        pocp_high_map = np.full(
                            (num_layers, num_heads), np.nan, dtype=np.float32
                        )
                        pocp_low_map = np.full(
                            (num_layers, num_heads), np.nan, dtype=np.float32
                        )

                        dist_edges = None
                        curves = []
                        curves_hi = []
                        curves_lo = []
                        curves_qk = []
                        qk_means = []
                        qk_stds = []
                        k_planes = None

                        for li, s in enumerate(pocp_layer_stats):
                            if not s:
                                continue
                            ph = s.get("pocp_per_head", None)
                            hh = s.get("pocp_high_per_head", None)
                            ll = s.get("pocp_low_per_head", None)
                            if isinstance(ph, torch.Tensor) and ph.numel() == num_heads:
                                pocp_map[li, :] = ph.detach().float().cpu().numpy()
                            if isinstance(hh, torch.Tensor) and hh.numel() == num_heads:
                                pocp_high_map[li, :] = hh.detach().float().cpu().numpy()
                            if isinstance(ll, torch.Tensor) and ll.numel() == num_heads:
                                pocp_low_map[li, :] = ll.detach().float().cpu().numpy()

                            if (
                                dist_edges is None
                                and s.get("dist_edges", None) is not None
                            ):
                                dist_edges = s.get("dist_edges", None)

                            if s.get("pocp_curve", None):
                                curves.append(list(s.get("pocp_curve")))
                                curves_hi.append(list(s.get("pocp_high_curve", [])))
                                curves_lo.append(list(s.get("pocp_low_curve", [])))
                                curves_qk.append(list(s.get("qk_curve", [])))

                            if "qk_dot_mean" in s:
                                qk_means.append(float(s["qk_dot_mean"]))
                            if "qk_dot_std" in s:
                                qk_stds.append(float(s["qk_dot_std"]))
                            if k_planes is None and "k_planes" in s:
                                k_planes = int(s["k_planes"])

                        pocp_mean = (
                            float(np.nanmean(pocp_map))
                            if np.isfinite(pocp_map).any()
                            else 0.0
                        )
                        pocp_high_mean = (
                            float(np.nanmean(pocp_high_map))
                            if np.isfinite(pocp_high_map).any()
                            else 0.0
                        )
                        pocp_low_mean = (
                            float(np.nanmean(pocp_low_map))
                            if np.isfinite(pocp_low_map).any()
                            else 0.0
                        )
                        wandb.log(
                            {
                                "epoch_pocp/mean": pocp_mean,
                                "epoch_pocp/high_freq_mean": pocp_high_mean,
                                "epoch_pocp/low_freq_mean": pocp_low_mean,
                                "epoch_pocp/qk_dot_mean": (
                                    sum(qk_means) / max(len(qk_means), 1)
                                ),
                                "epoch_pocp/qk_dot_std": (
                                    sum(qk_stds) / max(len(qk_stds), 1)
                                ),
                                "epoch_pocp/k_planes": float(k_planes or 0),
                            },
                            commit=False,
                        )

                        pocp_img = generate_pocp_per_head_heatmaps(
                            pocp_map,
                            pocp_high_map,
                            pocp_low_map,
                            title_prefix=f"POCP (Epoch {epoch})",
                        )
                        wandb.log(
                            {
                                "vis/pocp_per_head": wandb.Image(
                                    pocp_img, caption=f"Epoch {epoch}"
                                )
                            },
                            commit=False,
                        )

                        # Distance curves aggregated across layers (ignore NaNs).
                        if dist_edges is not None and curves:
                            edges = list(dist_edges)
                            centers = []
                            for b in range(len(edges) - 1):
                                lo = float(edges[b])
                                hi = float(edges[b + 1])
                                if lo <= 0.0:
                                    centers.append(0.5 * hi)
                                else:
                                    centers.append(math.sqrt(lo * hi))

                            def _nanmean_curve(
                                curve_list: list[list[float]],
                            ) -> list[float]:
                                if not curve_list:
                                    return []
                                a = np.asarray(curve_list, dtype=np.float32)
                                with np.errstate(all="ignore"):
                                    m = np.nanmean(a, axis=0)
                                return [float(x) for x in m.tolist()]

                            pocp_curve = _nanmean_curve(curves)
                            pocp_hi_curve = _nanmean_curve(curves_hi)
                            pocp_lo_curve = _nanmean_curve(curves_lo)
                            qk_curve = _nanmean_curve(curves_qk)

                            pocp_curve_plot = generate_xy_curves(
                                centers,
                                {
                                    "POCP": pocp_curve,
                                    "POCP_high": pocp_hi_curve,
                                    "POCP_low": pocp_lo_curve,
                                },
                                title=f"POCP vs Patch Distance (Epoch {epoch})",
                                xlabel="patch distance (L2)",
                                ylabel="POCP",
                            )
                            wandb.log(
                                {
                                    "vis/pocp_by_distance": wandb.Image(
                                        pocp_curve_plot, caption=f"Epoch {epoch}"
                                    )
                                },
                                commit=False,
                            )

                            qk_curve_plot = generate_xy_curves(
                                centers,
                                {"Q·K (unrotated)": qk_curve},
                                title=f"Unrotated Q·K vs Patch Distance (Epoch {epoch})",
                                xlabel="patch distance (L2)",
                                ylabel="mean Q·K",
                            )
                            wandb.log(
                                {
                                    "vis/pocp_qk_by_distance": wandb.Image(
                                        qk_curve_plot, caption=f"Epoch {epoch}"
                                    )
                                },
                                commit=False,
                            )

                    if attns:
                        last_attn = attns[-1]
                        wandb.log(
                            {
                                "epoch_attn/entropy": float(
                                    compute_entropy(last_attn).item()
                                ),
                                "epoch_attn/gini": float(
                                    compute_gini(last_attn).item()
                                ),
                                "epoch_attn/sparsity": float(
                                    compute_sparsity(last_attn).item()
                                ),
                            },
                            commit=False,
                        )

                        head_div = compute_head_diversity(last_attn)
                        wandb.log(
                            {
                                "epoch_attn/head_similarity": head_div.get(
                                    "head_similarity", 0.0
                                ),
                                "epoch_attn/head_variance": head_div.get(
                                    "head_variance", 0.0
                                ),
                                "epoch_attn/effective_heads": head_div.get(
                                    "effective_heads", 0.0
                                ),
                            },
                            commit=False,
                        )

                        # Per-layer attention structure + distance trends
                        if _every_or_first(epoch, config.transformer_diag_interval):
                            layer_p2c = []
                            layer_c2p = []
                            layer_diag = []
                            layer_head_ent = []
                            layer_attn_dist = []
                            layer_local_r1 = []
                            eps = 1e-8

                            epoch_attn_layer = {}
                            for li, a in enumerate(attns):
                                m = compute_attention_structure_metrics(a)
                                layer_p2c.append(m.get("patches_to_cls", 0.0))
                                layer_c2p.append(m.get("cls_to_patches", 0.0))
                                layer_diag.append(m.get("diag_mass", 0.0))

                                head_ent = (
                                    -(a * torch.log(a + eps)).sum(dim=-1).mean(dim=-1)
                                )  # (B,H)
                                layer_head_ent.append(head_ent.mean().item())

                                epoch_attn_layer[
                                    f"epoch_attn_layer/patches_to_cls/l{li}"
                                ] = m.get("patches_to_cls", 0.0)
                                epoch_attn_layer[
                                    f"epoch_attn_layer/cls_to_patches/l{li}"
                                ] = m.get("cls_to_patches", 0.0)
                                epoch_attn_layer[
                                    f"epoch_attn_layer/diag_mass/l{li}"
                                ] = m.get("diag_mass", 0.0)
                                epoch_attn_layer[
                                    f"epoch_attn_layer/head_entropy_mean/l{li}"
                                ] = head_ent.mean().item()

                                d = compute_attention_distance_metrics(
                                    a,
                                    grid_size=config.img_size // config.patch_size,
                                    radius=1,
                                )
                                layer_attn_dist.append(
                                    d.get("patch_attn_distance_mean", 0.0)
                                )
                                layer_local_r1.append(d.get("patch_local_mass_r1", 0.0))
                                epoch_attn_layer[
                                    f"epoch_attn_layer/patch_attn_distance_mean/l{li}"
                                ] = d.get("patch_attn_distance_mean", 0.0)
                                epoch_attn_layer[
                                    f"epoch_attn_layer/patch_local_mass_r1/l{li}"
                                ] = d.get("patch_local_mass_r1", 0.0)

                            wandb.log(epoch_attn_layer, commit=False)

                            attn_plot = generate_layerwise_curves(
                                {
                                    "patches→CLS": layer_p2c,
                                    "CLS→patches": layer_c2p,
                                    "diag_mass": layer_diag,
                                    "head_entropy_mean": layer_head_ent,
                                },
                                title=f"Transformer Attention Diagnostics (Epoch {epoch})",
                                ylabel="value",
                            )
                            wandb.log(
                                {
                                    "vis/attn_layer_diagnostics": wandb.Image(
                                        attn_plot, caption=f"Epoch {epoch}"
                                    )
                                },
                                commit=False,
                            )

                            dist_plot = generate_layerwise_curves(
                                {
                                    "attn_distance_mean": layer_attn_dist,
                                    "local_mass_r1": layer_local_r1,
                                },
                                title=f"Attention Distance/Locality (Epoch {epoch})",
                                ylabel="value",
                            )
                            wandb.log(
                                {
                                    "vis/attn_layer_distance": wandb.Image(
                                        dist_plot, caption=f"Epoch {epoch}"
                                    )
                                },
                                commit=False,
                            )

                        # Head×layer heatmaps (reuse computed maps; avoid extra forward)
                        if _every_or_first(
                            epoch, config.attn_distance_headmap_interval
                        ):
                            dist_head = (
                                generate_attention_distance_per_head_heatmap_from_maps(
                                    attns,
                                    grid_size=config.img_size // config.patch_size,
                                    radius=1,
                                )
                            )
                            wandb.log(
                                {
                                    "vis/attn_distance_per_head": wandb.Image(
                                        dist_head, caption=f"Epoch {epoch}"
                                    )
                                },
                                commit=False,
                            )

                        if _every_or_first(
                            epoch, getattr(config, "attn_entropy_headmap_interval", 0)
                        ):
                            ent_head = (
                                generate_attention_entropy_per_head_heatmap_from_maps(
                                    attns
                                )
                            )
                            wandb.log(
                                {
                                    "vis/attn_entropy_per_head": wandb.Image(
                                        ent_head, caption=f"Epoch {epoch}"
                                    )
                                },
                                commit=False,
                            )

                # 12b. Attention logit stats (pre-softmax QK^T) per layer
                if config.use_wandb and _every_or_first(
                    epoch, config.attn_logits_interval
                ):
                    import wandb

                    for blk in model.encoder.blocks:
                        blk.attn.output_attn_logits = True

                    _ = model.encoder(vis_images)

                    logit_metrics = {}
                    means = []
                    p99s = []
                    maxs = []
                    for li, blk in enumerate(model.encoder.blocks):
                        logits = getattr(blk.attn, "attn_logits", None)
                        if logits is None:
                            means.append(0.0)
                            p99s.append(0.0)
                            maxs.append(0.0)
                            continue
                        t = logits.detach().float()
                        means.append(float(t.mean().item()))
                        maxs.append(float(t.max().item()))
                        try:
                            p99s.append(
                                float(torch.quantile(t.reshape(-1), 0.99).item())
                            )
                        except Exception:
                            p99s.append(
                                float(
                                    t.flatten()
                                    .kthvalue(max(1, int(0.99 * t.numel())))
                                    .values.item()
                                )
                            )

                        logit_metrics[f"epoch_attn_logit/mean/l{li}"] = means[-1]
                        logit_metrics[f"epoch_attn_logit/p99/l{li}"] = p99s[-1]
                        logit_metrics[f"epoch_attn_logit/max/l{li}"] = maxs[-1]

                    for blk in model.encoder.blocks:
                        blk.attn.output_attn_logits = False
                        blk.attn.attn_logits = None

                    wandb.log(logit_metrics, commit=False)
                    logit_plot = generate_layerwise_curves(
                        {"mean": means, "p99": p99s, "max": maxs},
                        title=f"Attention Logits (pre-softmax) (Epoch {epoch})",
                        ylabel="value",
                    )
                    wandb.log(
                        {
                            "epoch_attn_logit/curves": wandb.Image(
                                logit_plot, caption=f"Epoch {epoch}"
                            )
                        },
                        commit=False,
                    )

                # 12c. MLP output stats per layer (outliers/saturation proxy)
                if config.use_wandb and _every_or_first(
                    epoch, config.mlp_output_stats_interval
                ):
                    import wandb

                    mlp_metrics = {}
                    means = []
                    stds = []
                    max_abs = []
                    outlier = []

                    handles = []

                    def _make_mlp_hook(li: int):
                        def _hook(_mod, _inp, out):
                            y = out.detach().float()
                            m = float(y.mean().item())
                            s = float(y.std().item())
                            ma = float(y.abs().max().item())
                            thr = 5.0 * s + 1e-6
                            o = float((y.abs() > thr).float().mean().item())

                            means.append(m)
                            stds.append(s)
                            max_abs.append(ma)
                            outlier.append(o)

                            mlp_metrics[f"epoch_mlp/out_mean/l{li}"] = m
                            mlp_metrics[f"epoch_mlp/out_std/l{li}"] = s
                            mlp_metrics[f"epoch_mlp/out_max_abs/l{li}"] = ma
                            mlp_metrics[f"epoch_mlp/out_outlier_rate_5std/l{li}"] = o

                        return _hook

                    for li, blk in enumerate(model.encoder.blocks):
                        handles.append(
                            blk.mlp.register_forward_hook(_make_mlp_hook(li))
                        )

                    _ = model.encoder(vis_images)

                    for h in handles:
                        h.remove()

                    wandb.log(mlp_metrics, commit=False)
                    mlp_plot = generate_layerwise_curves(
                        {"std": stds, "max_abs": max_abs, "outlier_rate_5std": outlier},
                        title=f"MLP Output Stats (Epoch {epoch})",
                        ylabel="value",
                    )
                    wandb.log(
                        {
                            "epoch_mlp/curves": wandb.Image(
                                mlp_plot, caption=f"Epoch {epoch}"
                            )
                        },
                        commit=False,
                    )

                # 14. t-SNE/UMAP Embedding Projection (optional; expensive)
                if config.use_wandb and _every_or_first(
                    epoch, config.embedding_projection_interval
                ):
                    try:
                        tsne_vis = generate_embedding_projection(
                            model,
                            val_loader_nette,
                            device,
                            method="tsne",
                            max_samples=300,
                        )
                        wandb.log(
                            {
                                "vis/embedding_tsne": wandb.Image(
                                    tsne_vis, caption=f"Epoch {epoch}"
                                )
                            },
                            commit=False,
                        )
                    except Exception as e:
                        print(f"t-SNE visualization failed: {e}")

                # 15. Expensive optimization diagnostics (can be slow)
                if config.use_wandb and _every_or_first(epoch, config.gns_interval):
                    try:
                        if device.type == "cuda":
                            free_b, _total_b = torch.cuda.mem_get_info()
                            free_mb = float(free_b) / (1024.0 * 1024.0)
                            if free_mb < float(
                                getattr(config, "heavy_diag_min_free_mb", 0)
                            ):
                                wandb.log(
                                    {
                                        "sys/heavy_diag_skipped": 1.0,
                                        "sys/cuda_free_mb": free_mb,
                                    },
                                    commit=False,
                                )
                                raise RuntimeError(
                                    f"Skipping heavy diagnostics: only {free_mb:.1f} MiB CUDA free"
                                )
                        batch_data = next(iter(train_loader))
                        if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                            diag_crops, diag_labels = batch_data[0], batch_data[1]
                        else:
                            diag_crops, diag_labels = batch_data, None
                        b = min(
                            int(
                                getattr(
                                    config,
                                    "heavy_diagnostic_batch_size",
                                    config.diagnostic_batch_size,
                                )
                            ),
                            int(config.diagnostic_batch_size),
                            int(diag_labels.shape[0]),
                        )
                        diag_crops = [c[:b] for c in diag_crops]
                        diag_labels = diag_labels[:b]
                        with torch.enable_grad():
                            gns = _gns_microbatch(
                                model,
                                probe,
                                loss_fn,
                                diag_crops,
                                diag_labels,
                                device,
                                mixed_precision=config.mixed_precision,
                                microbatches=config.gns_microbatches,
                            )
                        wandb.log(gns, commit=False)
                    except Exception as e:
                        print(f"GNS diagnostics failed: {e}")

                if config.use_wandb and _every_or_first(
                    epoch, config.sharpness_interval
                ):
                    try:
                        if device.type == "cuda":
                            free_b, _total_b = torch.cuda.mem_get_info()
                            free_mb = float(free_b) / (1024.0 * 1024.0)
                            if free_mb < float(
                                getattr(config, "heavy_diag_min_free_mb", 0)
                            ):
                                wandb.log(
                                    {
                                        "sys/heavy_diag_skipped": 1.0,
                                        "sys/cuda_free_mb": free_mb,
                                    },
                                    commit=False,
                                )
                                raise RuntimeError(
                                    f"Skipping heavy diagnostics: only {free_mb:.1f} MiB CUDA free"
                                )
                        batch_data = next(iter(train_loader))
                        if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                            diag_crops, diag_labels = batch_data[0], batch_data[1]
                        else:
                            diag_crops, diag_labels = batch_data, None
                        b = min(
                            int(
                                getattr(
                                    config,
                                    "heavy_diagnostic_batch_size",
                                    config.diagnostic_batch_size,
                                )
                            ),
                            int(config.diagnostic_batch_size),
                            int(diag_labels.shape[0]),
                        )
                        diag_crops = [c[:b] for c in diag_crops]
                        diag_labels = diag_labels[:b]
                        with torch.enable_grad():
                            sharp = _hessian_top_eig_power_iter(
                                model,
                                probe,
                                loss_fn,
                                diag_crops,
                                diag_labels,
                                device,
                                mixed_precision=config.mixed_precision,
                                iters=config.sharpness_power_iters,
                            )
                        wandb.log(sharp, commit=False)
                    except Exception as e:
                        print(f"Sharpness diagnostics failed: {e}")

                if config.use_wandb and _every_or_first(
                    epoch, config.landscape_interval
                ):
                    try:
                        if device.type == "cuda":
                            free_b, _total_b = torch.cuda.mem_get_info()
                            free_mb = float(free_b) / (1024.0 * 1024.0)
                            if free_mb < float(
                                getattr(config, "heavy_diag_min_free_mb", 0)
                            ):
                                wandb.log(
                                    {
                                        "sys/heavy_diag_skipped": 1.0,
                                        "sys/cuda_free_mb": free_mb,
                                    },
                                    commit=False,
                                )
                                raise RuntimeError(
                                    f"Skipping heavy diagnostics: only {free_mb:.1f} MiB CUDA free"
                                )
                        batch_data = next(iter(train_loader))
                        if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                            diag_crops, diag_labels = batch_data[0], batch_data[1]
                        else:
                            diag_crops, diag_labels = batch_data, None
                        b = min(
                            int(
                                getattr(
                                    config,
                                    "heavy_diagnostic_batch_size",
                                    config.diagnostic_batch_size,
                                )
                            ),
                            int(config.diagnostic_batch_size),
                            int(diag_labels.shape[0]),
                        )
                        diag_crops = [c[:b] for c in diag_crops]
                        diag_labels = diag_labels[:b]
                        with torch.enable_grad():
                            ls = _loss_landscape_slice(
                                model,
                                probe,
                                loss_fn,
                                diag_crops,
                                diag_labels,
                                device,
                                mixed_precision=config.mixed_precision,
                                radius=config.landscape_radius,
                                points=config.landscape_points,
                            )
                        if ls["alphas"]:
                            plot_g = generate_loss_landscape_slice(
                                ls["alphas"],
                                ls["losses_grad"],
                                title=f"Loss Landscape Slice (grad dir) Epoch {epoch}",
                            )
                            plot_r = generate_loss_landscape_slice(
                                ls["alphas"],
                                ls["losses_rand"],
                                title=f"Loss Landscape Slice (random dir) Epoch {epoch}",
                            )
                            wandb.log(
                                {
                                    "epoch_opt/loss_landscape_grad": wandb.Image(
                                        plot_g, caption=f"Epoch {epoch}"
                                    ),
                                    "epoch_opt/loss_landscape_rand": wandb.Image(
                                        plot_r, caption=f"Epoch {epoch}"
                                    ),
                                },
                                commit=False,
                            )
                    except Exception as e:
                        print(f"Loss landscape diagnostics failed: {e}")

                if config.use_wandb and _every_or_first(
                    epoch, config.landscape2d_interval
                ):
                    try:
                        if device.type == "cuda":
                            free_b, _total_b = torch.cuda.mem_get_info()
                            free_mb = float(free_b) / (1024.0 * 1024.0)
                            if free_mb < float(
                                getattr(config, "heavy_diag_min_free_mb", 0)
                            ):
                                wandb.log(
                                    {
                                        "sys/heavy_diag_skipped": 1.0,
                                        "sys/cuda_free_mb": free_mb,
                                    },
                                    commit=False,
                                )
                                raise RuntimeError(
                                    f"Skipping heavy diagnostics: only {free_mb:.1f} MiB CUDA free"
                                )
                        batch_data = next(iter(train_loader))
                        if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                            diag_crops, diag_labels = batch_data[0], batch_data[1]
                        else:
                            diag_crops, diag_labels = batch_data, None
                        b = min(
                            int(
                                getattr(
                                    config,
                                    "heavy_diagnostic_batch_size",
                                    config.diagnostic_batch_size,
                                )
                            ),
                            int(config.diagnostic_batch_size),
                            int(diag_labels.shape[0]),
                        )
                        diag_crops = [c[:b].to(device) for c in diag_crops]
                        diag_labels = diag_labels[:b].to(device)
                        # Use loss-landscapes library for 2D landscape
                        metric = LeJEPALossMetric(
                            probe,
                            loss_fn,
                            diag_crops,
                            diag_labels,
                            device,
                            config.mixed_precision,
                        )
                        landscape_data = loss_landscapes.random_plane(
                            model,
                            metric,
                            distance=config.landscape2d_radius,
                            steps=config.landscape2d_points,
                            normalization="filter",
                            deepcopy_model=True,
                        )
                        ls2 = {
                            "alphas": np.linspace(
                                -1.0, 1.0, config.landscape2d_points
                            ).tolist(),
                            "betas": np.linspace(
                                -1.0, 1.0, config.landscape2d_points
                            ).tolist(),
                            "loss_grid": landscape_data.tolist(),
                        }
                        if ls2["alphas"] and ls2["betas"] and ls2["loss_grid"]:
                            # Basic 2D heatmap
                            plot2d = generate_loss_landscape_2d(
                                ls2["alphas"],
                                ls2["betas"],
                                ls2["loss_grid"],
                                title=f"Loss Landscape 2D Epoch {epoch}",
                            )
                            wandb.log(
                                {
                                    "epoch_opt/loss_landscape_2d": wandb.Image(
                                        plot2d, caption=f"Epoch {epoch}"
                                    )
                                },
                                commit=False,
                            )

                            # Publication-quality contour
                            contour = generate_loss_landscape_contour(
                                ls2["alphas"],
                                ls2["betas"],
                                ls2["loss_grid"],
                                title=f"Loss Landscape Contour Epoch {epoch}",
                            )
                            wandb.log(
                                {
                                    "epoch_opt/loss_landscape_contour": wandb.Image(
                                        contour, caption=f"Epoch {epoch}"
                                    )
                                },
                                commit=False,
                            )

                            # 3D surface
                            surf3d = generate_loss_landscape_3d(
                                ls2["alphas"],
                                ls2["betas"],
                                ls2["loss_grid"],
                                title=f"Loss Landscape 3D Epoch {epoch}",
                            )
                            wandb.log(
                                {
                                    "epoch_opt/loss_landscape_3d": wandb.Image(
                                        surf3d, caption=f"Epoch {epoch}"
                                    )
                                },
                                commit=False,
                            )

                            # 3D + projection combined
                            combined = generate_loss_landscape_3d_with_contour(
                                ls2["alphas"],
                                ls2["betas"],
                                ls2["loss_grid"],
                                title=f"Loss Landscape 3D+Proj Epoch {epoch}",
                            )
                            wandb.log(
                                {
                                    "epoch_opt/loss_landscape_3d_proj": wandb.Image(
                                        combined, caption=f"Epoch {epoch}"
                                    )
                                },
                                commit=False,
                            )

                            # Interactive plotly (HTML)
                            html_str = generate_loss_landscape_plotly(
                                ls2["alphas"],
                                ls2["betas"],
                                ls2["loss_grid"],
                                title=f"Loss Landscape Interactive Epoch {epoch}",
                            )
                            wandb.log(
                                {
                                    "epoch_opt/loss_landscape_interactive": wandb.Html(
                                        html_str
                                    )
                                },
                                commit=False,
                            )
                    except Exception as e:
                        print(f"2D loss landscape diagnostics failed: {e}")
                    finally:
                        # Clean up GPU memory after heavy diagnostic
                        del diag_crops, diag_labels, metric
                        _cleanup_gpu_memory(device)

                if config.use_wandb and _every_or_first(
                    epoch, config.head_ablation_interval
                ):
                    try:
                        if device.type == "cuda":
                            free_b, _total_b = torch.cuda.mem_get_info()
                            free_mb = float(free_b) / (1024.0 * 1024.0)
                            if free_mb < float(
                                getattr(config, "heavy_diag_min_free_mb", 0)
                            ):
                                wandb.log(
                                    {
                                        "sys/heavy_diag_skipped": 1.0,
                                        "sys/cuda_free_mb": free_mb,
                                    },
                                    commit=False,
                                )
                                raise RuntimeError(
                                    f"Skipping heavy diagnostics: only {free_mb:.1f} MiB CUDA free"
                                )
                        batch_data = next(iter(train_loader))
                        if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                            diag_crops, diag_labels = batch_data[0], batch_data[1]
                        else:
                            diag_crops, diag_labels = batch_data, None
                        b = min(
                            int(
                                getattr(
                                    config,
                                    "heavy_diagnostic_batch_size",
                                    config.diagnostic_batch_size,
                                )
                            ),
                            int(config.diagnostic_batch_size),
                            int(diag_labels.shape[0]),
                        )
                        diag_crops = [c[:b] for c in diag_crops]
                        diag_labels = diag_labels[:b]
                        ab = _head_ablation_sensitivity(
                            model,
                            probe,
                            loss_fn,
                            diag_crops,
                            diag_labels,
                            device,
                            mixed_precision=config.mixed_precision,
                            last_layers=config.head_ablation_layers,
                        )
                        if ab.get("delta_loss"):
                            base = float(ab.get("base_loss", 0.0))
                            layers = ab.get("layers", [])
                            title = (
                                f"Head Ablation Δloss (last {len(layers)} layers) "
                                f"(base={base:.4f}) Epoch {epoch}"
                            )
                            hm = generate_head_ablation_heatmap(
                                ab["delta_loss"], title=title
                            )
                            wandb.log(
                                {
                                    "epoch_attn/head_ablation_delta_loss": wandb.Image(
                                        hm, caption=f"Epoch {epoch}"
                                    )
                                },
                                commit=False,
                            )
                    except Exception as e:
                        print(f"Head ablation diagnostics failed: {e}")
                    finally:
                        del diag_crops, diag_labels
                        _cleanup_gpu_memory(device)

                # ==============================================================
                # Attention Analysis Module - Epoch-level logging
                # ==============================================================
                if (
                    config.attention_analysis_enabled
                    and attribution_probe_images is not None
                    and config.use_wandb
                ):
                    try:
                        # Attribution visualization
                        if _every_or_first(epoch, config.attribution_vis_interval):
                            method_attrs = {}
                            for method_name in config.attribution_methods:
                                try:
                                    # Compute attribution for first probe image
                                    attr = get_attribution(
                                        model,
                                        attribution_probe_images[0:1],
                                        method=method_name,
                                        target_class=int(
                                            attribution_probe_labels[0].item()
                                        ),
                                    )
                                    method_attrs[method_name] = attr

                                    # Log individual attribution heatmap
                                    from attention_analysis.visualization.heatmaps import (
                                        overlay_attribution,
                                    )

                                    # Convert tensor to PIL to avoid dtype issues
                                    probe_img_pil = _tensor_to_pil(
                                        attribution_probe_images[0]
                                    )
                                    overlay = overlay_attribution(
                                        probe_img_pil,
                                        attr,
                                        alpha=0.5,
                                    )
                                    wandb.log(
                                        {
                                            f"attribution/{method_name}": wandb.Image(
                                                overlay, caption=f"Epoch {epoch}"
                                            )
                                        },
                                        commit=False,
                                    )
                                except Exception as e:
                                    print(f"  Attribution {method_name} failed: {e}")

                            # Log comparison visualization
                            if len(method_attrs) > 1:
                                try:
                                    # Use PIL image for comparison (reuse from above)
                                    comp_img_pil = _tensor_to_pil(
                                        attribution_probe_images[0]
                                    )
                                    comparison = compare_methods_visual(
                                        comp_img_pil,
                                        method_attrs,
                                    )
                                    wandb.log(
                                        {
                                            "attribution/comparison": wandb.Image(
                                                comparison, caption=f"Epoch {epoch}"
                                            )
                                        },
                                        commit=False,
                                    )
                                except Exception as e:
                                    print(f"  Attribution comparison failed: {e}")

                        # Head specialization analysis
                        if (
                            config.head_specialization_enabled
                            and head_tracker is not None
                            and _every_or_first(
                                epoch, config.head_specialization_interval
                            )
                        ):
                            try:
                                importance = head_tracker.compute_head_importance(
                                    method="activation"
                                )
                                if importance:
                                    importance_img = plot_head_importance(
                                        importance,
                                        title=f"Head Importance (Epoch {epoch})",
                                    )
                                    wandb.log(
                                        {
                                            "attention/head_importance": wandb.Image(
                                                importance_img, caption=f"Epoch {epoch}"
                                            )
                                        },
                                        commit=False,
                                    )
                            except Exception as e:
                                print(f"  Head importance failed: {e}")

                        # Faithfulness evaluation (expensive)
                        if config.faithfulness_eval_enabled and _every_or_first(
                            epoch, config.faithfulness_eval_interval
                        ):
                            try:
                                # Compute attributions for evaluation
                                eval_images = attribution_probe_images[
                                    : config.faithfulness_num_samples
                                ]
                                eval_labels = attribution_probe_labels[
                                    : config.faithfulness_num_samples
                                ]

                                for method_name in config.attribution_methods[
                                    :2
                                ]:  # Limit to 2 for speed
                                    attrs = []
                                    for i in range(len(eval_images)):
                                        try:
                                            attr = get_attribution(
                                                model,
                                                eval_images[i : i + 1],
                                                method=method_name,
                                                target_class=int(eval_labels[i].item()),
                                            )
                                            attrs.append(attr)
                                        except Exception:
                                            continue

                                    if attrs:
                                        attrs_tensor = torch.stack(attrs)
                                        faith_result = compute_faithfulness_metrics(
                                            model,
                                            eval_images[: len(attrs)],
                                            attrs_tensor,
                                            target_classes=[
                                                int(lbl.item())
                                                for lbl in eval_labels[: len(attrs)]
                                            ],
                                            num_steps=config.faithfulness_num_steps,
                                            show_progress=False,
                                        )

                                        wandb.log(
                                            {
                                                f"faithfulness/{method_name}_insertion_auc": faith_result[
                                                    "mean_insertion_auc"
                                                ],
                                                f"faithfulness/{method_name}_deletion_auc": faith_result[
                                                    "mean_deletion_auc"
                                                ],
                                                f"faithfulness/{method_name}_combined": faith_result[
                                                    "mean_combined"
                                                ],
                                            },
                                            commit=False,
                                        )
                            except Exception as e:
                                print(f"  Faithfulness eval failed: {e}")

                    except Exception as e:
                        print(f"Attention analysis failed: {e}")
                    finally:
                        _cleanup_gpu_memory(device)

                # ==============================================================
                # Production Dashboard - Health Checks & Summary Dashboards
                # ==============================================================
                if (
                    getattr(config, "production_dashboard_enabled", True)
                    and config.use_wandb
                ):
                    try:
                        # Fetch diagnostic batch for dashboard visualizations
                        batch_data = next(iter(train_loader))
                        # Handle different dataloader return formats:
                        # - (crops_list, labels) where crops_list is [global1, global2, local...]
                        # - single tensor
                        if (
                            isinstance(batch_data, (list, tuple))
                            and len(batch_data) >= 2
                        ):
                            diag_crops_dash = batch_data[0]
                        else:
                            diag_crops_dash = batch_data

                        # Get first crop tensor from the crops list
                        if isinstance(diag_crops_dash, (list, tuple)):
                            first_crop = diag_crops_dash[0]
                        else:
                            first_crop = diag_crops_dash

                        b_dash = min(config.diagnostic_batch_size, first_crop.shape[0])
                        diag_images = first_crop[:b_dash].to(device)

                        # Get attention maps if available
                        diag_attn_maps = None
                        for blk in model.encoder.blocks:
                            attn = getattr(blk, "attn", None)
                            if attn is not None and hasattr(attn, "output_attention"):
                                attn.output_attention = True

                        with torch.no_grad():
                            encoder_out = model.encoder(diag_images)
                            if isinstance(encoder_out, tuple):
                                encoder_out = encoder_out[0]
                            # Collect attention maps
                            attn_list = []
                            for blk in model.encoder.blocks:
                                attn = getattr(blk, "attn", None)
                                if (
                                    attn is not None
                                    and hasattr(attn, "attn_map")
                                    and attn.attn_map is not None
                                ):
                                    attn_list.append(attn.attn_map.detach())
                            if attn_list:
                                diag_attn_maps = torch.stack(
                                    attn_list, dim=1
                                )  # (B, L, H, N, N)

                        # Disable attention output after collecting
                        for blk in model.encoder.blocks:
                            attn = getattr(blk, "attn", None)
                            if attn is not None and hasattr(attn, "output_attention"):
                                attn.output_attention = False

                        # Collect current epoch metrics
                        current_epoch_metrics = {
                            "val_accuracy_nette": val_metrics_nette.get(
                                "val_accuracy", 0
                            )
                            / 100
                            if val_metrics_nette
                            else 0,
                            "val_accuracy_woof": val_metrics_woof.get("val_accuracy", 0)
                            / 100
                            if val_metrics_woof
                            else 0,
                            "knn/nette_top1": knn_metrics_nette.get("knn_top1", 0),
                            "knn/woof_top1": knn_metrics_woof.get("knn_top1", 0),
                            "train/loss": train_metrics.get("loss", 0),
                            "train/sigreg": train_metrics.get("sigreg_loss", 0),
                            "train/prediction": train_metrics.get("prediction_loss", 0),
                            "epoch_attn/entropy": 0,  # Will be filled if available
                            "rep/effective_dim": 0,
                            "rep/isotropy": 0,
                        }

                        # Update metric history
                        for key in metric_history:
                            if key in current_epoch_metrics:
                                metric_history[key].append(current_epoch_metrics[key])

                        # Health checks
                        if getattr(
                            config, "health_check_enabled", True
                        ) and _every_or_first(
                            epoch, getattr(config, "health_check_interval", 1)
                        ):
                            # Get representation stats from a small batch
                            with torch.no_grad():
                                if diag_images is not None:
                                    rep_out = model.encoder(diag_images.to(device))
                                    if isinstance(rep_out, tuple):
                                        rep_out = rep_out[0]
                                    pooled = rep_out.mean(dim=1)
                                    rep_stats = compute_representation_stats(pooled)
                                    current_epoch_metrics["rep/effective_dim"] = (
                                        rep_stats.get("effective_dim", 0)
                                    )
                                    current_epoch_metrics["rep/isotropy"] = (
                                        rep_stats.get("isotropy", 0)
                                    )

                                    # Check representation health
                                    rep_health = check_representation_health(
                                        effective_dim=rep_stats.get("effective_dim"),
                                        isotropy=rep_stats.get("isotropy"),
                                        norm_mean=rep_stats.get("norm_mean"),
                                        norm_std=rep_stats.get("norm_std"),
                                        variance=rep_stats.get("variance"),
                                    )

                                    # Check gradient health
                                    grad_norms = compute_global_norms(model)
                                    grad_health = check_gradient_health(
                                        grad_norm=grad_norms.get("grad_norm"),
                                        param_norm=grad_norms.get("param_norm"),
                                    )

                                    # Check attention health (simplified)
                                    attn_health = check_attention_health(
                                        entropy_mean=current_epoch_metrics.get(
                                            "epoch_attn/entropy", 2.0
                                        ),
                                    )

                                    # Check training health
                                    training_health = check_training_health(
                                        loss=train_metrics.get("loss"),
                                        accuracy=val_metrics_nette.get(
                                            "val_accuracy", 0
                                        )
                                        / 100
                                        if val_metrics_nette
                                        else 0,
                                        epoch=epoch,
                                    )

                                    # Generate health summary HTML
                                    health_html = generate_health_summary_html(
                                        rep_health,
                                        attn_health,
                                        grad_health,
                                        training_health,
                                        epoch=epoch,
                                        metrics=current_epoch_metrics,
                                    )

                                    # Compute overall health score
                                    overall_score = compute_overall_health_score(
                                        rep_health, attn_health, grad_health
                                    )

                                    # Log health metrics
                                    wandb.log(
                                        {
                                            "health/rep_status": rep_health.status.value,
                                            "health/attn_status": attn_health.status.value,
                                            "health/grad_status": grad_health.status.value,
                                            "health/training_status": training_health.status.value,
                                            "health/overall_score": overall_score,
                                            "health/rep_score": rep_health.score,
                                            "health/attn_score": attn_health.score,
                                            "health/grad_score": grad_health.score,
                                            "health/recommendation": wandb.Html(
                                                health_html
                                            ),
                                        },
                                        commit=False,
                                    )

                                try:
                                    print("  Debug: Generating epoch summary dashboard...")
                                    dashboard = generate_epoch_summary_dashboard(
                                        epoch=epoch,
                                        metrics=current_epoch_metrics,
                                        history=metric_history,
                                    )
                                    wandb.log(
                                        {
                                            "summary/dashboard": wandb.Image(
                                                dashboard, caption=f"Epoch {epoch}"
                                            )
                                        },
                                        commit=False,
                                    )
                                except Exception as e:
                                    print(f"    Summary dashboard failed: {e}")

                                # Epoch delta dashboard
                                if epoch > 1 and _every_or_first(
                                    epoch, getattr(config, "epoch_delta_interval", 1)
                                ):
                                    try:
                                        print("  Debug: Generating epoch delta dashboard...")
                                        delta_dashboard = generate_epoch_delta_dashboard(
                                            current_epoch=epoch,
                                            current_metrics=current_epoch_metrics,
                                            previous_metrics=prev_epoch_metrics,
                                        )
                                        wandb.log(
                                            {
                                                "summary/epoch_delta": wandb.Image(
                                                    delta_dashboard, caption=f"Epoch {epoch}"
                                                )
                                            },
                                            commit=False,
                                        )
                                    except Exception as e:
                                        print(f"    Delta dashboard failed: {e}")

                        # Training radar chart
                        if _every_or_first(
                            epoch, getattr(config, "training_radar_interval", 5)
                        ):
                            try:
                                radar = generate_training_radar(
                                    current_metrics=current_epoch_metrics,
                                    target_metrics={
                                        "val_accuracy": 0.90,
                                        "knn_accuracy": 0.85,
                                        "eff_dimension": 0.80,
                                        "attn_diversity": 0.70,
                                        "grad_health": 0.80,
                                        "head_specialization": 0.70,
                                    },
                                )
                                wandb.log(
                                    {
                                        "summary/progress_radar": wandb.Image(
                                            radar, caption=f"Epoch {epoch}"
                                        )
                                    },
                                    commit=False,
                                )
                            except Exception as e:
                                print(f"  Radar chart failed: {e}")

                        # Pattern classification
                        if diag_attn_maps is not None and _every_or_first(
                            epoch, getattr(config, "pattern_classification_interval", 5)
                        ):
                            try:
                                pattern_result = classify_attention_patterns(
                                    diag_attn_maps
                                )
                                wandb.log(
                                    {
                                        "attn/pattern_parallel": pattern_result.pattern_distribution.get(
                                            "parallel", 0
                                        ),
                                        "attn/pattern_radioactive": pattern_result.pattern_distribution.get(
                                            "radioactive", 0
                                        ),
                                        "attn/pattern_homogeneous": pattern_result.pattern_distribution.get(
                                            "homogeneous", 0
                                        ),
                                        "attn/pattern_xtype": pattern_result.pattern_distribution.get(
                                            "xtype", 0
                                        ),
                                        "attn/pattern_compound": pattern_result.pattern_distribution.get(
                                            "compound", 0
                                        ),
                                        "attn/pattern_dominant": pattern_result.dominant_pattern,
                                    },
                                    commit=False,
                                )

                                # Pattern distribution chart
                                pattern_chart = generate_pattern_distribution_chart(
                                    pattern_result.pattern_counts
                                )
                                wandb.log(
                                    {
                                        "attn/pattern_distribution": wandb.Image(
                                            pattern_chart
                                        )
                                    },
                                    commit=False,
                                )
                            except Exception as e:
                                print(f"  Pattern classification failed: {e}")

                        # Head specialization analysis
                        if diag_attn_maps is not None and _every_or_first(
                            epoch,
                            getattr(config, "head_specialization_analysis_interval", 5),
                        ):
                            try:
                                spec_result = compute_head_specialization(
                                    diag_attn_maps
                                )
                                wandb.log(
                                    {
                                        "arch/head_redundancy": spec_result.redundancy_score,
                                        "arch/diversity_score": spec_result.diversity_score,
                                    },
                                    commit=False,
                                )
                            except Exception as e:
                                print(f"  Head specialization analysis failed: {e}")

                        # Q-Score computation
                        if _every_or_first(
                            epoch, getattr(config, "qscore_interval", 10)
                        ):
                            try:
                                # Fetch a batch with labels for Q-Score
                                batch_data = next(iter(train_loader))
                                if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                                    qscore_crops, qscore_labels = batch_data[0], batch_data[1]
                                else:
                                    qscore_crops, qscore_labels = batch_data, None
                                qscore_max = getattr(config, "qscore_max_samples", 1024)
                                qscore_crops = [c[:qscore_max] for c in qscore_crops]
                                qscore_labels = qscore_labels[:qscore_max]
                                qscore_images = qscore_crops[0].to(device)

                                with torch.no_grad():
                                    qscore_out = model.encoder(qscore_images)
                                    if isinstance(qscore_out, tuple):
                                        qscore_out = qscore_out[0]
                                    qscore_embeddings = qscore_out.mean(dim=1)

                                qscore_result = compute_representation_qscore(
                                    qscore_embeddings, qscore_labels.to(device)
                                )
                                wandb.log(
                                    {
                                        "rep/qscore_mean": qscore_result.mean_qscore,
                                        "rep/qscore_std": qscore_result.std_qscore,
                                        "rep/qscore_min": qscore_result.min_qscore,
                                        "rep/qscore_max": qscore_result.max_qscore,
                                    },
                                    commit=False,
                                )
                                del (
                                    qscore_crops,
                                    qscore_labels,
                                    qscore_images,
                                    qscore_out,
                                    qscore_embeddings,
                                )
                            except Exception as e:
                                print(f"  Q-Score computation failed: {e}")

                        # JiT-specific metrics
                        if config.encoder == "jit":
                            # RoPE position correlation
                            if diag_attn_maps is not None and _every_or_first(
                                epoch, getattr(config, "rope_correlation_interval", 10)
                            ):
                                try:
                                    num_patches = config.img_size // config.patch_size
                                    rope_corr = compute_rope_position_correlation(
                                        diag_attn_maps, (num_patches, num_patches)
                                    )
                                    wandb.log(
                                        {"arch/rope_position_corr": rope_corr},
                                        commit=False,
                                    )
                                except Exception as e:
                                    print(f"  RoPE correlation failed: {e}")

                            # SwiGLU sparsity (requires MLP activations - use proxy from grad stats)
                            if _every_or_first(
                                epoch, getattr(config, "swiglu_sparsity_interval", 10)
                            ):
                                try:
                                    # Estimate sparsity from MLP weight statistics as proxy
                                    swiglu_stats = {
                                        "mean_sparsity": 0.0,
                                        "per_layer_sparsity": [],
                                    }
                                    for i, blk in enumerate(model.encoder.blocks):
                                        if hasattr(blk, "mlp"):
                                            mlp = blk.mlp
                                            if hasattr(mlp, "w1") and hasattr(
                                                mlp, "w2"
                                            ):
                                                # SwiGLU: approximate activation sparsity
                                                with torch.no_grad():
                                                    w1_abs = (
                                                        mlp.w1.weight.abs()
                                                        .mean()
                                                        .item()
                                                    )
                                                    w2_abs = (
                                                        mlp.w2.weight.abs()
                                                        .mean()
                                                        .item()
                                                    )
                                                    # Gate ratio as sparsity proxy
                                                    sparsity = w2_abs / (w1_abs + 1e-8)
                                                    sparsity = min(1.0, sparsity)
                                                swiglu_stats[
                                                    "per_layer_sparsity"
                                                ].append(sparsity)
                                    if swiglu_stats["per_layer_sparsity"]:
                                        swiglu_stats["mean_sparsity"] = sum(
                                            swiglu_stats["per_layer_sparsity"]
                                        ) / len(swiglu_stats["per_layer_sparsity"])
                                    wandb.log(
                                        {
                                            "arch/swiglu_sparsity": swiglu_stats[
                                                "mean_sparsity"
                                            ]
                                        },
                                        commit=False,
                                    )
                                except Exception as e:
                                    print(f"  SwiGLU sparsity failed: {e}")

                        # Layer importance analysis
                        if _every_or_first(
                            epoch, getattr(config, "layer_importance_interval", 10)
                        ):
                            try:
                                # Compute layer importance from block output norms
                                layer_importance_scores = []
                                with torch.no_grad():
                                    if diag_images is not None:
                                        x = diag_images.to(device)
                                        # Forward through patch embed
                                        if hasattr(model.encoder, "patch_embed"):
                                            x = model.encoder.patch_embed(x)
                                        elif hasattr(
                                            model.encoder, "bottleneck_patch_embed"
                                        ):
                                            x = model.encoder.bottleneck_patch_embed(x)

                                        # Add pos embed if exists
                                        if (
                                            hasattr(model.encoder, "pos_embed")
                                            and model.encoder.pos_embed is not None
                                        ):
                                            x = x + model.encoder.pos_embed

                                        # Collect layer output norms
                                        prev_norm = x.norm(dim=-1).mean().item()
                                        for blk in model.encoder.blocks:
                                            x = blk(x)
                                            curr_norm = x.norm(dim=-1).mean().item()
                                            # Importance = relative change in norm
                                            importance = abs(curr_norm - prev_norm) / (
                                                prev_norm + 1e-8
                                            )
                                            layer_importance_scores.append(importance)
                                            prev_norm = curr_norm

                                if layer_importance_scores:
                                    # Normalize to sum to 1
                                    total = sum(layer_importance_scores) + 1e-8
                                    layer_importance_scores = [
                                        s / total for s in layer_importance_scores
                                    ]

                                    # Depth utilization: how evenly distributed is importance
                                    import numpy as np

                                    scores_arr = np.array(layer_importance_scores)
                                    uniform = np.ones_like(scores_arr) / len(scores_arr)
                                    # KL divergence from uniform (lower = more uniform)
                                    kl_div = np.sum(
                                        scores_arr * np.log(scores_arr / uniform + 1e-8)
                                    )
                                    depth_util = 1.0 / (
                                        1.0 + kl_div
                                    )  # Transform to 0-1 scale

                                    wandb.log(
                                        {
                                            "arch/depth_utilization": depth_util,
                                            "arch/layer_importance_std": float(
                                                np.std(scores_arr)
                                            ),
                                        },
                                        commit=False,
                                    )

                                    # Generate layer importance curve
                                    importance_tensor = torch.tensor(
                                        layer_importance_scores
                                    )
                                    importance_curve = generate_layer_importance_curve(
                                        importance_tensor
                                    )
                                    wandb.log(
                                        {
                                            "arch/layer_importance_curve": wandb.Image(
                                                importance_curve,
                                                caption=f"Epoch {epoch}",
                                            )
                                        },
                                        commit=False,
                                    )
                            except Exception as e:
                                print(f"  Layer importance analysis failed: {e}")

                        # Store current metrics for next epoch delta
                        if "current_epoch_metrics" in locals():
                            prev_epoch_metrics = current_epoch_metrics.copy()
                        else:
                            print("Warning: current_epoch_metrics not found, skipping history update.")

                    except Exception as e:
                        print(f"Production dashboard failed: {e}")
                    finally:
                        _cleanup_gpu_memory(device)

            except Exception as e:
                print(f"Error generating visualization: {e}")

            # Ensure we don't keep large cached tensors from visualization passes.
            for blk in getattr(model.encoder, "blocks", []):
                attn = getattr(blk, "attn", None)
                if attn is None:
                    continue
                if hasattr(attn, "output_attention"):
                    attn.output_attention = False
                if hasattr(attn, "output_attn_logits"):
                    attn.output_attn_logits = False
                if hasattr(attn, "attn_map"):
                    attn.attn_map = None
                if hasattr(attn, "attn_logits"):
                    attn.attn_logits = None
                if hasattr(attn, "head_mask"):
                    attn.head_mask = None

            import gc

            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Save checkpoint
        if epoch % config.save_interval == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "probe_state_dict": probe.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": vars(config),
                },
                output_dir / f"checkpoint_epoch_{epoch}.pt",
            )

    # Final evaluation
    print("\nFinal evaluation...")
    val_metrics_nette = evaluate(model, probe, val_loader_nette, device, config)
    val_metrics_woof = evaluate(model, probe, val_loader_woof, device, config)

    print(f"Final Val Accuracy (Nette): {val_metrics_nette['val_accuracy']:.2f}%")
    print(f"Final Val Top-5 (Nette): {val_metrics_nette.get('val_top5', 0):.2f}%")
    print(f"Final Val Loss (Nette): {val_metrics_nette.get('val_loss', 0):.4f}")
    print(f"Final Val Accuracy (Woof): {val_metrics_woof['val_accuracy']:.2f}%")
    print(f"Final Val Top-5 (Woof): {val_metrics_woof.get('val_top5', 0):.2f}%")
    print(f"Final Val Loss (Woof): {val_metrics_woof.get('val_loss', 0):.4f}")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")

    # Save final model
    torch.save(
        {
            "epoch": config.epochs,
            "model_state_dict": model.state_dict(),
            "probe_state_dict": probe.state_dict(),
            "val_accuracy": val_metrics_nette.get("val_accuracy", 0),
            "val_accuracy_woof": val_metrics_woof.get("val_accuracy", 0),
            "best_val_accuracy": best_val_acc,
            "config": vars(config),
        },
        output_dir / "final_model.pt",
    )

    if config.use_wandb:
        import wandb

        # Log summary metrics for easy comparison in runs table
        wandb.run.summary["best_val_accuracy"] = best_val_acc
        wandb.run.summary["final_val_accuracy_nette"] = val_metrics_nette[
            "val_accuracy"
        ]
        wandb.run.summary["final_val_top5_nette"] = val_metrics_nette.get("val_top5", 0)
        wandb.run.summary["final_val_loss_nette"] = val_metrics_nette.get("val_loss", 0)
        wandb.run.summary["final_val_accuracy_woof"] = val_metrics_woof["val_accuracy"]
        wandb.run.summary["final_val_top5_woof"] = val_metrics_woof.get("val_top5", 0)
        wandb.run.summary["final_val_loss_woof"] = val_metrics_woof.get("val_loss", 0)
        wandb.finish()

    print(f"\nTraining complete! Models saved to {output_dir}")


if __name__ == "__main__":
    main()
