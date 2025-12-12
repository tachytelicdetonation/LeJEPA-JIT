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
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

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
)

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
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


def train_one_epoch(
    model: nn.Module,
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
) -> dict:
    """Train for one epoch."""
    model.train()
    probe.train()

    total_loss = 0
    total_sigreg = 0
    total_inv = 0
    total_probe_loss = 0
    total_correct = 0
    total_samples = 0
    total_entropy = 0
    total_gini = 0
    total_sparsity = 0
    total_grad_norm = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, (crops, labels) in enumerate(pbar):
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
            loss_dict = loss_fn(proj)
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

                # Compute metrics on last layer
                last_attn = attns[-1]
                current_entropy = compute_entropy(last_attn).item()
                current_gini = compute_gini(last_attn).item()
                current_sparsity = compute_sparsity(last_attn).item()

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Capture Grad Stats from hooks
        # attn_grads is a list populated by hooks
        current_grad_norm = 0.0
        if attn_grads:
            # Average norm over layers
            norms = [g.norm().item() for g in attn_grads]
            current_grad_norm = sum(norms) / len(norms)
            attn_grads.clear()  # Reset for next batch

        # Compute accuracy (on first global view only)
        with torch.no_grad():
            emb_view0 = emb_global.view(B, 2, -1)[:, 0]  # (B, D)
            logits_view0 = probe(emb_view0)
            pred = logits_view0.argmax(dim=1)
            correct = (pred == labels).sum().item()
            total_correct += correct
            total_samples += B

        # Accumulate losses
        total_loss += lejepa_loss.item()
        total_sigreg += loss_dict["sigreg_loss"].item()
        total_inv += loss_dict["invariance_loss"].item()
        total_probe_loss += probe_loss.item()
        if batch_idx % config.log_interval == 0:
            total_entropy += current_entropy
            total_gini += current_gini
            total_sparsity += current_sparsity
            total_grad_norm += current_grad_norm

        # Update progress bar and log to wandb
        if batch_idx % config.log_interval == 0:
            pbar.set_postfix(
                {
                    "loss": f"{lejepa_loss.item():.4f}",
                    "sigreg": f"{loss_dict['sigreg_loss'].item():.4f}",
                    "inv": f"{loss_dict['invariance_loss'].item():.4f}",
                    "acc": f"{100 * total_correct / total_samples:.1f}%",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )
            # Log batch metrics to wandb
            if config.use_wandb:
                import wandb

                global_step = (epoch - 1) * len(train_loader) + batch_idx
                wandb.log(
                    {
                        "step": global_step,
                        "train/loss": lejepa_loss.item(),
                        "train/sigreg": loss_dict["sigreg_loss"].item(),
                        "train/invariance": loss_dict["invariance_loss"].item(),
                        "train/probe_loss": probe_loss.item(),
                        "train/accuracy": 100 * total_correct / total_samples,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/attn_entropy": current_entropy,
                        "train/attn_gini": current_gini,
                        "train/attn_sparsity": current_sparsity,
                        "train/attn_grad_norm": current_grad_norm,
                    }
                )

    num_batches = len(train_loader)
    return {
        "loss": total_loss / num_batches,
        "sigreg_loss": total_sigreg / num_batches,
        "invariance_loss": total_inv / num_batches,
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

    for views, labels in tqdm(val_loader, desc="Evaluating"):
        views = views.to(device, non_blocking=True)  # (B, 1, C, H, W)
        labels = labels.to(device, non_blocking=True)

        # Get embeddings
        with _autocast_ctx(device, enabled=config.mixed_precision):
            emb, _ = model(views)
            logits = probe(emb)

        pred = logits.argmax(dim=1)
        total_correct += (pred == labels).sum().item()
        total_samples += labels.size(0)

    accuracy = 100 * total_correct / total_samples
    return {"val_accuracy": accuracy}


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

            wandb.init(
                project=config.wandb_project,
                config=vars(config),
                name=f"{config.encoder}-{time.strftime('%Y%m%d_%H%M%S')}",
            )
            # Define x-axes for different metric types
            wandb.define_metric("step")
            wandb.define_metric("epoch")
            wandb.define_metric("train/*", step_metric="step")
            wandb.define_metric("epoch_*", step_metric="epoch")
            wandb.define_metric("val_*", step_metric="epoch")
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
    scaler = GradScaler(
        "cuda", enabled=use_grad_scaler
    )

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

    # Get a batch from val_loader_nette
    try:
        iter_nette = iter(val_loader_nette)
        views, _ = next(iter_nette)
        vis_images_nette = views[:8, 0].to(device)
    except StopIteration:
        print("Warning: ImageNette Validation loader is empty.")

    # Get a batch from val_loader_woof
    try:
        iter_woof = iter(val_loader_woof)
        views, _ = next(iter_woof)
        vis_images_woof = views[:8, 0].to(device)
    except StopIteration:
        print("Warning: ImageWoof Validation loader is empty.")

    # Import visualization utils
    from utils.visualization import (
        generate_pca_visualization,
        generate_attention_rollout,
        generate_attention_grid,
        generate_layer_attention_evolution,
        generate_per_head_attention,
        generate_head_importance_heatmap,
        generate_token_similarity_heatmap,
        generate_rsm_across_layers,
        generate_gradient_flow_heatmap,
        generate_embedding_projection,
        generate_collapse_monitor,
        generate_training_dashboard,
        AttentionTracker,
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
    attn_tracker = AttentionTracker(max_snapshots=10)

    # History lists for training dashboard
    loss_history = []
    acc_history = []
    entropy_history = []
    gini_history = []
    lr_history = []

    for epoch in range(1, config.epochs + 1):
        start_time = time.time()

        # Train
        train_metrics = train_one_epoch(
            model=model,
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
                        "config": vars(config),
                    },
                    output_dir / "best_model.pt",
                )

        # Log metrics
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch}/{config.epochs}")
        print(
            f"  Loss: {train_metrics['loss']:.4f} (SIGReg: {train_metrics['sigreg_loss']:.4f}, Inv: {train_metrics['invariance_loss']:.4f})"
        )
        print(f"  Train Acc: {train_metrics['accuracy']:.2f}%")
        if val_metrics_nette:
            print(
                f"  Val Acc (Nette): {val_metrics_nette['val_accuracy']:.2f}% (Best: {best_val_acc:.2f}%)"
            )
            print(f"  Val Acc (Woof): {val_metrics_woof['val_accuracy']:.2f}%")
        print(f"  Time: {epoch_time:.1f}s")

        if config.use_wandb:
            import wandb

            wandb.log(
                {
                    "epoch": epoch,
                    "epoch_loss": train_metrics["loss"],
                    "epoch_sigreg_loss": train_metrics["sigreg_loss"],
                    "epoch_invariance_loss": train_metrics["invariance_loss"],
                    "epoch_probe_loss": train_metrics["probe_loss"],
                    "epoch_train_accuracy": train_metrics["accuracy"],
                    "val_accuracy_nette": val_metrics_nette.get("val_accuracy", 0),
                    "val_accuracy_woof": val_metrics_woof.get("val_accuracy", 0),
                    "best_val_accuracy": best_val_acc,
                }
            )

        # Update history for dashboard
        loss_history.append(train_metrics["loss"])
        acc_history.append(
            val_metrics_nette.get("val_accuracy", train_metrics["accuracy"])
        )
        lr_history.append(scheduler.get_last_lr()[0])

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

                # === Original Visualizations ===
                # 1. ImageNette Vis
                if vis_images_nette is not None:
                    vis_grid_nette = generate_pca_visualization(
                        model,
                        vis_images_nette,
                        device,
                        img_size=config.img_size,
                        patch_size=config.patch_size,
                    )
                    vis_frames_nette.append(vis_grid_nette)
                    if config.use_wandb:
                        wandb.log(
                            {
                                "pca_vis_nette": wandb.Image(
                                    vis_grid_nette, caption=f"Epoch {epoch} (Nette)"
                                )
                            },
                            commit=False,
                        )
                        if len(vis_frames_nette) > 1:
                            log_gif(vis_frames_nette, "pca_progression_nette", "nette")

                    # Attention Rollout
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
                                "attn_vis_nette": wandb.Image(
                                    attn_grid_nette,
                                    caption=f"Epoch {epoch} (Nette Attn)",
                                )
                            },
                            commit=False,
                        )
                        if len(attn_frames_nette) > 1:
                            log_gif(
                                attn_frames_nette,
                                "attn_progression_nette",
                                "attn_nette",
                            )

                    # Attention Grid (Raw)
                    grid_raw_nette = generate_attention_grid(
                        model,
                        vis_images_nette,
                        device,
                        img_size=config.img_size,
                        patch_size=config.patch_size,
                    )
                    if config.use_wandb:
                        wandb.log(
                            {
                                "attn_grid_nette": wandb.Image(
                                    grid_raw_nette,
                                    caption=f"Epoch {epoch} (Nette Grid)",
                                )
                            },
                            commit=False,
                        )

                # 2. ImageWoof Vis
                if vis_images_woof is not None:
                    vis_grid_woof = generate_pca_visualization(
                        model,
                        vis_images_woof,
                        device,
                        img_size=config.img_size,
                        patch_size=config.patch_size,
                    )
                    vis_frames_woof.append(vis_grid_woof)
                    if config.use_wandb:
                        wandb.log(
                            {
                                "pca_vis_woof": wandb.Image(
                                    vis_grid_woof, caption=f"Epoch {epoch} (Woof)"
                                )
                            },
                            commit=False,
                        )
                        if len(vis_frames_woof) > 1:
                            log_gif(vis_frames_woof, "pca_progression_woof", "woof")

                    # Attention Rollout
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
                                "attn_vis_woof": wandb.Image(
                                    attn_grid_woof, caption=f"Epoch {epoch} (Woof Attn)"
                                )
                            },
                            commit=False,
                        )
                        if len(attn_frames_woof) > 1:
                            log_gif(
                                attn_frames_woof, "attn_progression_woof", "attn_woof"
                            )

                    # Attention Grid (Raw)
                    grid_raw_woof = generate_attention_grid(
                        model,
                        vis_images_woof,
                        device,
                        img_size=config.img_size,
                        patch_size=config.patch_size,
                    )
                    if config.use_wandb:
                        wandb.log(
                            {
                                "attn_grid_woof": wandb.Image(
                                    grid_raw_woof, caption=f"Epoch {epoch} (Woof Grid)"
                                )
                            },
                            commit=False,
                        )

                # === NEW VISUALIZATIONS: Training & Attention Dynamics ===
                vis_images = (
                    vis_images_nette
                    if vis_images_nette is not None
                    else vis_images_woof
                )

                # 3. Layer-wise Attention Evolution (every 5 epochs)
                if epoch % config.layer_attention_interval == 0:
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
                                "layer_attention_evolution": wandb.Image(
                                    layer_attn_vis, caption=f"Epoch {epoch}"
                                )
                            },
                            commit=False,
                        )

                # 4. Per-Head Attention (every 10 epochs)
                if epoch % config.per_head_attention_interval == 0:
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
                                "per_head_attention": wandb.Image(
                                    per_head_vis, caption=f"Epoch {epoch} Last Layer"
                                )
                            },
                            commit=False,
                        )

                # 5. Head Importance Heatmap (every 10 epochs)
                if epoch % config.head_importance_interval == 0:
                    head_importance_vis = generate_head_importance_heatmap(
                        model, vis_images, device
                    )
                    if config.use_wandb:
                        wandb.log(
                            {
                                "head_importance": wandb.Image(
                                    head_importance_vis, caption=f"Epoch {epoch}"
                                )
                            },
                            commit=False,
                        )

                # 6. Token Similarity Heatmap (every 10 epochs)
                if epoch % config.token_similarity_interval == 0:
                    token_sim_vis = generate_token_similarity_heatmap(
                        model, vis_images, device, sample_idx=0
                    )
                    if config.use_wandb:
                        wandb.log(
                            {
                                "token_similarity": wandb.Image(
                                    token_sim_vis, caption=f"Epoch {epoch}"
                                )
                            },
                            commit=False,
                        )

                # 7. RSM Across Layers (every 10 epochs)
                if epoch % config.rsm_interval == 0:
                    rsm_vis = generate_rsm_across_layers(model, vis_images, device)
                    if config.use_wandb:
                        wandb.log(
                            {
                                "rsm_layers": wandb.Image(
                                    rsm_vis, caption=f"Epoch {epoch}"
                                )
                            },
                            commit=False,
                        )

                # 8. Capture Attention for Difference Tracking
                attn_tracker.capture(model, vis_images, device, epoch)

                # 9. Attention Difference Visualization (after epoch 2)
                if epoch > 1:
                    attn_diff_vis = attn_tracker.generate_difference_visualization(
                        img_size=config.img_size, patch_size=config.patch_size
                    )
                    if attn_diff_vis is not None and config.use_wandb:
                        wandb.log(
                            {
                                "attention_difference": wandb.Image(
                                    attn_diff_vis, caption=f"Epoch 1 → {epoch}"
                                )
                            },
                            commit=False,
                        )

                # 10. Feature Collapse Monitor (every 5 epochs)
                if epoch % config.collapse_monitor_interval == 0:
                    with torch.no_grad():
                        emb_sample, _ = model(vis_images.unsqueeze(1))
                    collapse_vis = generate_collapse_monitor(emb_sample)
                    if config.use_wandb:
                        wandb.log(
                            {
                                "collapse_monitor": wandb.Image(
                                    collapse_vis, caption=f"Epoch {epoch}"
                                )
                            },
                            commit=False,
                        )

                    # Also compute and log collapse metrics
                    collapse_metrics = compute_feature_collapse_metrics(emb_sample)
                    if config.use_wandb:
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
                            },
                            commit=False,
                        )

                # 11. Gradient Flow Heatmap (every 5 epochs, after backward)
                if epoch % config.gradient_flow_interval == 0:
                    grad_stats = compute_layer_gradient_stats(model)
                    grad_flow_vis = generate_gradient_flow_heatmap(grad_stats)
                    if config.use_wandb:
                        wandb.log(
                            {
                                "gradient_flow": wandb.Image(
                                    grad_flow_vis, caption=f"Epoch {epoch}"
                                )
                            },
                            commit=False,
                        )

                # 12. Compute and log attention entropy/gini for history
                with torch.no_grad():
                    for blk in model.encoder.blocks:
                        blk.attn.output_attention = True
                    _ = model.encoder(vis_images)
                    attns = model.encoder.get_attention_maps()
                    for blk in model.encoder.blocks:
                        blk.attn.output_attention = False
                    if attns:
                        last_attn = attns[-1]
                        entropy_history.append(compute_entropy(last_attn).item())
                        gini_history.append(compute_gini(last_attn).item())

                        # Log head diversity metrics
                        head_div = compute_head_diversity(last_attn)
                        if config.use_wandb:
                            wandb.log(
                                {
                                    "attention/head_similarity": head_div[
                                        "head_similarity"
                                    ],
                                    "attention/head_variance": head_div[
                                        "head_variance"
                                    ],
                                    "attention/effective_heads": head_div[
                                        "effective_heads"
                                    ],
                                },
                                commit=False,
                            )

                # 13. Training Dashboard (every 10 epochs)
                if epoch % config.training_dashboard_interval == 0 and len(loss_history) > 1:
                    dashboard = generate_training_dashboard(
                        loss_history,
                        acc_history,
                        entropy_history,
                        gini_history,
                        lr_history,
                    )
                    if config.use_wandb:
                        wandb.log(
                            {
                                "training_dashboard": wandb.Image(
                                    dashboard, caption=f"Epoch {epoch}"
                                )
                            },
                            commit=False,
                        )

                # 14. t-SNE Embedding Projection (every 20 epochs - expensive)
                if epoch % config.embedding_projection_interval == 0:
                    try:
                        tsne_vis = generate_embedding_projection(
                            model,
                            val_loader_nette,
                            device,
                            method="tsne",
                            max_samples=300,
                        )
                        if config.use_wandb:
                            wandb.log(
                                {
                                    "embedding_tsne": wandb.Image(
                                        tsne_vis, caption=f"Epoch {epoch}"
                                    )
                                },
                                commit=False,
                            )
                    except Exception as e:
                        print(f"t-SNE visualization failed: {e}")

            except Exception as e:
                print(f"Error generating visualization: {e}")

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
    print(f"Final Val Accuracy (Woof): {val_metrics_woof['val_accuracy']:.2f}%")
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
        wandb.run.summary["final_val_accuracy_woof"] = val_metrics_woof["val_accuracy"]
        wandb.finish()

    print(f"\nTraining complete! Models saved to {output_dir}")


if __name__ == "__main__":
    main()
