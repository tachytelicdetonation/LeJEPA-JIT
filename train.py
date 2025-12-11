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
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from config import Config, get_config
from data import get_dataloaders
from losses import LeJEPALoss
from models.lejepa import LinearProbe, create_lejepa


def get_schedulers(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> SequentialLR:
    """Create learning rate scheduler with warmup and cosine decay."""
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=1e-5
    )
    return SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )


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

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, (views, labels) in enumerate(pbar):
        views = views.to(device, non_blocking=True)  # (B, V, C, H, W)
        labels = labels.to(device, non_blocking=True)  # (B,)

        B, V = views.shape[:2]

        # Forward pass with mixed precision
        with autocast("cuda", enabled=config.mixed_precision, dtype=torch.bfloat16):
            # Get embeddings and projections
            emb, proj = model(views)  # emb: (B*V, D), proj: (V, B, D)

            # LeJEPA loss on projections
            loss_dict = loss_fn(proj)
            lejepa_loss = loss_dict["loss"]

            # Linear probe on embeddings (detached)
            # Labels need to be repeated for all views
            labels_rep = labels.repeat_interleave(V)
            probe_logits = probe(emb.detach())
            probe_loss = F.cross_entropy(probe_logits, labels_rep)

            # Combined loss
            loss = lejepa_loss + probe_loss

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Compute accuracy (on first view only)
        with torch.no_grad():
            emb_view0 = emb.view(B, V, -1)[:, 0]  # (B, D)
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

        # Update progress bar
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
        with autocast("cuda", enabled=config.mixed_precision, dtype=torch.bfloat16):
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
    overrides = {k: v for k, v in vars(args).items() if v is not None and k != "no_wandb"}
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
        except ImportError:
            print("wandb not installed, skipping logging")
            config.use_wandb = False

    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader = get_dataloaders(
        batch_size=config.batch_size,
        img_size=config.img_size,
        num_views=config.num_views,
        num_workers=config.num_workers,
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

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
        proj_dim=config.proj_dim,
        **jit_kwargs,
    ).to(device)

    # Create linear probe
    probe = LinearProbe(config.embed_dim, num_classes=10).to(device)

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
    warmup_steps = len(train_loader)  # 1 epoch warmup
    total_steps = len(train_loader) * config.epochs
    scheduler = get_schedulers(optimizer, warmup_steps, total_steps)

    # Create gradient scaler for mixed precision
    scaler = GradScaler(
        "cuda", enabled=config.mixed_precision and device.type == "cuda"
    )

    # Training loop
    print("\nStarting training...")
    print(
        f"Config: epochs={config.epochs}, batch_size={config.batch_size}, "
        f"num_views={config.num_views}, lambda={config.lambda_sigreg}"
    )
    best_val_acc = 0

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
        )

        # Evaluate
        if epoch % config.eval_interval == 0:
            val_metrics = evaluate(model, probe, val_loader, device, config)
            val_acc = val_metrics["val_accuracy"]

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Save best model
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "probe_state_dict": probe.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_accuracy": val_acc,
                        "config": vars(config),
                    },
                    output_dir / "best_model.pt",
                )
        else:
            val_metrics = {}

        # Log metrics
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch}/{config.epochs}")
        print(
            f"  Loss: {train_metrics['loss']:.4f} (SIGReg: {train_metrics['sigreg_loss']:.4f}, Inv: {train_metrics['invariance_loss']:.4f})"
        )
        print(f"  Train Acc: {train_metrics['accuracy']:.2f}%")
        if val_metrics:
            print(
                f"  Val Acc: {val_metrics['val_accuracy']:.2f}% (Best: {best_val_acc:.2f}%)"
            )
        print(f"  Time: {epoch_time:.1f}s")

        if config.use_wandb:
            import wandb

            wandb.log(
                {
                    "epoch": epoch,
                    **train_metrics,
                    **val_metrics,
                    "best_val_accuracy": best_val_acc,
                    "lr": scheduler.get_last_lr()[0],
                }
            )

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
    val_metrics = evaluate(model, probe, val_loader, device, config)
    print(f"Final Val Accuracy: {val_metrics['val_accuracy']:.2f}%")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")

    # Save final model
    torch.save(
        {
            "epoch": config.epochs,
            "model_state_dict": model.state_dict(),
            "probe_state_dict": probe.state_dict(),
            "val_accuracy": val_metrics["val_accuracy"],
            "best_val_accuracy": best_val_acc,
            "config": vars(config),
        },
        output_dir / "final_model.pt",
    )

    if config.use_wandb:
        import wandb

        wandb.finish()

    print(f"\nTraining complete! Models saved to {output_dir}")


if __name__ == "__main__":
    main()
