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
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from config import Config, get_config
from models import LeJEPA, JiTEncoder, ViTEncoder
from models.lejepa import LinearProbe, create_lejepa
from losses import LeJEPALoss
from data import get_dataloaders


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create learning rate scheduler with warmup and cosine decay."""
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: LeJEPA,
    probe: LinearProbe,
    loss_fn: LeJEPALoss,
    optimizer: torch.optim.Optimizer,
    probe_optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
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

    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)  # (B, V, C, H, W)
        labels = batch["label"].to(device)  # (B,)

        B, V = images.shape[:2]

        # Forward pass with mixed precision
        with autocast(enabled=config.mixed_precision, dtype=torch.bfloat16):
            # Get embeddings and projections
            embeddings, projections = model(images)  # (B*V, dim)

            # LeJEPA loss on projections
            loss_dict = loss_fn(projections, embeddings)
            loss = loss_dict["loss"]

            # Linear probe on embeddings (detached)
            # Use first view only for probe
            emb_view0 = embeddings.view(B, V, -1)[:, 0].detach()  # (B, embed_dim)
            probe_logits = probe(emb_view0)
            probe_loss = F.cross_entropy(probe_logits, labels)

        # Backward pass for main loss
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)

        # Backward pass for probe
        probe_optimizer.zero_grad()
        scaler.scale(probe_loss).backward()
        scaler.step(probe_optimizer)

        scaler.update()
        scheduler.step()

        # Compute accuracy
        with torch.no_grad():
            pred = probe_logits.argmax(dim=1)
            correct = (pred == labels).sum().item()
            total_correct += correct
            total_samples += B

        # Accumulate losses
        total_loss += loss.item()
        total_sigreg += loss_dict["sigreg_loss"].item()
        total_inv += loss_dict["invariance_loss"].item()
        total_probe_loss += probe_loss.item()

        # Update progress bar
        if batch_idx % config.log_interval == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "sigreg": f"{loss_dict['sigreg_loss'].item():.4f}",
                "inv": f"{loss_dict['invariance_loss'].item():.4f}",
                "acc": f"{100 * total_correct / total_samples:.1f}%",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

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
    model: LeJEPA,
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

    for batch in tqdm(val_loader, desc="Evaluating"):
        images = batch["image"].to(device)  # (B, C, H, W)
        labels = batch["label"].to(device)

        # Get embeddings
        with autocast(enabled=config.mixed_precision, dtype=torch.bfloat16):
            embeddings = model.get_embedding(images)
            logits = probe(embeddings)

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
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--lr_probe", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--lambda_sigreg", type=float, default=0.5)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="lejepa-jit")
    args = parser.parse_args()

    # Create config
    config = get_config(
        encoder=args.encoder,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_encoder=args.lr,
        lr_probe=args.lr_probe,
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        lambda_sigreg=args.lambda_sigreg,
        warmup_epochs=args.warmup_epochs,
        num_workers=args.num_workers,
        seed=args.seed,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
    )

    # Set seed
    set_seed(config.seed)

    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(config.output_dir) / f"{config.encoder}_{time.strftime('%Y%m%d_%H%M%S')}"
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
    model = create_lejepa(
        encoder_type=config.encoder,
        img_size=config.img_size,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        proj_hidden_dim=config.proj_hidden_dim,
        proj_dim=config.proj_dim,
        bottleneck_dim=config.bottleneck_dim if config.encoder == "jit" else None,
    ).to(device)

    # Create linear probe
    probe = LinearProbe(config.embed_dim, num_classes=10).to(device)

    # Print model info
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Encoder parameters: {count_parameters(model.encoder):,}")
    print(f"Projector parameters: {count_parameters(model.projector):,}")
    print(f"Probe parameters: {count_parameters(probe):,}")

    # Create loss function
    loss_fn = LeJEPALoss(
        lambda_sigreg=config.lambda_sigreg,
        num_knots=config.sigreg_num_knots,
        max_t=config.sigreg_max_t,
        num_views=config.num_views,
    )

    # Create optimizers
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr_encoder,
        weight_decay=config.weight_decay,
    )

    probe_optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=config.lr_probe,
        weight_decay=config.weight_decay,
    )

    # Create scheduler
    scheduler = get_lr_scheduler(
        optimizer,
        warmup_epochs=config.warmup_epochs,
        total_epochs=config.epochs,
        steps_per_epoch=len(train_loader),
    )

    # Create gradient scaler for mixed precision
    scaler = GradScaler(enabled=config.mixed_precision)

    # Training loop
    print("\nStarting training...")
    best_val_acc = 0

    for epoch in range(1, config.epochs + 1):
        start_time = time.time()

        # Train
        train_metrics = train_one_epoch(
            model=model,
            probe=probe,
            loss_fn=loss_fn,
            optimizer=optimizer,
            probe_optimizer=probe_optimizer,
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
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "probe_state_dict": probe.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_acc,
                    "config": vars(config),
                }, output_dir / "best_model.pt")
        else:
            val_metrics = {}

        # Log metrics
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch}/{config.epochs}")
        print(f"  Loss: {train_metrics['loss']:.4f} (SIGReg: {train_metrics['sigreg_loss']:.4f}, Inv: {train_metrics['invariance_loss']:.4f})")
        print(f"  Train Acc: {train_metrics['accuracy']:.2f}%")
        if val_metrics:
            print(f"  Val Acc: {val_metrics['val_accuracy']:.2f}% (Best: {best_val_acc:.2f}%)")
        print(f"  Time: {epoch_time:.1f}s")

        if config.use_wandb:
            wandb.log({
                "epoch": epoch,
                **train_metrics,
                **val_metrics,
                "best_val_accuracy": best_val_acc,
                "lr": scheduler.get_last_lr()[0],
            })

        # Save checkpoint
        if epoch % config.save_interval == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "probe_state_dict": probe.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": vars(config),
            }, output_dir / f"checkpoint_epoch_{epoch}.pt")

    # Final evaluation
    print("\nFinal evaluation...")
    val_metrics = evaluate(model, probe, val_loader, device, config)
    print(f"Final Val Accuracy: {val_metrics['val_accuracy']:.2f}%")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")

    # Save final model
    torch.save({
        "epoch": config.epochs,
        "model_state_dict": model.state_dict(),
        "probe_state_dict": probe.state_dict(),
        "val_accuracy": val_metrics["val_accuracy"],
        "best_val_accuracy": best_val_acc,
        "config": vars(config),
    }, output_dir / "final_model.pt")

    if config.use_wandb:
        wandb.finish()

    print(f"\nTraining complete! Models saved to {output_dir}")


if __name__ == "__main__":
    main()
