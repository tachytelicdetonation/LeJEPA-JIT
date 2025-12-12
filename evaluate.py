"""
Evaluation script for LeJEPA models.

This script evaluates trained models using:
1. Linear probe accuracy (train a linear classifier on frozen embeddings)
2. k-NN accuracy (non-parametric nearest neighbor classification)

Usage:
    # Evaluate a saved checkpoint
    python evaluate.py --checkpoint outputs/jit_xxx/best_model.pt

    # Compare JiT vs ViT
    python evaluate.py --checkpoint_jit outputs/jit_xxx/best_model.pt \
                       --checkpoint_vit outputs/vit_xxx/best_model.pt
"""

import argparse
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from models.lejepa import create_lejepa
from data import get_dataloaders


def _amp_dtype_for_device(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.bfloat16


def _autocast_ctx(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=_amp_dtype_for_device(device))


@torch.no_grad()
def extract_features(
    model,
    dataloader,
    device: torch.device,
    mixed_precision: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract features from all images in dataloader."""
    model.eval()

    all_features = []
    all_labels = []

    for views, labels in tqdm(dataloader, desc="Extracting features"):
        views = views.to(device)
        labels = labels if torch.is_tensor(labels) else torch.tensor(labels)

        with _autocast_ctx(device, enabled=mixed_precision):
            features = model.get_embedding(views)

        all_features.append(features.cpu())
        all_labels.append(labels.cpu())

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return features, labels


def train_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    num_classes: int = 10,
    epochs: int = 100,
    lr: float = 0.01,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Train and evaluate a linear probe on extracted features."""
    # Move to device
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)

    # Create linear probe
    embed_dim = train_features.shape[1]
    probe = nn.Linear(embed_dim, num_classes).to(device)

    # Optimizer
    optimizer = torch.optim.SGD(
        probe.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_acc = 0
    batch_size = 256

    for epoch in range(epochs):
        probe.train()
        indices = torch.randperm(len(train_features))

        for i in range(0, len(train_features), batch_size):
            batch_idx = indices[i : i + batch_size]
            features = train_features[batch_idx]
            labels = train_labels[batch_idx]

            logits = probe(features)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Evaluate
        probe.eval()
        with torch.no_grad():
            logits = probe(val_features)
            pred = logits.argmax(dim=1)
            acc = (pred == val_labels).float().mean().item() * 100

        if acc > best_acc:
            best_acc = acc

    return best_acc


@torch.no_grad()
def knn_accuracy(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    k: int = 20,
) -> float:
    """Compute k-NN accuracy."""
    # Normalize features
    train_features = F.normalize(train_features, dim=1)
    val_features = F.normalize(val_features, dim=1)

    # Compute similarities
    similarities = val_features @ train_features.T  # (num_val, num_train)

    # Get top-k neighbors
    _, indices = similarities.topk(k, dim=1)  # (num_val, k)

    # Get neighbor labels
    neighbor_labels = train_labels[indices]  # (num_val, k)

    # Vote
    pred = torch.mode(neighbor_labels, dim=1).values

    # Accuracy
    acc = (pred == val_labels).float().mean().item() * 100

    return acc


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_dict = checkpoint["config"]

    # Create model
    encoder_type = config_dict["encoder"]
    encoder_kwargs = {}
    if encoder_type == "jit":
        encoder_kwargs["bottleneck_dim"] = config_dict.get("bottleneck_dim", 128)

    model = create_lejepa(
        encoder_type=encoder_type,
        img_size=config_dict["img_size"],
        patch_size=config_dict["patch_size"],
        embed_dim=config_dict["embed_dim"],
        depth=config_dict["depth"],
        num_heads=config_dict["num_heads"],
        proj_hidden_dim=config_dict.get("proj_hidden_dim", 2048),
        proj_dim=config_dict["proj_dim"],
        **encoder_kwargs,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, config_dict


def evaluate_model(
    checkpoint_path: str,
    device: torch.device,
    num_workers: int = 8,
) -> dict:
    """Evaluate a single model."""
    print(f"\nLoading model from {checkpoint_path}")
    model, config = load_model(checkpoint_path, device)

    print(f"Encoder: {config['encoder'].upper()}")
    print(f"Embed dim: {config['embed_dim']}")

    # Create dataloaders (matches training pipeline: ImageNette + ImageWoof)
    _, val_loader_nette, val_loader_woof = get_dataloaders(
        batch_size=256,
        img_size=config["img_size"],
        num_workers=num_workers,
    )

    # For linear probe training, we want a standard (non-multicrop) transform.
    # Build a combined train dataset (ImageNette + ImageWoof) with val transforms.
    from data import ImageNetteDataset
    from torch.utils.data import DataLoader
    import torch.utils.data

    train_dataset_nette = ImageNetteDataset(
        split="train",
        img_size=config["img_size"],
        is_training=False,  # Use validation transform
        class_offset=0,
    )

    # The ImageNetteDataset constructor already has defaults for ImageWoof via args in get_dataloaders,
    # so we mirror that explicitly here to keep labels aligned.
    from data.dataset import IMAGEWOOF_DIR, IMAGEWOOF_URL

    train_dataset_woof = ImageNetteDataset(
        split="train",
        img_size=config["img_size"],
        data_dir=IMAGEWOOF_DIR,
        url=IMAGEWOOF_URL,
        is_training=False,
        class_offset=10,
    )

    train_dataset_combined = torch.utils.data.ConcatDataset(
        [train_dataset_nette, train_dataset_woof]
    )

    train_loader = DataLoader(
        train_dataset_combined,
        batch_size=256,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Combined val set (20 classes)
    val_dataset_combined = torch.utils.data.ConcatDataset(
        [val_loader_nette.dataset, val_loader_woof.dataset]
    )
    val_loader = DataLoader(
        val_dataset_combined,
        batch_size=256,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Extract features
    print("Extracting training features...")
    train_features, train_labels = extract_features(
        model, train_loader, device, mixed_precision=True
    )

    print("Extracting validation features...")
    val_features, val_labels = extract_features(model, val_loader, device, mixed_precision=True)

    print(f"Train features shape: {train_features.shape}")
    print(f"Val features shape: {val_features.shape}")

    # Linear probe evaluation
    print("\nTraining linear probe...")
    linear_acc = train_linear_probe(
        train_features,
        train_labels,
        val_features,
        val_labels,
        num_classes=20,
        epochs=100,
        device=device,
    )
    print(f"Linear probe accuracy: {linear_acc:.2f}%")

    # k-NN evaluation
    print("\nComputing k-NN accuracy...")
    knn_accs = {}
    for k in [1, 5, 10, 20]:
        acc = knn_accuracy(train_features, train_labels, val_features, val_labels, k=k)
        knn_accs[k] = acc
        print(f"  k={k}: {acc:.2f}%")

    return {
        "encoder": config["encoder"],
        "linear_probe_acc": linear_acc,
        "knn_accs": knn_accs,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate LeJEPA models")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument(
        "--checkpoint_jit", type=str, help="Path to JiT model checkpoint"
    )
    parser.add_argument(
        "--checkpoint_vit", type=str, help="Path to ViT model checkpoint"
    )
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    results = []

    if args.checkpoint:
        result = evaluate_model(args.checkpoint, device, args.num_workers)
        results.append(result)

    if args.checkpoint_jit:
        result = evaluate_model(args.checkpoint_jit, device, args.num_workers)
        results.append(result)

    if args.checkpoint_vit:
        result = evaluate_model(args.checkpoint_vit, device, args.num_workers)
        results.append(result)

    # Print comparison
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)
        print(f"{'Encoder':<10} {'Linear Probe':<15} {'k-NN (k=20)':<15}")
        print("-" * 40)
        for r in results:
            print(
                f"{r['encoder'].upper():<10} {r['linear_probe_acc']:<15.2f} {r['knn_accs'][20]:<15.2f}"
            )


if __name__ == "__main__":
    main()
