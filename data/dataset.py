"""
ImageNette dataset loading with augmentations for LeJEPA training.

ImageNette is a subset of ImageNet with 10 easy-to-classify classes:
- tench, English springer, cassette player, chain saw, church,
- French horn, garbage truck, gas pump, golf ball, parachute

Uses HuggingFace datasets for easy downloading and caching.
"""

from typing import Callable, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

try:
    from datasets import load_dataset

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class MultiViewTransform:
    """
    Generates multiple augmented views of the same image.

    Used for self-supervised learning where we want different
    augmentations of the same image to have similar embeddings.
    """

    def __init__(self, base_transform: Callable, num_views: int = 2):
        self.base_transform = base_transform
        self.num_views = num_views

    def __call__(self, img: Image.Image) -> torch.Tensor:
        """
        Apply transform multiple times to generate views.

        Returns:
            (num_views, C, H, W) tensor of augmented views
        """
        views = [self.base_transform(img) for _ in range(self.num_views)]
        return torch.stack(views)


def get_train_transform(img_size: int = 128) -> transforms.Compose:
    """
    Strong augmentations for training (from LeJEPA).

    Includes:
    - Random resized crop
    - Color jitter
    - Grayscale conversion
    - Gaussian blur
    - Solarization
    - Horizontal flip
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                img_size,
                scale=(0.2, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.2,
                        hue=0.1,
                    )
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5
            ),
            transforms.RandomSolarize(threshold=128, p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_val_transform(img_size: int = 128) -> transforms.Compose:
    """Standard validation transform (resize + center crop)."""
    return transforms.Compose(
        [
            transforms.Resize(
                int(img_size * 1.14),  # Resize to slightly larger
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class ImageNetteDataset(Dataset):
    """
    ImageNette dataset wrapper for LeJEPA training.

    Supports both single-view (for validation) and multi-view (for training)
    transforms.
    """

    def __init__(
        self,
        split: str = "train",
        img_size: int = 128,
        num_views: int = 2,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            split: "train" or "validation"
            img_size: Target image size
            num_views: Number of augmented views (only for training)
            transform: Custom transform (overrides default)
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "HuggingFace datasets required. Install with: pip install datasets"
            )

        self.split = split
        self.img_size = img_size
        self.num_views = num_views

        # Load dataset from HuggingFace (use "160px" config like LeJEPA)
        self.dataset = load_dataset(
            "frgfm/imagenette", "160px", split=split, trust_remote_code=True
        )

        # Set up transforms
        if transform is not None:
            self.transform = transform
        elif split == "train":
            base_transform = get_train_transform(img_size)
            self.transform = MultiViewTransform(base_transform, num_views)
        else:
            self.transform = get_val_transform(img_size)

        # Get number of classes
        self.num_classes = 10  # ImageNette has 10 classes

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        """
        Get item from dataset.

        Returns:
            Dictionary with:
                - image: (V, C, H, W) for training or (C, H, W) for validation
                - label: Integer class label
        """
        item = self.dataset[idx]
        image = item["image"]
        label = item["label"]

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Apply transform
        image = self.transform(image)

        return {
            "image": image,
            "label": label,
        }


def get_dataloaders(
    batch_size: int = 256,
    img_size: int = 128,
    num_views: int = 2,
    num_workers: int = 8,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.

    Args:
        batch_size: Batch size
        img_size: Image size
        num_views: Number of augmented views for training
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = ImageNetteDataset(
        split="train",
        img_size=img_size,
        num_views=num_views,
    )

    val_dataset = ImageNetteDataset(
        split="validation",
        img_size=img_size,
        num_views=1,  # Single view for validation
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


def get_class_names() -> list[str]:
    """Return ImageNette class names."""
    return [
        "tench",
        "English springer",
        "cassette player",
        "chain saw",
        "church",
        "French horn",
        "garbage truck",
        "gas pump",
        "golf ball",
        "parachute",
    ]
