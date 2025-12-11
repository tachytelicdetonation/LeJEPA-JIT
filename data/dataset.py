"""
ImageNette dataset loading with augmentations for LeJEPA training.

ImageNette is a subset of ImageNet with 10 easy-to-classify classes:
- tench, English springer, cassette player, chain saw, church,
- French horn, garbage truck, gas pump, golf ball, parachute

Downloads from fastai's S3 bucket and uses torchvision ImageFolder.
"""

import tarfile
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2


# Dataset URL and paths
DATA_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
DATA_DIR = Path("./data/imagenette2-160")


def download_imagenette(data_dir: Path = DATA_DIR) -> Path:
    """Download and extract ImageNette dataset if not present."""
    if data_dir.exists():
        return data_dir

    data_dir.parent.mkdir(parents=True, exist_ok=True)
    tgz_path = data_dir.parent / "imagenette2-160.tgz"

    print(f"Downloading ImageNette from {DATA_URL}...")
    urllib.request.urlretrieve(DATA_URL, tgz_path)
    print("Download complete. Extracting...")

    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(data_dir.parent, filter="data")

    tgz_path.unlink()  # Remove archive
    print(f"Extracted to {data_dir}")
    return data_dir


def get_train_transform(img_size: int = 128) -> v2.Compose:
    """
    Strong augmentations for training (matching LeJEPA MINIMAL.md).

    Includes:
    - Random resized crop (scale 0.08-1.0)
    - Color jitter
    - Grayscale conversion
    - Gaussian blur
    - Solarization
    - Horizontal flip
    """
    return v2.Compose(
        [
            v2.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
            v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))]),
            v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_val_transform(img_size: int = 128) -> v2.Compose:
    """Standard validation transform (resize + center crop)."""
    return v2.Compose(
        [
            v2.Resize(img_size),
            v2.CenterCrop(img_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class ImageNetteDataset(Dataset):
    """
    ImageNette dataset wrapper for LeJEPA training.

    Supports multi-view transforms for self-supervised learning.
    """

    def __init__(
        self,
        split: str = "train",
        img_size: int = 128,
        num_views: int = 2,
        data_dir: Path = DATA_DIR,
    ):
        """
        Args:
            split: "train" or "val"
            img_size: Target image size
            num_views: Number of augmented views per sample
            data_dir: Path to imagenette2-160 directory
        """
        # Ensure dataset is downloaded
        download_imagenette(data_dir)

        self.split = split
        self.img_size = img_size
        self.num_views = num_views

        # Map split names
        folder_split = "val" if split == "validation" else split
        split_dir = data_dir / folder_split

        # Load with ImageFolder (no transform - we apply our own)
        self.dataset = ImageFolder(split_dir)
        self.num_classes = 10

        # Set up transforms
        if split == "train":
            self.transform = get_train_transform(img_size)
        else:
            self.transform = get_val_transform(img_size)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Get item from dataset.

        Returns:
            (images, label) where images is (V, C, H, W) tensor
        """
        img, label = self.dataset[idx]

        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Generate multiple views
        views = torch.stack([self.transform(img) for _ in range(self.num_views)])

        return views, label


def get_dataloaders(
    batch_size: int = 256,
    img_size: int = 128,
    num_views: int = 4,
    num_workers: int = 8,
    pin_memory: bool = True,
    data_dir: Path = DATA_DIR,
) -> tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.

    Args:
        batch_size: Batch size
        img_size: Image size
        num_views: Number of augmented views for training
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        data_dir: Path to dataset

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = ImageNetteDataset(
        split="train",
        img_size=img_size,
        num_views=num_views,
        data_dir=data_dir,
    )

    val_dataset = ImageNetteDataset(
        split="val",
        img_size=img_size,
        num_views=1,  # Single view for validation
        data_dir=data_dir,
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


# ImageNet class ID to name mapping for ImageNette
CLASS_ID_TO_NAME = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}


def get_class_names() -> list[str]:
    """Return ImageNette class names in folder order."""
    return list(CLASS_ID_TO_NAME.values())
