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
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

# Dataset URL and paths
# Dataset URL and paths
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
IMAGENETTE_DIR = Path("./data/imagenette2-160")

IMAGEWOOF_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-160.tgz"
IMAGEWOOF_DIR = Path("./data/imagewoof2-160")


def download_dataset(url: str, data_dir: Path) -> Path:
    """Download and extract dataset if not present."""
    if data_dir.exists():
        return data_dir

    data_dir.parent.mkdir(parents=True, exist_ok=True)
    tgz_path = data_dir.parent / Path(url).name

    print(f"Downloading dataset from {url}...")
    urllib.request.urlretrieve(url, tgz_path)
    print("Download complete. Extracting...")

    with tarfile.open(tgz_path, "r:gz") as tar:
        # Python 3.12+ supports `filter=` for safer extraction; older versions don't.
        try:
            tar.extractall(data_dir.parent, filter="data")
        except TypeError:
            tar.extractall(data_dir.parent)

    tgz_path.unlink()  # Remove archive
    print(f"Extracted to {data_dir}")
    return data_dir


class DataAugmentationLEJEPA:
    """
    Multi-crop augmentation for LeJEPA.

    Generates:
    - 2 Global views (224x224)
    - 6 Local views (96x96)
    """

    def __init__(
        self,
        global_crops_scale=(0.3, 1.0),
        local_crops_scale=(0.05, 0.3),
        local_crops_number=6,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.local_crops_number = local_crops_number

        # Base augmentations (same for all views)
        self.color_jitter = v2.Compose(
            [
                v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                v2.RandomGrayscale(p=0.2),
            ]
        )

        self.gaussian_blur = v2.RandomApply(
            [v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=0.5
        )
        self.solarize = v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2)
        self.normalize = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.flip = v2.RandomHorizontalFlip(p=0.5)

        # Global crop transformation
        self.global_transfo = v2.Compose(
            [
                v2.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, antialias=True
                ),
                self.flip,
                self.color_jitter,
                self.gaussian_blur,
                self.solarize,
                self.normalize,
            ]
        )

        # Local crop transformation
        self.local_transfo = v2.Compose(
            [
                v2.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, antialias=True
                ),
                self.flip,
                self.color_jitter,
                self.gaussian_blur,
                self.solarize,
                self.normalize,
            ]
        )

    def __call__(self, image):
        crops = []
        # 2 Global Views
        crops.append(self.global_transfo(image))
        crops.append(self.global_transfo(image))

        # Local Views
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))

        return crops


def get_val_transform(img_size: int = 224) -> v2.Compose:
    """Standard validation transform (resize + center crop)."""
    return v2.Compose(
        [
            v2.Resize(
                img_size, antialias=True
            ),  # Standard typically resizes to slightly larger then crops
            v2.CenterCrop(img_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class ImageNetteDataset(Dataset):
    """
    ImageNette dataset wrapper for LeJEPA training.

    Supports Multi-Crop (list of tensors) for training.
    """

    def __init__(
        self,
        split: str = "train",
        img_size: int = 224,
        data_dir: Path = IMAGENETTE_DIR,
        url: str = IMAGENETTE_URL,
        is_training: bool = True,
        class_offset: int = 0,
        # Multi-crop params
        local_crops_number: int = 6,
        local_crops_size: int = 96,
        local_crops_scale: tuple = (0.05, 0.3),
        global_crops_scale: tuple = (0.3, 1.0),
    ):
        """
        Args:
            split: "train" or "val"
            img_size: Target global image size
            data_dir: Path to dataset directory
            url: Download URL if not present
            is_training: Whether to use Multi-Crop augmentation
            class_offset: Integer to add to all labels (for merging datasets)
        """
        # Ensure dataset is downloaded
        download_dataset(url, data_dir)

        self.split = split
        self.img_size = img_size
        self.is_training = is_training
        self.class_offset = class_offset

        # Map split names
        folder_split = "val" if split == "validation" else split
        split_dir = data_dir / folder_split

        # Load with ImageFolder (no transform - we apply our own)
        self.dataset = ImageFolder(split_dir)
        self.num_classes = 10

        # Set up transforms
        if is_training:
            self.transform = DataAugmentationLEJEPA(
                global_crops_scale=global_crops_scale,
                local_crops_scale=local_crops_scale,
                local_crops_number=local_crops_number,
                global_crops_size=img_size,
                local_crops_size=local_crops_size,
            )
        else:
            self.transform = get_val_transform(img_size)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        """
        Get item from dataset.

        Returns:
            - Training: (crops_list, label)
              where crops_list is [global_1, global_2, local_1, ..., local_6]
            - Validation: (image, label) where image is (1, C, H, W)
        """
        img, label = self.dataset[idx]

        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Apply transforms
        output = self.transform(img)

        # For validation, ensure standard format
        if not self.is_training:
            output = output.unsqueeze(0)  # (1, C, H, W)

        # Apply class offset
        label = label + self.class_offset

        return output, label


def get_dataloaders(
    batch_size: int = 256,
    img_size: int = 224,
    num_workers: int = 8,
    pin_memory: bool = True,
    data_dir: Path = IMAGENETTE_DIR,
    # Multi-crop params
    local_crops_number: int = 6,
    local_crops_size: int = 96,
    local_crops_scale: tuple = (0.05, 0.3),
    global_crops_scale: tuple = (0.3, 1.0),
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create training and validation dataloaders for combined ImageNette + ImageWoof.

    Args:
        batch_size: Batch size
        img_size: Image size (global crop size)
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        data_dir: Ignored (uses IMAGENETTE_DIR and IMAGEWOOF_DIR)
        ... : Multi-crop parameters

    Returns:
        (train_loader, val_loader_imagenette, val_loader_imagewoof)
    """
    # 1. ImageNette Training
    train_ds_nette = ImageNetteDataset(
        split="train",
        img_size=img_size,
        data_dir=IMAGENETTE_DIR,
        url=IMAGENETTE_URL,
        is_training=True,
        class_offset=0,
        local_crops_number=local_crops_number,
        local_crops_size=local_crops_size,
        local_crops_scale=local_crops_scale,
        global_crops_scale=global_crops_scale,
    )

    # 2. ImageWoof Training (Offset labels by 10)
    train_ds_woof = ImageNetteDataset(
        split="train",
        img_size=img_size,
        data_dir=IMAGEWOOF_DIR,
        url=IMAGEWOOF_URL,
        is_training=True,
        class_offset=10,
        local_crops_number=local_crops_number,
        local_crops_size=local_crops_size,
        local_crops_scale=local_crops_scale,
        global_crops_scale=global_crops_scale,
    )

    # Combined Training Dataset
    train_dataset = torch.utils.data.ConcatDataset([train_ds_nette, train_ds_woof])

    # 3. ImageNette Validation
    val_ds_nette = ImageNetteDataset(
        split="val",
        img_size=img_size,
        data_dir=IMAGENETTE_DIR,
        url=IMAGENETTE_URL,
        is_training=False,
        class_offset=0,
    )

    # 4. ImageWoof Validation
    val_ds_woof = ImageNetteDataset(
        split="val",
        img_size=img_size,
        data_dir=IMAGEWOOF_DIR,
        url=IMAGEWOOF_URL,
        is_training=False,
        class_offset=10,  # Maintain offset for consistent evaluation with probe
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader_nette = DataLoader(
        val_ds_nette,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader_woof = DataLoader(
        val_ds_woof,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader_nette, val_loader_woof


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


# ImageNet class ID to name mapping for ImageWoof
CLASS_ID_TO_NAME_WOOF = {
    "n02086240": "Shih-Tzu",
    "n02087394": "Rhodesian ridgeback",
    "n02088364": "Beagle",
    "n02089973": "English foxhound",
    "n02093754": "Australian terrier",
    "n02096294": "Border terrier",
    "n02099601": "Golden retriever",
    "n02105641": "Old English sheepdog",
    "n02111889": "Samoyed",
    "n02115641": "Dingo",
}


def get_class_names() -> list[str]:
    """Return ImageNette + ImageWoof class names in folder order."""
    return list(CLASS_ID_TO_NAME.values()) + list(CLASS_ID_TO_NAME_WOOF.values())
