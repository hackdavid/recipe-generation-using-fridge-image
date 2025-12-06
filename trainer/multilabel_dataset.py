"""
Multi-Label Dataset Loader for HuggingFace Datasets

Loads multi-label classification datasets from HuggingFace Hub
with proper data augmentation and preprocessing.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class MultiLabelDataset(Dataset):
    """
    Multi-label classification dataset wrapper for HuggingFace datasets
    """
    
    def __init__(
        self,
        hf_dataset: HFDataset,
        transform: Optional[transforms.Compose] = None,
        num_classes: Optional[int] = None
    ):
        """
        Initialize multi-label dataset
        
        Args:
            hf_dataset: HuggingFace dataset instance
            transform: Image transformations
            num_classes: Number of classes (inferred if None)
        """
        self.dataset = hf_dataset
        self.transform = transform
        
        # Infer num_classes from data if not provided
        if num_classes is None:
            all_labels = set()
            for item in self.dataset:
                all_labels.update(item['labels'])
            self.num_classes = max(all_labels) + 1 if all_labels else 0
        else:
            self.num_classes = num_classes
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item from dataset
        
        Returns:
            Dictionary with 'image', 'labels', 'label_names'
        """
        item = self.dataset[idx]
        
        # Load image
        image = item['image']
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Convert labels to multi-hot encoding
        labels = item['labels']
        multi_hot = torch.zeros(self.num_classes, dtype=torch.float32)
        if isinstance(labels, list):
            multi_hot[labels] = 1.0
        elif isinstance(labels, torch.Tensor):
            multi_hot[labels.long()] = 1.0
        
        return {
            'image': image,
            'labels': multi_hot,
            'label_names': item.get('label_names', []),
            'num_labels': item.get('num_labels', 0)
        }


def get_multilabel_transforms(
    is_training: bool = True,
    image_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> transforms.Compose:
    """
    Get data augmentation transforms for multi-label classification
    
    Args:
        is_training: Whether this is training data
        image_size: Target image size
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        Compose transform
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


def custom_collate_fn(batch):
    """
    Custom collate function for multi-label dataset
    Handles variable-length label_names lists
    """
    images = torch.stack([item['image'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    # Handle variable-length label_names (keep as list)
    label_names = [item['label_names'] for item in batch]
    num_labels = torch.tensor([item['num_labels'] for item in batch], dtype=torch.long)
    
    return {
        'image': images,
        'labels': labels,
        'label_names': label_names,  # Keep as list (not tensor)
        'num_labels': num_labels
    }


def get_multilabel_data_loaders(
    dataset_name: str,
    split_ratio: float = 0.8,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    seed: int = 42,
    debug_mode: bool = False,
    debug_max_samples: int = 10
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Get data loaders for multi-label classification
    
    Args:
        dataset_name: HuggingFace dataset name or path
        split_ratio: Train/val split ratio
        batch_size: Batch size
        num_workers: Number of data loader workers
        image_size: Image size
        seed: Random seed
        debug_mode: Enable debug mode (small dataset)
        debug_max_samples: Max samples in debug mode
    
    Returns:
        Tuple of (train_loader, val_loader, num_classes)
    """
    print(f"Loading dataset: {dataset_name}")
    
    # Load dataset from HuggingFace
    try:
        dataset = load_dataset(dataset_name, streaming=False)
    except Exception as e:
        raise ValueError(f"Failed to load dataset {dataset_name}: {e}")
    
    # Get splits
    if 'train' in dataset and 'validation' in dataset:
        train_dataset = dataset['train']
        val_dataset = dataset['validation']
        print(f"✓ Using predefined splits")
    else:
        # Manual split if needed
        full_dataset = dataset['train'] if 'train' in dataset else list(dataset.values())[0]
        full_dataset = full_dataset.shuffle(seed=seed)
        split_idx = int(len(full_dataset) * split_ratio)
        train_dataset = full_dataset.select(range(split_idx))
        val_dataset = full_dataset.select(range(split_idx, len(full_dataset)))
        print(f"✓ Created manual split ({split_ratio:.0%}/{1-split_ratio:.0%})")
    
    # Debug mode: use small subset
    if debug_mode:
        print(f"⚠ DEBUG MODE: Using only {debug_max_samples} samples")
        train_dataset = train_dataset.select(range(min(debug_max_samples, len(train_dataset))))
        val_dataset = val_dataset.select(range(min(debug_max_samples, len(val_dataset))))
    
    # Infer num_classes
    all_labels = set()
    for item in train_dataset:
        all_labels.update(item['labels'])
    num_classes = max(all_labels) + 1 if all_labels else 0
    print(f"✓ Found {num_classes} classes")
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    
    # Create datasets
    train_transform = get_multilabel_transforms(is_training=True, image_size=image_size)
    val_transform = get_multilabel_transforms(is_training=False, image_size=image_size)
    
    train_ds = MultiLabelDataset(train_dataset, transform=train_transform, num_classes=num_classes)
    val_ds = MultiLabelDataset(val_dataset, transform=val_transform, num_classes=num_classes)
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=custom_collate_fn
    )
    
    return train_loader, val_loader, num_classes

