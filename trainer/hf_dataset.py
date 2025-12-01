"""
Custom PyTorch Dataset Loader for HuggingFace Datasets in Streaming Mode

This module provides a custom IterableDataset class that loads HuggingFace datasets
in streaming mode, making it memory-efficient for large datasets.

Usage:
    from hf_dataset import get_hf_data_loaders
    
    train_loader, val_loader, num_classes, class_names = get_hf_data_loaders(
        dataset_name='SunnyAgarwal4274/Food_Ingredients',
        train_split='train',
        val_split='validation',
        batch_size=32
    )
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
import torchvision.transforms as transforms
from PIL import Image
from typing import Optional, Callable


class HuggingFaceStreamDataset(IterableDataset):
    """
    Custom PyTorch Dataset for loading HuggingFace datasets in streaming mode.
    
    This class implements an IterableDataset that streams data from HuggingFace
    datasets without loading the entire dataset into memory. It supports:
    - Streaming mode for memory efficiency
    - Custom transforms
    - Shuffling with buffer-based approach
    
    Note: Labels should already be integers in the HuggingFace dataset.
    
    Args:
        dataset_name: HuggingFace dataset identifier (e.g., 'SunnyAgarwal4274/Food_Ingredients')
        split: Dataset split ('train', 'validation', 'test', etc.)
        transform: Optional torchvision transforms to apply
        label_key: Key name for labels in the dataset (default: 'label')
        image_key: Key name for images in the dataset (default: 'image')
        num_classes: Number of classes (required, should be provided from config)
        shuffle: Whether to shuffle the dataset (uses HF shuffle with buffer_size)
        shuffle_buffer_size: Buffer size for shuffling (default: 10000)
        seed: Random seed for shuffling
    """
    
    def __init__(
        self,
        dataset_name: str,
        num_classes: int,
        split: str = 'train',
        transform: Optional[Callable] = None,
        label_key: str = 'label',
        image_key: str = 'image',
        shuffle: bool = True,
        shuffle_buffer_size: int = 10000,
        seed: int = 42
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform
        self.label_key = label_key
        self.image_key = image_key
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        
        # Load dataset in streaming mode
        self.dataset_stream = load_dataset(dataset_name, split=split, streaming=True)
        
        # Apply shuffling if requested
        if shuffle:
            self.dataset_stream = self.dataset_stream.shuffle(
                seed=seed, 
                buffer_size=shuffle_buffer_size
            )
    
    def __iter__(self):
        """Iterate over the dataset"""
        for sample in self.dataset_stream:
            # Extract image
            image = sample[self.image_key]
            
            # Convert PIL Image if needed
            if not isinstance(image, Image.Image):
                if hasattr(image, 'convert'):
                    image = image.convert('RGB')
                else:
                    raise ValueError(f"Unexpected image type: {type(image)}")
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            # Extract and process label
            label = sample[self.label_key]
            
            # Ensure label is integer (labels should already be integers from HF dataset)
            if not isinstance(label, (int, torch.Tensor)):
                label = int(label)
            
            yield image, label
    
    def __len__(self):
        """Note: Streaming datasets don't have a fixed length"""
        # Try to get length from dataset info
        try:
            dataset_info = load_dataset(self.dataset_name, split=self.split, streaming=False)
            return len(dataset_info)
        except:
            return None


def get_hf_data_loaders(
    dataset_name: str,
    num_classes: int,
    train_split: str = 'train',
    val_split: str = 'validation',
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    shuffle_train: bool = True,
    shuffle_buffer_size: int = 10000,
    seed: int = 42
):
    """
    Create data loaders for HuggingFace dataset in streaming mode.
    
    This function creates train and validation data loaders from a HuggingFace
    dataset using streaming mode for memory efficiency.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        num_classes: Number of classes (required, should be provided from config)
        train_split: Training split name (default: 'train')
        val_split: Validation split name (default: 'validation')
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        image_size: Target image size
        shuffle_train: Whether to shuffle training data
        shuffle_buffer_size: Buffer size for shuffling
        seed: Random seed
        
    Returns:
        train_loader, val_loader, num_classes, None (class_names not available, can be added later for inference)
        
    Example:
        >>> train_loader, val_loader, num_classes, class_names = get_hf_data_loaders(
        ...     dataset_name='SunnyAgarwal4274/Food_Ingredients',
        ...     num_classes=100,  # Required: from config
        ...     batch_size=32
        ... )
    """
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = HuggingFaceStreamDataset(
        dataset_name=dataset_name,
        num_classes=num_classes,
        split=train_split,
        transform=train_transform,
        shuffle=shuffle_train,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed
    )
    
    val_dataset = HuggingFaceStreamDataset(
        dataset_name=dataset_name,
        num_classes=num_classes,
        split=val_split,
        transform=val_transform,
        shuffle=False,  # No shuffling for validation
        seed=seed
    )
    
    # Get number of classes
    num_classes = train_dataset.num_classes
    
    # Create data loaders
    # Note: For IterableDataset, we can't use shuffle=True in DataLoader
    # Shuffling is handled by the dataset itself
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Shuffling handled by dataset
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, num_classes, None

