"""
Custom PyTorch Dataset Loader for Multiple Data Sources

This module provides unified dataset loading that supports:
1. Local folder (ImageFolder) - with automatic 80/20 split if no train/val folders exist
2. HuggingFace datasets - with automatic 80/20 split if no predefined splits exist
3. HuggingFace datasets - with standard predefined splits

Usage:
    from trainer.hf_dataset import get_data_loaders
    
    train_loader, val_loader, num_classes, class_names = get_data_loaders(
        data_source='folder',  # or 'huggingface'
        data_dir='./data',  # Local path (for folder) or HuggingFace dataset name
        num_classes=100,  # Required: from config
        batch_size=32
    )
"""

import os
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import ImageFolder
from datasets import load_dataset
import torchvision.transforms as transforms
from PIL import Image
from typing import Optional, Callable, Tuple, List, Dict
import hashlib
import json


class HuggingFaceStreamDataset(torch.utils.data.IterableDataset):
    """
    Custom PyTorch Dataset for loading HuggingFace datasets in streaming mode.
    
    This class implements a Dataset that streams data from HuggingFace
    datasets without loading the entire dataset into memory. It supports:
    - Streaming mode for memory efficiency
    - Custom transforms
    - Shuffling with buffer-based approach
    - Custom 80/20 train/val split in streaming mode
    
    Note: Labels should already be integers in the HuggingFace dataset.
    
    Args:
        dataset_name: HuggingFace dataset identifier (e.g., 'SunnyAgarwal4274/Food_Ingredients')
        num_classes: Number of classes (required, should be provided from config)
        split: Dataset split ('train', 'validation', 'test', etc.) OR None for custom split
        transform: Optional torchvision transforms to apply
        label_key: Key name for labels in the dataset (default: 'label')
        image_key: Key name for images in the dataset (default: 'image')
        shuffle: Whether to shuffle the dataset (uses HF shuffle with buffer_size)
        shuffle_buffer_size: Buffer size for shuffling (default: 10000)
        seed: Random seed for shuffling
        use_custom_split: If True, ignore predefined splits and create 80/20 train/val split
        train_ratio: Ratio for training split when use_custom_split=True (default: 0.8)
        split_type: 'train' or 'val' when use_custom_split=True
    """
    
    def __init__(
        self,
        dataset_name: str,
        num_classes: int,
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        label_key: str = 'label',
        image_key: str = 'image',
        shuffle: bool = True,
        shuffle_buffer_size: int = 10000,
        seed: int = 42,
        use_custom_split: bool = False,
        train_ratio: float = 0.8,
        split_type: str = 'train',
        label_to_id: Optional[Dict[str, int]] = None
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
        self.use_custom_split = use_custom_split
        self.train_ratio = train_ratio
        self.split_type = split_type  # 'train' or 'val' when use_custom_split=True
        self.label_to_id = label_to_id  # Mapping from string labels to integer IDs
        
        # Load dataset
        if use_custom_split or split is None:
            # Load entire dataset for custom splitting
            try:
                dataset_dict = load_dataset(dataset_name, streaming=True)
                # If it's a DatasetDict, use first available split
                if hasattr(dataset_dict, 'values'):
                    # Get first available split (usually 'train')
                    first_split = next(iter(dataset_dict.keys()))
                    self.dataset_stream = dataset_dict[first_split]
                else:
                    # Already a single dataset
                    self.dataset_stream = dataset_dict
            except Exception as e:
                # Fallback: try loading 'train' split directly
                try:
                    self.dataset_stream = load_dataset(dataset_name, split='train', streaming=True)
                except Exception as e2:
                    raise ValueError(
                        f"Could not load dataset {dataset_name} for custom split. "
                        f"Tried loading all splits and 'train' split. Error: {e2}"
                    )
        else:
            # Use predefined split
            self.dataset_stream = load_dataset(dataset_name, split=split, streaming=True)
        
        # Apply shuffling if requested (before custom split filtering)
        if shuffle:
            self.dataset_stream = self.dataset_stream.shuffle(
                seed=seed, 
                buffer_size=shuffle_buffer_size
            )
    
    def __iter__(self):
        """Iterate over the dataset (streaming)"""
        for sample in self.dataset_stream:
            # If using custom split, filter samples based on hash
            if self.use_custom_split:
                sample_str = str(sample.get(self.label_key, '')) + str(sample.get(self.image_key, ''))
                sample_hash = int(hashlib.md5(sample_str.encode()).hexdigest(), 16)
                hash_ratio = (sample_hash % 100) / 100.0
                
                if self.split_type == 'train':
                    if hash_ratio >= self.train_ratio:
                        continue  # Skip this sample (belongs to validation)
                elif self.split_type == 'val':
                    if hash_ratio < self.train_ratio:
                        continue  # Skip this sample (belongs to training)
            
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
            # First check if 'label_id' exists (preferred)
            if 'label_id' in sample:
                label = sample['label_id']
            else:
                label = sample[self.label_key]
            
            # Convert label to integer
            if isinstance(label, torch.Tensor):
                label = label.item()
            elif isinstance(label, str):
                # String label - use mapping
                if self.label_to_id:
                    if label in self.label_to_id:
                        label = self.label_to_id[label]
                    else:
                        # Label not in mapping - skip this sample
                        print(f"Warning: Label '{label}' not found in mapping. Skipping sample.")
                        continue
                else:
                    # No mapping available - try to convert directly (will fail if string)
                    try:
                        label = int(label)
                    except (ValueError, TypeError):
                        raise ValueError(
                            f"Label '{label}' is a string but no label mapping is available. "
                            f"Please generate class mapping JSON first using: "
                            f"python trainer/generate_class_mapping.py --dataset_name {self.dataset_name}"
                        )
            elif not isinstance(label, int):
                # Try to convert to int
                try:
                    label = int(label)
                except (ValueError, TypeError):
                    raise ValueError(f"Cannot convert label '{label}' (type: {type(label)}) to integer")
            
            yield image, label
    
    def __len__(self):
        """Note: Streaming datasets don't have a fixed length"""
        # Try to get length from dataset info
        try:
            if self.split:
                dataset_info = load_dataset(self.dataset_name, split=self.split, streaming=False)
                return len(dataset_info)
        except:
            pass
        return None


def load_class_mapping(mapping_path: str) -> Dict:
    """
    Load class mapping from JSON file.
    
    Args:
        mapping_path: Path to the JSON file containing class mapping
        
    Returns:
        Dictionary containing:
            - 'label_to_id': Dict mapping class names to IDs
            - 'id_to_label': Dict mapping IDs to class names
            - 'class_names': List of class names in order
            - 'num_classes': Number of classes
    """
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(
            f"Class mapping file not found: {mapping_path}\n"
            f"Please generate it first using: python trainer/generate_class_mapping.py"
        )
    
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)
    
    print(f"✓ Class mapping loaded from: {mapping_path}")
    print(f"  Number of classes: {mapping_data.get('num_classes', len(mapping_data.get('class_names', [])))}")
    
    return mapping_data


def get_data_loaders(
    data_source: str,
    data_dir: str,
    num_classes: int,
    train_split: Optional[str] = None,
    val_split: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    shuffle_train: bool = True,
    shuffle_buffer_size: int = 10000,
    seed: int = 42,
    train_ratio: float = 0.8,
    class_mapping_path: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, int, Optional[List[str]]]:
    """
    Unified function to create data loaders from different sources.
    
    Supports three scenarios:
    1. Local folder (ImageFolder): If data_dir exists locally
       - If train/val folders exist: use them directly
       - If no train/val folders: create 80/20 split automatically
    2. HuggingFace dataset (no splits): If data_dir doesn't exist locally and no splits
       - Load whole dataset in streaming mode and create 80/20 split
    3. HuggingFace dataset (with splits): If data_dir doesn't exist locally and splits exist
       - Use predefined splits (train_split and val_split)
    
    Args:
        data_source: 'folder' or 'huggingface'
        data_dir: Local directory path (for folder) or HuggingFace dataset name
        num_classes: Number of classes (required, from config)
        train_split: Training split name (for HuggingFace, default: 'train')
        val_split: Validation split name (for HuggingFace, default: 'validation')
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        image_size: Target image size
        shuffle_train: Whether to shuffle training data
        shuffle_buffer_size: Buffer size for shuffling (HuggingFace only)
        seed: Random seed
        train_ratio: Ratio for train split when creating custom splits (default: 0.8)
        
    Returns:
        train_loader, val_loader, num_classes, class_names (or None)
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
    
    # Scenario 1: Local folder (ImageFolder)
    if data_source == 'folder' and os.path.exists(data_dir):
        print(f"Loading dataset from local folder: {data_dir}")
        
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')
        
        # Check if train/val folders exist
        if os.path.exists(train_dir) and os.path.exists(val_dir):
            print("Found train/val folders, using them directly")
            train_dataset = ImageFolder(train_dir, transform=train_transform)
            val_dataset = ImageFolder(val_dir, transform=val_transform)
            class_names = train_dataset.classes
        else:
            print("No train/val folders found, creating 80/20 split from root directory")
            # Load entire dataset and split 80/20
            full_dataset = ImageFolder(data_dir, transform=train_transform)
            class_names = full_dataset.classes
            
            # Calculate split sizes
            total_size = len(full_dataset)
            train_size = int(train_ratio * total_size)
            val_size = total_size - train_size
            
            # Split dataset indices
            indices = list(range(total_size))
            import random
            random.seed(seed)
            random.shuffle(indices)
            
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            
            # Create datasets with proper transforms
            train_dataset = Subset(full_dataset, train_indices)
            
            # Create validation dataset with val transform
            val_full_dataset = ImageFolder(data_dir, transform=val_transform)
            val_dataset = Subset(val_full_dataset, val_indices)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
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
        
        return train_loader, val_loader, num_classes, class_names
    
    # Scenario 2 & 3: HuggingFace dataset
    elif data_source == 'huggingface' or not os.path.exists(data_dir):
        dataset_name = data_dir  # data_dir is the HuggingFace dataset name
        
        print(f"Loading HuggingFace dataset: {dataset_name}")
        
        # Load class mapping if provided
        label_to_id = None
        class_names = None
        if class_mapping_path:
            mapping_data = load_class_mapping(class_mapping_path)
            label_to_id = mapping_data.get('label_to_id', {})
            class_names = mapping_data.get('class_names', [])
            print(f"Using class mapping with {len(class_names)} classes")
        
        # Check if dataset has predefined splits
        try:
            dataset_dict = load_dataset(dataset_name, streaming=False)
            has_splits = hasattr(dataset_dict, 'keys') and len(dataset_dict.keys()) > 0
            
            if has_splits:
                available_splits = list(dataset_dict.keys())
                print(f"Found predefined splits: {available_splits}")
                
                # Use standard splits
                train_split = train_split or 'train'
                val_split = val_split or 'validation'
                
                if train_split not in available_splits:
                    raise ValueError(f"Train split '{train_split}' not found. Available splits: {available_splits}")
                if val_split not in available_splits:
                    print(f"Warning: Val split '{val_split}' not found. Available splits: {available_splits}")
                    # Use first available split as val if specified one doesn't exist
                    val_split = available_splits[1] if len(available_splits) > 1 else available_splits[0]
                
                print(f"Using splits: train='{train_split}', val='{val_split}'")
                
                train_dataset = HuggingFaceStreamDataset(
                    dataset_name=dataset_name,
                    num_classes=num_classes,
                    split=train_split,
                    transform=train_transform,
                    shuffle=shuffle_train,
                    shuffle_buffer_size=shuffle_buffer_size,
                    seed=seed,
                    use_custom_split=False,
                    label_to_id=label_to_id
                )
                
                val_dataset = HuggingFaceStreamDataset(
                    dataset_name=dataset_name,
                    num_classes=num_classes,
                    split=val_split,
                    transform=val_transform,
                    shuffle=False,
                    seed=seed,
                    use_custom_split=False,
                    label_to_id=label_to_id
                )
            else:
                # No predefined splits, create 80/20 split
                print("No predefined splits found, creating 80/20 split in streaming mode")
                
                train_dataset = HuggingFaceStreamDataset(
                    dataset_name=dataset_name,
                    num_classes=num_classes,
                    split=None,
                    transform=train_transform,
                    shuffle=shuffle_train,
                    shuffle_buffer_size=shuffle_buffer_size,
                    seed=seed,
                    use_custom_split=True,
                    train_ratio=train_ratio,
                    split_type='train',
                    label_to_id=label_to_id
                )
                
                val_dataset = HuggingFaceStreamDataset(
                    dataset_name=dataset_name,
                    num_classes=num_classes,
                    split=None,
                    transform=val_transform,
                    shuffle=False,
                    seed=seed,
                    use_custom_split=True,
                    train_ratio=train_ratio,
                    split_type='val',
                    label_to_id=label_to_id
                )
        except Exception as e:
            # Fallback: assume no splits and create custom split
            print(f"Could not detect splits, creating 80/20 split: {e}")
            
            train_dataset = HuggingFaceStreamDataset(
                dataset_name=dataset_name,
                num_classes=num_classes,
                split=None,
                transform=train_transform,
                shuffle=shuffle_train,
                shuffle_buffer_size=shuffle_buffer_size,
                seed=seed,
                use_custom_split=True,
                train_ratio=train_ratio,
                split_type='train',
                label_to_id=label_to_id
            )
            
            val_dataset = HuggingFaceStreamDataset(
                dataset_name=dataset_name,
                num_classes=num_classes,
                split=None,
                transform=val_transform,
                shuffle=False,
                seed=seed,
                use_custom_split=True,
                train_ratio=train_ratio,
                split_type='val',
                label_to_id=label_to_id
            )
        
        # Create data loaders
        # Note: For IterableDataset, we can't use shuffle=True in DataLoader
        # For streaming datasets, limit num_workers to avoid issues with dataset shards
        # On Windows or with streaming datasets, num_workers > 0 can cause issues
        effective_num_workers = num_workers
        
        # Try to detect dataset shards and limit workers accordingly
        try:
            if hasattr(train_dataset, 'dataset_stream'):
                # Check if dataset has num_shards attribute
                if hasattr(train_dataset.dataset_stream, 'info') and hasattr(train_dataset.dataset_stream.info, 'num_shards'):
                    max_shards = train_dataset.dataset_stream.info.num_shards
                    if max_shards and num_workers > max_shards:
                        print(f"Warning: Limiting num_workers from {num_workers} to {max_shards} (dataset has {max_shards} shards)")
                        effective_num_workers = min(num_workers, max_shards)
        except:
            pass
        
        # On Windows, multiprocessing can be problematic - use 0 workers if issues occur
        import sys
        if sys.platform == 'win32' and num_workers > 0:
            # For Windows, prefer 0 workers for streaming datasets to avoid multiprocessing issues
            if effective_num_workers > 0:
                print(f"Note: Using num_workers={effective_num_workers} on Windows. Set to 0 if you encounter worker errors.")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,  # Shuffling handled by dataset
            num_workers=effective_num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=effective_num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        return train_loader, val_loader, num_classes, class_names
    
    else:
        raise ValueError(f"Invalid data_source: {data_source}. Must be 'folder' or 'huggingface'")


# Alias for backward compatibility
get_hf_data_loaders = get_data_loaders
