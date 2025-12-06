"""
Upload multi-label dataset to HuggingFace Hub with Parquet format and train/val splits.

This script:
1. Loads the generated multi-label dataset
2. Splits into train/validation (80/20)
3. Converts to Parquet format
4. Uploads to HuggingFace Hub
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value, Sequence, Image as ImageFeature
from PIL import Image
import numpy as np
from huggingface_hub import login, HfApi
from tqdm import tqdm


def load_annotations(annotations_path: Path) -> Dict:
    """Load annotations from JSON file"""
    print(f"Loading annotations from {annotations_path}...")
    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    print(f"✓ Loaded {len(annotations)} annotations")
    return annotations


def load_image(image_path: Path) -> Image.Image:
    """Load a single image"""
    try:
        img = Image.open(image_path).convert('RGB')
        return img
    except Exception as e:
        raise ValueError(f"Failed to load {image_path}: {e}")


def create_dataset_split(
    annotations: Dict,
    image_dir: Path,
    split_ratio: float = 0.8,
    seed: int = 42
) -> tuple:
    """
    Split dataset into train/validation sets (memory-efficient)
    
    Args:
        annotations: Dictionary of annotations
        image_dir: Directory containing images
        split_ratio: Ratio for training set (default: 0.8 for 80/20 split)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_keys, val_keys) - lists of image filenames
    """
    # Shuffle keys for random split
    keys = list(annotations.keys())
    np.random.seed(seed)
    np.random.shuffle(keys)
    
    # Split keys
    split_idx = int(len(keys) * split_ratio)
    train_keys = keys[:split_idx]
    val_keys = keys[split_idx:]
    
    print(f"\nSplitting dataset:")
    print(f"  Train: {len(train_keys)} images ({len(train_keys)/len(keys)*100:.1f}%)")
    print(f"  Validation: {len(val_keys)} images ({len(val_keys)/len(keys)*100:.1f}%)")
    
    return train_keys, val_keys


def create_data_generator(keys: List[str], annotations: Dict, image_dir: Path):
    """Generator function to create data on-demand (memory-efficient)"""
    for key in keys:
        image_path = image_dir / key
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue
        
        annotation = annotations[key]
        
        # Load image on-demand
        try:
            img = load_image(image_path)
            yield {
                'image': img,
                'labels': annotation['labels'],
                'label_names': annotation['label_names'],
                'num_labels': annotation['num_labels']
            }
        except Exception as e:
            print(f"Warning: Skipping {key}: {e}")
            continue


def create_huggingface_dataset(
    train_keys: List[str],
    val_keys: List[str],
    annotations: Dict,
    image_dir: Path,
    num_classes: int,
    batch_size: int = 1000
) -> DatasetDict:
    """
    Create HuggingFace DatasetDict from data (memory-efficient)
    
    Args:
        train_keys: List of training image filenames
        val_keys: List of validation image filenames
        annotations: Dictionary of annotations
        image_dir: Directory containing images
        num_classes: Number of classes
        batch_size: Batch size for processing (default: 1000)
    
    Returns:
        DatasetDict with train and validation splits
    """
    print("\nCreating HuggingFace datasets...")
    
    # Define features schema - use Image feature type from datasets
    features = Features({
        'image': ImageFeature(),
        'labels': Sequence(Value('int32')),  # Multi-hot encoded labels
        'label_names': Sequence(Value('string')),
        'num_labels': Value('int32')
    })
    
    # Process training data in batches to avoid memory issues
    print("Creating train dataset (processing in batches)...")
    train_batches = []
    for i in tqdm(range(0, len(train_keys), batch_size), desc="Processing train batches"):
        batch_keys = train_keys[i:i+batch_size]
        batch_data = list(create_data_generator(batch_keys, annotations, image_dir))
        if batch_data:
            train_batches.append(batch_data)
    
    # Flatten batches
    train_data = []
    for batch in train_batches:
        train_data.extend(batch)
    
    print(f"  Loaded {len(train_data)} training samples")
    
    # Process validation data in batches
    print("Creating validation dataset (processing in batches)...")
    val_batches = []
    for i in tqdm(range(0, len(val_keys), batch_size), desc="Processing val batches"):
        batch_keys = val_keys[i:i+batch_size]
        batch_data = list(create_data_generator(batch_keys, annotations, image_dir))
        if batch_data:
            val_batches.append(batch_data)
    
    # Flatten batches
    val_data = []
    for batch in val_batches:
        val_data.extend(batch)
    
    print(f"  Loaded {len(val_data)} validation samples")
    
    # Create datasets
    print("Creating train dataset object...")
    train_dataset = Dataset.from_list(train_data, features=features)
    
    print("Creating validation dataset object...")
    val_dataset = Dataset.from_list(val_data, features=features)
    
    # Clear memory
    del train_data, val_data, train_batches, val_batches
    
    # Create dataset dict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })
    
    print(f"✓ Created dataset with {len(train_dataset)} train and {len(val_dataset)} validation samples")
    
    return dataset_dict


def upload_to_huggingface(
    dataset_dict: DatasetDict,
    repo_id: str,
    private: bool = False,
    push_to_hub: bool = True
) -> str:
    """
    Upload dataset to HuggingFace Hub
    
    Args:
        dataset_dict: DatasetDict to upload
        repo_id: HuggingFace repository ID (e.g., 'username/dataset-name')
        private: Whether the dataset should be private
        push_to_hub: Whether to actually push (False for testing)
    
    Returns:
        Repository URL
    """
    print(f"\n{'='*70}")
    print(f"Uploading to HuggingFace Hub")
    print(f"{'='*70}")
    print(f"Repository: {repo_id}")
    print(f"Private: {private}")
    
    if push_to_hub:
        print("\nPushing dataset to hub...")
        dataset_dict.push_to_hub(
            repo_id=repo_id,
            private=private,
            max_shard_size="500MB"  # Split large datasets into shards
        )
        repo_url = f"https://huggingface.co/datasets/{repo_id}"
        print(f"\n✓ Dataset uploaded successfully!")
        print(f"  View at: {repo_url}")
        return repo_url
    else:
        print("\n[DRY RUN] Would upload dataset to hub")
        return f"https://huggingface.co/datasets/{repo_id}"


def create_dataset_card(
    repo_id: str,
    num_classes: int,
    train_size: int,
    val_size: int,
    description: str = None
) -> str:
    """Create dataset card markdown"""
    
    if description is None:
        description = """
This is a multi-label food recognition dataset generated from single-class food images.
Each image contains 2-5 different food items composited together using natural composition methods.
"""
    
    card = f"""---
license: mit
task_categories:
- image-classification
- multi-label-classification
tags:
- food-recognition
- multi-label
- computer-vision
- food-classification
size_categories:
- 10K<n<100K
---

# Multi-Label Food Recognition Dataset

{description}

## Dataset Details

- **Total Images**: {train_size + val_size:,}
- **Training Images**: {train_size:,} (80%)
- **Validation Images**: {val_size:,} (20%)
- **Number of Classes**: {num_classes}
- **Labels per Image**: 2-5 labels
- **Image Format**: RGB, 512x512 pixels
- **File Format**: Parquet

## Dataset Structure

Each sample contains:
- `image`: PIL Image (RGB, 512x512)
- `labels`: List of integer label IDs (multi-hot encoded)
- `label_names`: List of string class names
- `num_labels`: Number of labels in the image (2-5)

## Usage

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("{repo_id}")

# Access splits
train_data = dataset['train']
val_data = dataset['validation']

# Example: Get first training sample
sample = train_data[0]
print(f"Image: {{sample['image']}}")
print(f"Labels: {{sample['label_names']}}")
print(f"Label IDs: {{sample['labels']}}")
```

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{{multi_label_food_recognition,
  title={{Multi-Label Food Recognition Dataset}},
  author={{Your Name}},
  year={{2024}},
  url={{https://huggingface.co/datasets/{repo_id}}}
}}
```

## License

MIT License
"""
    return card


def main():
    parser = argparse.ArgumentParser(
        description='Upload multi-label dataset to HuggingFace Hub'
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        required=True,
        help='Path to multilabel dataset directory'
    )
    parser.add_argument(
        '--repo_id',
        type=str,
        required=True,
        help='HuggingFace repository ID (e.g., username/dataset-name)'
    )
    parser.add_argument(
        '--split_ratio',
        type=float,
        default=0.8,
        help='Train/validation split ratio (default: 0.8 for 80/20)'
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help='Make the dataset private'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Dry run without actually uploading'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for splitting (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Paths
    dataset_dir = Path(args.dataset_dir)
    annotations_path = dataset_dir / 'annotations' / 'labels.json'
    images_dir = dataset_dir / 'images'
    metadata_path = dataset_dir / 'metadata' / 'dataset_info.json'
    
    # Validate paths
    if not annotations_path.exists():
        print(f"Error: {annotations_path} not found!")
        return
    
    if not images_dir.exists():
        print(f"Error: {images_dir} not found!")
        return
    
    # Load metadata
    num_classes = None
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            num_classes = metadata.get('num_classes')
    
    # Load annotations
    annotations = load_annotations(annotations_path)
    
    # Verify images exist (but don't load them yet)
    print("\nVerifying images exist...")
    valid_keys = []
    for key in tqdm(annotations.keys(), desc="Checking images"):
        image_path = images_dir / key
        if image_path.exists():
            valid_keys.append(key)
        else:
            print(f"Warning: Image not found: {image_path}")
    
    # Filter annotations to only include valid images
    annotations = {k: v for k, v in annotations.items() if k in valid_keys}
    
    if len(annotations) == 0:
        print("Error: No valid image-annotation pairs found!")
        return
    
    print(f"✓ Found {len(annotations)} valid image-annotation pairs")
    
    # Split dataset (returns keys, not loaded images)
    train_keys, val_keys = create_dataset_split(
        annotations,
        images_dir,
        split_ratio=args.split_ratio,
        seed=args.seed
    )
    
    # Infer num_classes from data if not in metadata
    if num_classes is None:
        all_labels = set()
        for ann in annotations.values():
            all_labels.update(ann['labels'])
        num_classes = max(all_labels) + 1 if all_labels else len(set().union(*[ann['labels'] for ann in annotations.values()]))
        print(f"Inferred {num_classes} classes from data")
    
    # Create HuggingFace dataset (loads images on-demand)
    dataset_dict = create_huggingface_dataset(
        train_keys,
        val_keys,
        annotations,
        images_dir,
        num_classes,
        batch_size=500  # Smaller batch size to reduce memory usage
    )
    
    # Get sizes from dataset dict
    train_size = len(dataset_dict['train'])
    val_size = len(dataset_dict['validation'])
    
    # Create dataset card
    dataset_card = create_dataset_card(
        repo_id=args.repo_id,
        num_classes=num_classes,
        train_size=train_size,
        val_size=val_size
    )
    
    # Save dataset card locally
    card_path = dataset_dir / 'README.md'
    with open(card_path, 'w', encoding='utf-8') as f:
        f.write(dataset_card)
    print(f"\n✓ Saved dataset card to {card_path}")
    
    # Upload to HuggingFace
    if not args.dry_run:
        # Check if logged in
        try:
            api = HfApi()
            api.whoami()
        except Exception:
            print("\n⚠ Not logged in to HuggingFace. Please login:")
            print("  huggingface-cli login")
            print("  or")
            print("  from huggingface_hub import login; login()")
            return
        
        repo_url = upload_to_huggingface(
            dataset_dict,
            repo_id=args.repo_id,
            private=args.private,
            push_to_hub=True
        )
        
        # Upload README
        print("\nUploading dataset card (README.md)...")
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset"
        )
        print("✓ Dataset card uploaded")
    else:
        print("\n[DRY RUN] Skipping upload")
        print(f"Would upload to: https://huggingface.co/datasets/{args.repo_id}")
    
    print("\n" + "="*70)
    print("Complete!")
    print("="*70)


if __name__ == '__main__':
    main()

