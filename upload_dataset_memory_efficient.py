"""
Memory-Efficient Script to upload merged food recognition dataset to Hugging Face Hub

This script uses HuggingFace's imagefolder format which is much more memory efficient.
It creates temporary train/val folders and uses HF's optimized loader.

Usage:
    python upload_dataset_memory_efficient.py --dataset_name "your-username/dataset-name" --data_dir "merged_dataset"
    
Requirements:
    pip install datasets pillow huggingface_hub pyarrow
"""

import os
import argparse
from pathlib import Path
from typing import List
from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil
import gc


def create_train_val_structure(
    data_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42,
    temp_dir: str = "temp_dataset"
) -> tuple:
    """
    Create train/val folder structure for imagefolder format.
    Uses hard links when possible to save disk space.
    
    Args:
        data_dir: Path to merged_dataset directory
        train_ratio: Ratio for training split (default: 0.8)
        seed: Random seed for reproducibility
        temp_dir: Temporary directory to create train/val structure
        
    Returns:
        Tuple of (class_names, id_to_label dict)
    """
    print("\nCreating train/validation folder structure...")
    print("This method is memory-efficient as it doesn't load images into RAM.")
    
    data_path = Path(data_dir)
    temp_path = Path(temp_dir)
    
    # Create temporary directory structure
    train_path = temp_path / "train"
    val_path = temp_path / "validation"
    
    # Clean up if exists
    if temp_path.exists():
        print(f"Cleaning up existing temp directory: {temp_dir}")
        shutil.rmtree(temp_path)
    
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)
    
    # Get all class folders
    class_folders = [d for d in data_path.iterdir() if d.is_dir()]
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
    print(f"Found {len(class_folders)} classes")
    
    class_names = []
    total_train = 0
    total_val = 0
    
    # Process each class
    for class_folder in tqdm(class_folders, desc="Organizing images"):
        label = class_folder.name
        class_names.append(label)
        
        # Get all image files
        image_files = [f for f in class_folder.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if len(image_files) == 0:
            continue
        
        # Split images for this class
        train_files, val_files = train_test_split(
            image_files,
            test_size=1 - train_ratio,
            random_state=seed
        )
        
        # Create class directories in train and val
        train_class_dir = train_path / label
        val_class_dir = val_path / label
        train_class_dir.mkdir(parents=True, exist_ok=True)
        val_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files (using copy2 preserves metadata)
        # For very large datasets, consider using hard links instead
        for img_file in train_files:
            try:
                shutil.copy2(img_file, train_class_dir / img_file.name)
            except Exception as e:
                print(f"Warning: Could not copy {img_file}: {e}")
        
        for img_file in val_files:
            try:
                shutil.copy2(img_file, val_class_dir / img_file.name)
            except Exception as e:
                print(f"Warning: Could not copy {img_file}: {e}")
        
        total_train += len(train_files)
        total_val += len(val_files)
    
    class_names = sorted(class_names)
    id_to_label = {idx: name for idx, name in enumerate(class_names)}
    
    print(f"\n✓ Folder structure created:")
    print(f"  Train images: {total_train}")
    print(f"  Validation images: {total_val}")
    print(f"  Classes: {len(class_names)}")
    
    return class_names, id_to_label


def load_imagefolder_dataset(temp_dir: str) -> DatasetDict:
    """
    Load dataset using HuggingFace's imagefolder format.
    
    Args:
        temp_dir: Path to temporary directory with train/val folders
        
    Returns:
        DatasetDict with 'train' and 'validation' splits
    """
    print("\nLoading dataset using imagefolder format...")
    print("This is memory-efficient as images are loaded on-demand.")
    
    try:
        # Load using imagefolder format - automatically detects train/val folders
        dataset_dict = load_dataset(
            "imagefolder",
            data_dir=str(temp_dir)
        )
        
        # Ensure we have train and validation splits
        if "train" not in dataset_dict or "validation" not in dataset_dict:
            raise ValueError("Dataset must have both 'train' and 'validation' splits")
        
    except Exception as e:
        print(f"Error loading imagefolder dataset: {e}")
        print("Trying to load train and val separately...")
        
        # Fallback: load train and val separately
        train_path = Path(temp_dir) / "train"
        val_path = Path(temp_dir) / "validation"
        
        train_ds = load_dataset("imagefolder", data_dir=str(train_path), split="train")
        val_ds = load_dataset("imagefolder", data_dir=str(val_path), split="train")
        
        dataset_dict = DatasetDict({
            "train": train_ds,
            "validation": val_ds
        })
    
    print(f"✓ Dataset loaded:")
    print(f"  Train samples: {len(dataset_dict['train'])}")
    print(f"  Validation samples: {len(dataset_dict['validation'])}")
    
    return dataset_dict


def add_label_columns(dataset_dict: DatasetDict, class_names: List[str]) -> DatasetDict:
    """
    Add label string and label_id columns to the dataset.
    
    Args:
        dataset_dict: DatasetDict with imagefolder format
        class_names: List of class names
        
    Returns:
        Updated DatasetDict with label and label_id columns
    """
    print("\nAdding label columns...")
    
    # Get label names from dataset features
    label_names = dataset_dict['train'].features['label'].names
    
    def add_labels(example):
        label_id = example['label']
        example['label_id'] = label_id
        example['label'] = label_names[label_id]
        return example
    
    dataset_dict = dataset_dict.map(add_labels)
    
    print("✓ Label columns added")
    return dataset_dict


def upload_to_hub(
    dataset_dict: DatasetDict,
    dataset_name: str,
    private: bool = True,
    readme_path: str = None,
    hf_token: str = None
):
    """
    Upload dataset to HuggingFace Hub.
    
    Args:
        dataset_dict: DatasetDict to upload
        dataset_name: Name of the dataset (format: "username/dataset-name")
        private: Whether the repository should be private
        readme_path: Path to README.md file to upload (optional)
        hf_token: HuggingFace token (optional)
    """
    print(f"\nUploading dataset to HuggingFace Hub: {dataset_name}")
    print(f"Private: {private}")
    
    # Get token from parameter or environment
    token = hf_token or os.environ.get('HF_TOKEN')
    
    # Push to hub - HuggingFace will convert to parquet automatically
    print("Pushing dataset to hub (this may take a while)...")
    dataset_dict.push_to_hub(
        repo_id=dataset_name,
        private=private,
        repo_type="dataset",
        token=token
    )
    
    # Upload README if provided
    if readme_path and os.path.exists(readme_path):
        print(f"\nUploading README from: {readme_path}")
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=dataset_name,
            repo_type="dataset",
            token=token
        )
        print("✓ README uploaded successfully")
    
    print(f"\n✓ Dataset uploaded successfully!")
    print(f"✓ View at: https://huggingface.co/datasets/{dataset_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Memory-efficient upload of merged food recognition dataset to HuggingFace Hub"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="HuggingFace dataset name (format: 'username/dataset-name')"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="merged_dataset",
        help="Path to merged_dataset directory (default: 'merged_dataset')"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio for training split (default: 0.8)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=True,
        help="Make the repository private (default: True)"
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make the repository public (overrides --private)"
    )
    parser.add_argument(
        "--readme",
        type=str,
        default="DATASET_README.md",
        help="Path to README.md file to upload (default: 'DATASET_README.md')"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token (alternative to login). Can also use HF_TOKEN environment variable."
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        default="temp_dataset",
        help="Temporary directory for train/val structure (default: 'temp_dataset')"
    )
    parser.add_argument(
        "--keep_temp",
        action="store_true",
        help="Keep temporary directory after upload (default: False)"
    )
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found!")
        return
    
    # Check HuggingFace login
    hf_token = args.hf_token or os.environ.get('HF_TOKEN')
    
    try:
        api = HfApi(token=hf_token)
        user = api.whoami(token=hf_token)
        print(f"✓ Logged in as: {user['name']}")
        if hf_token:
            os.environ['HF_TOKEN'] = hf_token
    except Exception as e:
        print("⚠ Error: Not logged in to HuggingFace Hub or invalid token!")
        print("\nTroubleshooting steps:")
        print("1. Generate a new token at: https://huggingface.co/settings/tokens")
        print("   - Make sure to select 'Write' permissions")
        print("2. Use: python upload_dataset_memory_efficient.py --hf_token YOUR_TOKEN --dataset_name ...")
        print(f"\nError details: {str(e)}")
        return
    
    try:
        # Step 1: Create train/val folder structure
        class_names, id_to_label = create_train_val_structure(
            args.data_dir,
            train_ratio=args.train_ratio,
            seed=args.seed,
            temp_dir=args.temp_dir
        )
        
        # Step 2: Load dataset using imagefolder format
        dataset_dict = load_imagefolder_dataset(args.temp_dir)
        
        # Step 3: Add label columns
        dataset_dict = add_label_columns(dataset_dict, class_names)
        
        # Step 4: Upload to hub
        is_private = args.private and not args.public
        upload_to_hub(
            dataset_dict,
            dataset_name=args.dataset_name,
            private=is_private,
            readme_path=args.readme if os.path.exists(args.readme) else None,
            hf_token=hf_token
        )
        
        # Step 5: Clean up temp directory
        if not args.keep_temp and os.path.exists(args.temp_dir):
            print(f"\nCleaning up temporary directory: {args.temp_dir}")
            shutil.rmtree(args.temp_dir)
            print("✓ Cleanup complete")
        
        # Save class mapping
        import json
        class_mapping_file = "class_mapping.json"
        with open(class_mapping_file, 'w') as f:
            json.dump({
                'id_to_label': id_to_label,
                'label_to_id': {label: idx for idx, label in id_to_label.items()},
                'class_names': class_names,
                'num_classes': len(class_names)
            }, f, indent=2)
        
        print(f"\n✓ Class mapping saved to: {class_mapping_file}")
        print(f"\nDataset Statistics:")
        print(f"  Total classes: {len(class_names)}")
        print(f"  Train samples: {len(dataset_dict['train'])}")
        print(f"  Validation samples: {len(dataset_dict['validation'])}")
        print(f"  Train ratio: {args.train_ratio}")
        print(f"  Validation ratio: {1 - args.train_ratio}")
        
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

