"""
Script to upload merged food recognition dataset to Hugging Face Hub

This script processes images in batches to avoid memory issues:
1. Scans images from the merged_dataset folder (only paths, not loading images)
2. Creates train/validation splits (80/20 ratio) based on file paths
3. Processes images in batches and creates HuggingFace dataset incrementally
4. Converts to parquet format
5. Pushes to HuggingFace Hub as a private repository

Usage:
    python upload_dataset_to_hf.py --dataset_name "your-username/dataset-name" --data_dir "merged_dataset" --batch_size 1000
    
Requirements:
    pip install datasets pillow huggingface_hub pyarrow
"""

import os
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Generator
from PIL import Image
from datasets import Dataset, DatasetDict, Features, Value, load_dataset, concatenate_datasets
from huggingface_hub import HfApi, login
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import gc


def get_image_files(data_dir: str) -> List[Tuple[str, str]]:
    """
    Get all image files from the dataset directory.
    
    Args:
        data_dir: Path to the merged_dataset directory
        
    Returns:
        List of tuples (image_path, label)
    """
    image_files = []
    data_path = Path(data_dir)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
    # Get all class folders
    class_folders = [d for d in data_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(class_folders)} classes")
    
    for class_folder in tqdm(class_folders, desc="Scanning classes"):
        label = class_folder.name
        
        # Get all image files in this class folder
        for img_file in class_folder.iterdir():
            if img_file.suffix.lower() in image_extensions:
                image_files.append((str(img_file), label))
    
    print(f"Total images found: {len(image_files)}")
    return image_files


def process_batch(image_paths: List[str], labels: List[str], label_ids: List[int], batch_num: int) -> Dict:
    """
    Process a batch of images and return a dictionary with image data.
    Uses smaller batch processing to avoid memory issues.
    
    Args:
        image_paths: List of image file paths
        labels: List of string labels
        label_ids: List of integer label IDs
        batch_num: Batch number for progress tracking
        
    Returns:
        Dictionary with 'image', 'label', 'label_id' keys
    """
    images = []
    valid_labels = []
    valid_label_ids = []
    
    # Process in micro-batches to avoid memory spikes
    micro_batch_size = 50
    
    for i in range(0, len(image_paths), micro_batch_size):
        micro_batch_paths = image_paths[i:i+micro_batch_size]
        micro_batch_labels = labels[i:i+micro_batch_size]
        micro_batch_label_ids = label_ids[i:i+micro_batch_size]
        
        for img_path, label, label_id in zip(micro_batch_paths, micro_batch_labels, micro_batch_label_ids):
            try:
                # Load image using PIL with memory optimization
                img = Image.open(img_path)
                # Convert to RGB and resize if too large (memory optimization)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Optionally resize very large images to save memory
                max_size = 1024
                if max(img.size) > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                images.append(img)
                valid_labels.append(label)
                valid_label_ids.append(label_id)
            except Exception as e:
                print(f"Warning: Could not load image {img_path}: {e}")
                continue
        
        # Force garbage collection after each micro-batch
        if i % (micro_batch_size * 2) == 0:
            gc.collect()
    
    return {
        'image': images,
        'label': valid_labels,
        'label_id': valid_label_ids
    }


def create_dataset_batched(
    image_files: List[Tuple[str, str]], 
    train_ratio: float = 0.8, 
    seed: int = 42,
    batch_size: int = 1000
) -> DatasetDict:
    """
    Create a HuggingFace dataset from image files in batches to avoid memory issues.
    
    Args:
        image_files: List of tuples (image_path, label)
        train_ratio: Ratio for training split (default: 0.8)
        seed: Random seed for reproducibility
        batch_size: Number of images to process at once (default: 1000)
        
    Returns:
        DatasetDict with 'train' and 'validation' splits
    """
    print("\nCreating dataset in batches...")
    print(f"Batch size: {batch_size} images per batch")
    
    # Convert to DataFrame for easier manipulation (only paths, not loading images)
    df = pd.DataFrame(image_files, columns=['image_path', 'label'])
    
    # Get unique labels and create label mapping
    unique_labels = sorted(df['label'].unique())
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    
    print(f"Number of classes: {len(unique_labels)}")
    
    # Add label_id column
    df['label_id'] = df['label'].map(label_to_id)
    
    # Split into train and validation
    # Stratified split to maintain class distribution
    train_df, val_df = train_test_split(
        df,
        test_size=1 - train_ratio,
        random_state=seed,
        stratify=df['label_id']
    )
    
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    def create_hf_dataset_batched(df_split: pd.DataFrame, split_name: str) -> Dataset:
        """Create a HuggingFace dataset from a DataFrame in batches using incremental approach."""
        # Use a more memory-efficient approach: create dataset incrementally
        datasets_list = []
        
        # Process in smaller batches to avoid memory accumulation
        effective_batch_size = min(batch_size, 500)  # Cap at 500 for memory safety
        num_batches = (len(df_split) + effective_batch_size - 1) // effective_batch_size
        
        print(f"  Processing {num_batches} batches of ~{effective_batch_size} images each")
        
        for batch_idx in tqdm(range(num_batches), desc=f"Processing {split_name}"):
            start_idx = batch_idx * effective_batch_size
            end_idx = min((batch_idx + 1) * effective_batch_size, len(df_split))
            
            batch_df = df_split.iloc[start_idx:end_idx]
            
            # Process this batch
            batch_data = process_batch(
                batch_df['image_path'].tolist(),
                batch_df['label'].tolist(),
                batch_df['label_id'].tolist(),
                batch_idx + 1
            )
            
            if len(batch_data['image']) > 0:
                # Create a small dataset from this batch
                batch_dataset = Dataset.from_dict(batch_data)
                datasets_list.append(batch_dataset)
                
                # Clear batch data immediately
                del batch_data
                gc.collect()
        
        print(f"  Concatenating {len(datasets_list)} batch datasets...")
        
        # Concatenate all batch datasets
        if len(datasets_list) == 0:
            raise ValueError(f"No images were successfully loaded for {split_name}")
        elif len(datasets_list) == 1:
            final_dataset = datasets_list[0]
        else:
            # Concatenate incrementally to save memory
            final_dataset = datasets_list[0]
            for ds in datasets_list[1:]:
                final_dataset = concatenate_datasets([final_dataset, ds])
                gc.collect()
        
        # Clear remaining references
        del datasets_list
        gc.collect()
        
        print(f"  ✓ Created dataset with {len(final_dataset)} images")
        return final_dataset
    
    print("\nProcessing train split...")
    train_dataset = create_hf_dataset_batched(train_df, "train")
    
    print("\nProcessing validation split...")
    val_dataset = create_hf_dataset_batched(val_df, "validation")
    
    # Create DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })
    
    # Note: Metadata can be added via README.md when uploading to HuggingFace Hub
    # DatasetDict doesn't support direct info attribute assignment
    
    return dataset_dict, unique_labels, id_to_label


def create_dataset_imagefolder(
    data_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42,
    temp_dir: str = "temp_dataset"
) -> DatasetDict:
    """
    Create dataset using HuggingFace's imagefolder format (more memory efficient).
    This method creates temporary train/val folders and uses HF's built-in loader.
    
    Args:
        data_dir: Path to merged_dataset directory
        train_ratio: Ratio for training split (default: 0.8)
        seed: Random seed for reproducibility
        temp_dir: Temporary directory to create train/val structure
        
    Returns:
        DatasetDict with 'train' and 'validation' splits
    """
    print("\nUsing HuggingFace imagefolder format (memory efficient)...")
    print("This will create temporary train/val folders...")
    
    from sklearn.model_selection import train_test_split
    import shutil
    
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
    
    # Process each class
    for class_folder in tqdm(class_folders, desc="Organizing images"):
        label = class_folder.name
        
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
        
        # Copy files (create symlinks would be faster but less portable)
        for img_file in train_files:
            shutil.copy2(img_file, train_class_dir / img_file.name)
        
        for img_file in val_files:
            shutil.copy2(img_file, val_class_dir / img_file.name)
    
    print("\nLoading dataset using imagefolder format...")
    # Load using imagefolder format - it automatically detects train/val folders
    try:
        dataset_dict = load_dataset(
            "imagefolder",
            data_dir=str(temp_path)
        )
    except Exception as e:
        print(f"Error loading imagefolder dataset: {e}")
        print("Falling back to manual loading...")
        # Fallback: load train and val separately
        train_ds = load_dataset("imagefolder", data_dir=str(train_path), split="train")
        val_ds = load_dataset("imagefolder", data_dir=str(val_path), split="train")
        dataset_dict = DatasetDict({
            "train": train_ds,
            "validation": val_ds
        })
    
    # Get class names
    class_names = sorted(dataset_dict['train'].features['label'].names)
    id_to_label = {idx: name for idx, name in enumerate(class_names)}
    
    # Add label_id column
    def add_label_id(example):
        example['label_id'] = example['label']
        example['label'] = class_names[example['label']]
        return example
    
    dataset_dict = dataset_dict.map(add_label_id)
    
    print(f"\n✓ Dataset created successfully!")
    print(f"  Train samples: {len(dataset_dict['train'])}")
    print(f"  Validation samples: {len(dataset_dict['validation'])}")
    print(f"  Classes: {len(class_names)}")
    
    return dataset_dict, class_names, id_to_label


def upload_to_hub(
    dataset_dict: DatasetDict,
    dataset_name: str,
    private: bool = True,
    repo_type: str = "dataset",
    readme_path: str = None,
    hf_token: str = None
):
    """
    Upload dataset to HuggingFace Hub.
    
    Args:
        dataset_dict: DatasetDict to upload
        dataset_name: Name of the dataset (format: "username/dataset-name")
        private: Whether the repository should be private
        repo_type: Type of repository ("dataset" or "model")
        readme_path: Path to README.md file to upload (optional)
        hf_token: HuggingFace token (optional, uses environment or login if not provided)
    """
    print(f"\nUploading dataset to HuggingFace Hub: {dataset_name}")
    print(f"Private: {private}")
    
    # Get token from parameter or environment
    token = hf_token or os.environ.get('HF_TOKEN')
    
    # Push to hub with parquet format
    # Note: repo_type is not needed for DatasetDict.push_to_hub() - datasets are always "dataset" type
    dataset_dict.push_to_hub(
        repo_id=dataset_name,
        private=private,
        token=token
    )
    
    # Upload README if provided
    if readme_path and os.path.exists(readme_path):
        print(f"\nUploading README from: {readme_path}")
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=dataset_name,
            repo_type=repo_type,
            token=token
        )
        print("✓ README uploaded successfully")
    
    print(f"\n✓ Dataset uploaded successfully!")
    print(f"✓ View at: https://huggingface.co/datasets/{dataset_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload merged food recognition dataset to HuggingFace Hub"
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
        "--batch_size",
        type=int,
        default=500,
        help="Number of images to process per batch (default: 500). Reduce to 100-200 if memory issues occur."
    )
    parser.add_argument(
        "--use_imagefolder",
        action="store_true",
        help="Use HuggingFace's imagefolder format (more memory efficient, requires temporary folder structure)"
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        default="temp_dataset",
        help="Temporary directory for imagefolder format (default: 'temp_dataset')"
    )
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found!")
        return
    
    # Check HuggingFace login
    # Try to get token from args, environment variable, or existing login
    hf_token = args.hf_token or os.environ.get('HF_TOKEN')
    
    try:
        api = HfApi(token=hf_token)
        user = api.whoami(token=hf_token)
        print(f"✓ Logged in as: {user['name']}")
        # Set token in environment for later use
        if hf_token:
            os.environ['HF_TOKEN'] = hf_token
    except Exception as e:
        print("⚠ Error: Not logged in to HuggingFace Hub or invalid token!")
        print("\nTroubleshooting steps:")
        print("1. Generate a new token at: https://huggingface.co/settings/tokens")
        print("   - Make sure to select 'Write' permissions")
        print("2. Use one of these methods to provide the token:")
        print("   Method 1: python upload_dataset_to_hf.py --hf_token YOUR_TOKEN --dataset_name ...")
        print("   Method 2: Set environment variable: $env:HF_TOKEN='your_token' (PowerShell)")
        print("   Method 3: Set environment variable: set HF_TOKEN=your_token (CMD)")
        print("   Method 4: Try: python -c 'from huggingface_hub import login; login()'")
        print("\nIf the error persists:")
        print("- Check your internet connection")
        print("- Verify the token is correct (no extra spaces, starts with 'hf_')")
        print("- Make sure the token has 'Write' permissions")
        print("- Try generating a new token")
        print(f"\nError details: {str(e)}")
        return
    
    # Create dataset using appropriate method
    if args.use_imagefolder:
        # Use imagefolder format (more memory efficient)
        dataset_dict, class_names, id_to_label = create_dataset_imagefolder(
            args.data_dir,
            train_ratio=args.train_ratio,
            seed=args.seed,
            temp_dir=args.temp_dir
        )
    else:
        # Use batch processing method
        print("Using batch processing method...")
        image_files = get_image_files(args.data_dir)
        
        if len(image_files) == 0:
            print("Error: No images found in the dataset directory!")
            return
        
        dataset_dict, class_names, id_to_label = create_dataset_batched(
            image_files,
            train_ratio=args.train_ratio,
            seed=args.seed,
            batch_size=args.batch_size
        )
    
    # Determine privacy setting
    is_private = args.private and not args.public
    
    # Upload to hub
    upload_to_hub(
        dataset_dict,
        dataset_name=args.dataset_name,
        private=is_private,
        readme_path=args.readme if os.path.exists(args.readme) else None,
        hf_token=hf_token
    )
    
    # Save class mapping
    class_mapping_file = "class_mapping.json"
    import json
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
    
    # Clean up temp directory if using imagefolder format
    if args.use_imagefolder and os.path.exists(args.temp_dir):
        print(f"\nCleaning up temporary directory: {args.temp_dir}")
        import shutil
        shutil.rmtree(args.temp_dir)
        print("✓ Cleanup complete")


if __name__ == "__main__":
    main()

