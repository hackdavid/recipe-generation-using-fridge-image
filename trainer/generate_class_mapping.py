"""
Script to generate class mapping JSON file from dataset.
Run this BEFORE training to create a fixed class mapping file.

Usage:
    python trainer/generate_class_mapping.py --dataset_name "ibrahimdaud/raw-food-recognition" --output trainer/class_mapping.json
    python trainer/generate_class_mapping.py --data_dir "./data" --output trainer/class_mapping.json
"""

import os
import json
import argparse
from datasets import load_dataset
from torchvision.datasets import ImageFolder
from typing import Dict, List, Optional


def generate_mapping_from_hf_dataset(
    dataset_name: str,
    split: str = 'train',
    label_key: str = 'label',
    label_id_key: str = 'label_id'
) -> Dict:
    """
    Generate class mapping from HuggingFace dataset.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        split: Dataset split to use (default: 'train')
        label_key: Key for string labels (default: 'label')
        label_id_key: Key for integer label IDs (default: 'label_id')
        
    Returns:
        Dictionary with mapping data
    """
    print(f"Loading HuggingFace dataset: {dataset_name}")
    print(f"Using split: {split}")
    
    try:
        dataset = load_dataset(dataset_name, split=split, streaming=False)
    except Exception as e:
        # Try loading as DatasetDict
        dataset_dict = load_dataset(dataset_name, streaming=False)
        if hasattr(dataset_dict, 'keys') and len(dataset_dict.keys()) > 0:
            if split in dataset_dict.keys():
                dataset = dataset_dict[split]
            else:
                # Use first available split
                first_split = next(iter(dataset_dict.keys()))
                print(f"Split '{split}' not found. Using '{first_split}' instead.")
                dataset = dataset_dict[first_split]
        else:
            dataset = dataset_dict
    
    print(f"Dataset loaded. Number of samples: {len(dataset)}")
    
    # Check dataset features
    print(f"Dataset features: {list(dataset.features.keys())}")
    
    # Strategy 1: Check if label_id exists and label has ClassLabel feature
    if label_id_key in dataset.features and label_key in dataset.features:
        label_feature = dataset.features[label_key]
        
        if hasattr(label_feature, 'names') and label_feature.names:
            # ClassLabel feature with names
            class_names = label_feature.names
            label_to_id = {name: idx for idx, name in enumerate(class_names)}
            id_to_label = {idx: name for idx, name in enumerate(class_names)}
            
            print(f"✓ Found ClassLabel feature with {len(class_names)} classes")
            return {
                'label_to_id': label_to_id,
                'id_to_label': id_to_label,
                'class_names': class_names,
                'num_classes': len(class_names),
                'dataset_name': dataset_name,
                'source': 'huggingface_classlabel'
            }
    
    # Strategy 2: Check if label_id exists, sample to get unique labels
    if label_id_key in dataset.features:
        print("Found 'label_id' column. Sampling to get class names...")
        
        # Sample dataset to get unique labels
        unique_labels = set()
        label_id_to_label = {}
        
        sample_size = min(10000, len(dataset))  # Sample up to 10k examples
        for i, example in enumerate(dataset):
            if i >= sample_size:
                break
            
            if label_key in example and label_id_key in example:
                label_str = example[label_key]
                label_id = example[label_id_key]
                
                if isinstance(label_str, str):
                    unique_labels.add(label_str)
                    label_id_to_label[label_id] = label_str
        
        if unique_labels:
            # Sort labels to ensure consistent ordering
            class_names = sorted(unique_labels)
            label_to_id = {name: idx for idx, name in enumerate(class_names)}
            id_to_label = {idx: name for idx, name in enumerate(class_names)}
            
            print(f"✓ Created mapping from {len(class_names)} unique labels")
            return {
                'label_to_id': label_to_id,
                'id_to_label': id_to_label,
                'class_names': class_names,
                'num_classes': len(class_names),
                'dataset_name': dataset_name,
                'source': 'huggingface_sampled'
            }
    
    # Strategy 3: Check if label is ClassLabel feature
    if label_key in dataset.features:
        label_feature = dataset.features[label_key]
        
        if hasattr(label_feature, 'names') and label_feature.names:
            class_names = label_feature.names
            label_to_id = {name: idx for idx, name in enumerate(class_names)}
            id_to_label = {idx: name for idx, name in enumerate(class_names)}
            
            print(f"✓ Found ClassLabel feature with {len(class_names)} classes")
            return {
                'label_to_id': label_to_id,
                'id_to_label': id_to_label,
                'class_names': class_names,
                'num_classes': len(class_names),
                'dataset_name': dataset_name,
                'source': 'huggingface_classlabel'
            }
    
    # Strategy 4: Sample string labels and create mapping
    print("Sampling dataset to extract unique string labels...")
    unique_labels = set()
    
    sample_size = min(10000, len(dataset))
    for i, example in enumerate(dataset):
        if i >= sample_size:
            break
        
        if label_key in example:
            label_val = example[label_key]
            if isinstance(label_val, str):
                unique_labels.add(label_val)
    
    if unique_labels:
        class_names = sorted(unique_labels)
        label_to_id = {name: idx for idx, name in enumerate(class_names)}
        id_to_label = {idx: name for idx, name in enumerate(class_names)}
        
        print(f"✓ Created mapping from {len(class_names)} unique string labels")
        return {
            'label_to_id': label_to_id,
            'id_to_label': id_to_label,
            'class_names': class_names,
            'num_classes': len(class_names),
            'dataset_name': dataset_name,
            'source': 'huggingface_sampled_strings'
        }
    
    raise ValueError(
        f"Could not determine class mapping from dataset. "
        f"Please ensure the dataset has '{label_key}' or '{label_id_key}' columns."
    )


def generate_mapping_from_folder(data_dir: str) -> Dict:
    """
    Generate class mapping from local ImageFolder dataset.
    
    Args:
        data_dir: Path to dataset directory
        
    Returns:
        Dictionary with mapping data
    """
    print(f"Loading dataset from folder: {data_dir}")
    
    # Check for train folder first
    train_dir = os.path.join(data_dir, 'train')
    if os.path.exists(train_dir):
        dataset = ImageFolder(train_dir)
    elif os.path.exists(data_dir):
        dataset = ImageFolder(data_dir)
    else:
        raise ValueError(f"Dataset directory not found: {data_dir}")
    
    class_names = dataset.classes
    label_to_id = {name: idx for idx, name in enumerate(class_names)}
    id_to_label = {idx: name for idx, name in enumerate(class_names)}
    
    print(f"✓ Found {len(class_names)} classes")
    
    return {
        'label_to_id': label_to_id,
        'id_to_label': id_to_label,
        'class_names': class_names,
        'num_classes': len(class_names),
        'dataset_name': None,
        'source': 'imagefolder'
    }


def save_class_mapping(mapping_data: Dict, output_path: str):
    """
    Save class mapping to JSON file.
    
    Args:
        mapping_data: Dictionary containing mapping data
        output_path: Path where to save the JSON file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Add metadata
    mapping_data['metadata'] = {
        'description': 'Class mapping for ingredient recognition model',
        'format': {
            'label_to_id': 'Dictionary mapping class name (string) to integer ID',
            'id_to_label': 'Dictionary mapping integer ID to class name (string)',
            'class_names': 'List of class names in order (index corresponds to ID)'
        }
    }
    
    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Class mapping saved to: {output_path}")
    print(f"  Number of classes: {mapping_data['num_classes']}")
    print(f"  Source: {mapping_data['source']}")
    if mapping_data.get('dataset_name'):
        print(f"  Dataset: {mapping_data['dataset_name']}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate class mapping JSON file from dataset'
    )
    
    # Dataset source options
    parser.add_argument(
        '--dataset_name',
        type=str,
        help='HuggingFace dataset name (e.g., "ibrahimdaud/raw-food-recognition")'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        help='Local dataset directory path (for ImageFolder datasets)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        help='Dataset split to use for HuggingFace datasets (default: train)'
    )
    
    # Output
    parser.add_argument(
        '--output',
        type=str,
        default='trainer/class_mapping.json',
        help='Output path for class mapping JSON file (default: trainer/class_mapping.json)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.dataset_name and not args.data_dir:
        parser.error("Either --dataset_name or --data_dir must be provided")
    
    if args.dataset_name and args.data_dir:
        parser.error("Provide either --dataset_name OR --data_dir, not both")
    
    # Generate mapping
    try:
        if args.dataset_name:
            mapping_data = generate_mapping_from_hf_dataset(
                dataset_name=args.dataset_name,
                split=args.split
            )
        else:
            mapping_data = generate_mapping_from_folder(data_dir=args.data_dir)
        
        # Save mapping
        save_class_mapping(mapping_data, args.output)
        
        print("\n✓ Class mapping generation completed successfully!")
        print(f"\nYou can now use this mapping file in your training pipeline:")
        print(f"  from trainer.hf_dataset import load_class_mapping")
        print(f"  mapping = load_class_mapping('{args.output}')")
        
    except Exception as e:
        print(f"\n✗ Error generating class mapping: {e}")
        raise


if __name__ == '__main__':
    main()

