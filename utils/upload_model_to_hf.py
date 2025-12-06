"""
Upload Trained Model Checkpoints to HuggingFace Hub

This script uploads PyTorch model checkpoints to HuggingFace Hub for easy sharing and inference.
Supports uploading a single model or both ResNet-50 and SE-ResNet-50 to a single repository.

Usage (Single Model):
    python utils/upload_model_to_hf.py \
        --checkpoint ./checkpoints/resnet50_best.pth \
        --repo_id "your-username/resnet50-raw-food-recognition" \
        --private

Usage (Both Models - Single Repository):
    python utils/upload_model_to_hf.py --resnet_checkpoint resnet50_best.pth --se_resnet_checkpoint se_resnet50_best.pth --repo_id "ibrahimdaud/raw-food-recognition-models"
"""

import torch
import argparse
import json
import os
from pathlib import Path
from huggingface_hub import HfApi, login, create_repo, upload_file
from huggingface_hub.utils import RepositoryNotFoundError


def load_checkpoint(checkpoint_path):
    """Load checkpoint file"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint


def create_model_card(checkpoint, repo_id, model_type):
    """Create a model card (README.md) for HuggingFace"""
    
    num_classes = checkpoint.get('num_classes', 'Unknown')
    best_val_acc = checkpoint.get('best_val_acc', 'Unknown')
    epoch = checkpoint.get('epoch', 'Unknown')
    class_names = checkpoint.get('class_names', None)
    
    # Count parameters
    model_state = checkpoint['model_state_dict']
    total_params = sum(p.numel() for p in model_state.values())
    
    model_card = f"""---
license: mit
tags:
- image-classification
- food-recognition
- raw-food
- pytorch
- resnet
- computer-vision
datasets:
- ibrahimdaud/raw-food-recognition
metrics:
- accuracy
---

# {model_type.upper()} - Raw Food Recognition Model

This model is trained for raw food ingredient recognition using the merged raw food recognition dataset.

## Model Details

- **Model Type**: {model_type}
- **Number of Classes**: {num_classes}
- **Total Parameters**: {total_params:,}
- **Best Validation Accuracy**: {best_val_acc:.2f}% (if available)
- **Training Epoch**: {epoch}

## Dataset

This model was trained on the [ibrahimdaud/raw-food-recognition](https://huggingface.co/datasets/ibrahimdaud/raw-food-recognition) dataset, which contains 90+ raw food categories.

## Usage

### Using PyTorch

```python
import torch
from models.{model_type} import create_{model_type.replace('-', '_')}

# Load checkpoint
checkpoint = torch.load('pytorch_model.bin', map_location='cpu')

# Create model
model = create_{model_type.replace('-', '_')}(
    num_classes={num_classes},
    pretrained=False
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
# ... your inference code here
```

### Using with HuggingFace Transformers (if converted)

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained("{repo_id}")
model = AutoModelForImageClassification.from_pretrained("{repo_id}")
```

## Training Details

- **Architecture**: {model_type}
- **Pretrained**: ImageNet weights
- **Image Size**: 224x224
- **Optimizer**: Adam (default)
- **Learning Rate**: 0.001 (default)

## Evaluation

The model achieves {best_val_acc:.2f}% accuracy on the validation set.

## Limitations

- Trained specifically on raw food items
- May not generalize well to prepared or cooked foods
- Performance may vary for underrepresented classes

## Citation

If you use this model, please cite:

```bibtex
@model{{raw_food_recognition_{model_type}_2024,
  title={{Raw Food Recognition Model - {model_type}}},
  author={{Ibrahim Daud}},
  year={{2024}},
  publisher={{HuggingFace}},
  url={{https://huggingface.co/{repo_id}}}
}}
```

## License

MIT License
"""
    
    return model_card


def upload_model_to_hf(checkpoint_path, repo_id, private=False, token=None, model_name=None):
    """
    Upload model checkpoint to HuggingFace Hub
    
    Args:
        checkpoint_path: Path to .pth checkpoint file
        repo_id: HuggingFace repository ID (e.g., "username/model-name")
        private: Whether the repository should be private
        token: HuggingFace API token (optional, will prompt if not provided)
        model_name: Optional name for the model file (e.g., "resnet50" or "se_resnet50")
                    If None, uses model_type from checkpoint
    """
    
    # Login to HuggingFace (only once)
    if token:
        login(token=token)
    else:
        # Check if already logged in
        try:
            api = HfApi()
            api.whoami()
        except Exception:
            print("Please login to HuggingFace:")
            login()
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    
    # Extract model type
    model_type = checkpoint.get('model_type', 'resnet50')
    if model_type == 'se_resnet50':
        model_type = 'se-resnet50'
    
    # Determine filename
    if model_name:
        filename = f"{model_name}_pytorch_model.bin"
    else:
        filename = f"{model_type.replace('-', '_')}_pytorch_model.bin"
    
    # Create API instance
    api = HfApi()
    
    # Create repository if it doesn't exist
    try:
        api.repo_info(repo_id)
        print(f"Repository {repo_id} already exists. Uploading files...")
    except RepositoryNotFoundError:
        print(f"Creating repository: {repo_id}")
        create_repo(repo_id, private=private, repo_type="model")
    
    # Upload checkpoint
    print(f"\nUploading {model_type} model...")
    print(f"  - Uploading model weights ({filename})...")
    upload_file(
        path_or_fileobj=checkpoint_path,
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="model"
    )
    
    # Save and upload metadata for this model
    metadata = {
        'model_type': model_type,
        'num_classes': checkpoint.get('num_classes'),
        'best_val_acc': checkpoint.get('best_val_acc'),
        'epoch': checkpoint.get('epoch'),
        'class_names': checkpoint.get('class_names'),
        'checkpoint_keys': list(checkpoint.keys())
    }
    
    metadata_filename = f"{model_type.replace('-', '_')}_metadata.json"
    metadata_path = f"temp_{metadata_filename}"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  - Uploading metadata ({metadata_filename})...")
    upload_file(
        path_or_fileobj=metadata_path,
        path_in_repo=metadata_filename,
        repo_id=repo_id,
        repo_type="model"
    )
    
    # Clean up temp file
    os.remove(metadata_path)
    
    print(f"✓ {model_type} uploaded successfully!")
    return metadata


def upload_both_models(resnet_checkpoint, se_resnet_checkpoint, repo_id, private=False, token=None):
    """
    Upload both ResNet-50 and SE-ResNet-50 models to a single repository
    
    Args:
        resnet_checkpoint: Path to ResNet-50 checkpoint
        se_resnet_checkpoint: Path to SE-ResNet-50 checkpoint
        repo_id: HuggingFace repository ID
        private: Whether repository should be private
        token: HuggingFace API token
    """
    
    # Login once
    if token:
        login(token=token)
    else:
        try:
            api = HfApi()
            api.whoami()
        except Exception:
            print("Please login to HuggingFace:")
            login()
    
    # Create API instance
    api = HfApi()
    
    # Create repository if it doesn't exist
    try:
        api.repo_info(repo_id)
        print(f"Repository {repo_id} already exists. Uploading files...")
    except RepositoryNotFoundError:
        print(f"Creating repository: {repo_id}")
        create_repo(repo_id, private=private, repo_type="model")
    
    # Upload ResNet-50
    print("\n" + "="*70)
    print("Uploading ResNet-50 Model")
    print("="*70)
    resnet_metadata = upload_model_to_hf(
        resnet_checkpoint,
        repo_id,
        private=private,
        token=token,
        model_name="resnet50"
    )
    
    # Upload SE-ResNet-50
    print("\n" + "="*70)
    print("Uploading SE-ResNet-50 Model")
    print("="*70)
    se_resnet_metadata = upload_model_to_hf(
        se_resnet_checkpoint,
        repo_id,
        private=private,
        token=token,
        model_name="se_resnet50"
    )
    
    # Create combined model card
    model_card = create_combined_model_card(resnet_metadata, se_resnet_metadata, repo_id)
    
    # Upload combined README
    temp_readme = "temp_readme.md"
    with open(temp_readme, 'w', encoding='utf-8') as f:
        f.write(model_card)
    
    print("\n" + "="*70)
    print("Uploading Combined Model Card")
    print("="*70)
    upload_file(
        path_or_fileobj=temp_readme,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model"
    )
    os.remove(temp_readme)
    
    # Summary
    print("\n" + "="*70)
    print("Upload Summary")
    print("="*70)
    print(f"\n✓ Both models uploaded successfully!")
    print(f"\nRepository: https://huggingface.co/{repo_id}")
    print(f"\nAvailable files:")
    print(f"  - resnet50_pytorch_model.bin")
    print(f"  - se_resnet50_pytorch_model.bin")
    print(f"  - resnet50_metadata.json")
    print(f"  - se_resnet50_metadata.json")
    print(f"  - README.md")


def create_combined_model_card(resnet_metadata, se_resnet_metadata, repo_id):
    """Create a combined model card for both models"""
    
    resnet_acc = resnet_metadata.get('best_val_acc', 'Unknown')
    se_resnet_acc = se_resnet_metadata.get('best_val_acc', 'Unknown')
    num_classes = resnet_metadata.get('num_classes', 'Unknown')
    
    model_card = f"""---
license: mit
tags:
- image-classification
- food-recognition
- raw-food
- pytorch
- resnet
- se-resnet
- computer-vision
- model-comparison
datasets:
- ibrahimdaud/raw-food-recognition
metrics:
- accuracy
---

# Raw Food Recognition Models: ResNet-50 vs SE-ResNet-50

This repository contains both ResNet-50 and SE-ResNet-50 models trained for raw food ingredient recognition using the merged raw food recognition dataset.

## Model Comparison

| Model | Parameters | Validation Accuracy | Architecture |
|-------|-----------|-------------------|--------------|
| ResNet-50 | ~25.6M | {resnet_acc:.2f}% | Standard residual network |
| SE-ResNet-50 | ~26.0M | {se_resnet_acc:.2f}% | ResNet-50 with SE attention |

## Dataset

Both models were trained on the [ibrahimdaud/raw-food-recognition](https://huggingface.co/datasets/ibrahimdaud/raw-food-recognition) dataset, which contains 90+ raw food categories.

## Usage

### Download Both Models

```python
from huggingface_hub import hf_hub_download
import torch
from models.resnet50 import create_resnet50
from models.se_resnet50 import create_se_resnet50

# Download ResNet-50 checkpoint
resnet_path = hf_hub_download(
    repo_id="{repo_id}",
    filename="resnet50_pytorch_model.bin"
)
resnet_checkpoint = torch.load(resnet_path, map_location='cpu')

# Download SE-ResNet-50 checkpoint
se_resnet_path = hf_hub_download(
    repo_id="{repo_id}",
    filename="se_resnet50_pytorch_model.bin"
)
se_resnet_checkpoint = torch.load(se_resnet_path, map_location='cpu')
```

### Load ResNet-50

```python
# Create ResNet-50 model
resnet_model = create_resnet50(
    num_classes={num_classes},
    pretrained=False
)
resnet_model.load_state_dict(resnet_checkpoint['model_state_dict'])
resnet_model.eval()
```

### Load SE-ResNet-50

```python
# Create SE-ResNet-50 model
se_resnet_model = create_se_resnet50(
    num_classes={num_classes},
    pretrained=False,
    reduction=16
)
se_resnet_model.load_state_dict(se_resnet_checkpoint['model_state_dict'])
se_resnet_model.eval()
```

### Compare Predictions

```python
import torch
from PIL import Image
import torchvision.transforms as transforms

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

image = Image.open('path/to/image.jpg').convert('RGB')
image_tensor = transform(image).unsqueeze(0)

# ResNet-50 prediction
with torch.no_grad():
    resnet_outputs = resnet_model(image_tensor)
    resnet_probs = torch.nn.functional.softmax(resnet_outputs[0], dim=0)
    resnet_pred_id = torch.argmax(resnet_probs).item()
    resnet_pred_class = resnet_checkpoint['class_names'][resnet_pred_id]
    resnet_confidence = resnet_probs[resnet_pred_id].item()

# SE-ResNet-50 prediction
with torch.no_grad():
    se_resnet_outputs = se_resnet_model(image_tensor)
    se_resnet_probs = torch.nn.functional.softmax(se_resnet_outputs[0], dim=0)
    se_resnet_pred_id = torch.argmax(se_resnet_probs).item()
    se_resnet_pred_class = se_resnet_checkpoint['class_names'][se_resnet_pred_id]
    se_resnet_confidence = se_resnet_probs[se_resnet_pred_id].item()

# Compare results
print("ResNet-50 Prediction:")
print(f"  Class: {{resnet_pred_class}}")
print(f"  Confidence: {{resnet_confidence*100:.2f}}%")

print("\\nSE-ResNet-50 Prediction:")
print(f"  Class: {{se_resnet_pred_class}}")
print(f"  Confidence: {{se_resnet_confidence*100:.2f}}%")
```

## Model Details

### ResNet-50
- **Architecture**: Standard residual network with bottleneck blocks
- **Parameters**: ~25.6M
- **Pretrained**: ImageNet weights
- **Best Validation Accuracy**: {resnet_acc:.2f}%

### SE-ResNet-50
- **Architecture**: ResNet-50 with Squeeze-and-Excitation attention blocks
- **Parameters**: ~26.0M
- **Pretrained**: ImageNet weights (excluding SE blocks)
- **SE Reduction Ratio**: 16
- **Best Validation Accuracy**: {se_resnet_acc:.2f}%

## Training Details

- **Dataset**: ibrahimdaud/raw-food-recognition
- **Number of Classes**: {num_classes}
- **Image Size**: 224x224
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Batch Size**: 32

## Files in Repository

- `resnet50_pytorch_model.bin` - ResNet-50 model weights
- `se_resnet50_pytorch_model.bin` - SE-ResNet-50 model weights
- `resnet50_metadata.json` - ResNet-50 metadata
- `se_resnet50_metadata.json` - SE-ResNet-50 metadata
- `README.md` - This file

## Citation

If you use these models, please cite:

```bibtex
@model{{raw_food_recognition_models_2024,
  title={{Raw Food Recognition Models: ResNet-50 and SE-ResNet-50}},
  author={{Ibrahim Daud}},
  year={{2024}},
  publisher={{HuggingFace}},
  url={{https://huggingface.co/{repo_id}}}
}}
```

## License

MIT License
"""
    
    return model_card


def main():
    parser = argparse.ArgumentParser(
        description='Upload PyTorch model checkpoint(s) to HuggingFace Hub'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to .pth checkpoint file (for single model upload)'
    )
    parser.add_argument(
        '--resnet_checkpoint',
        type=str,
        default=None,
        help='Path to ResNet-50 checkpoint (for uploading both models)'
    )
    parser.add_argument(
        '--se_resnet_checkpoint',
        type=str,
        default=None,
        help='Path to SE-ResNet-50 checkpoint (for uploading both models)'
    )
    parser.add_argument(
        '--repo_id',
        type=str,
        required=True,
        help='HuggingFace repository ID (e.g., "username/model-name")'
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help='Make the repository private'
    )
    parser.add_argument(
        '--token',
        type=str,
        default=None,
        help='HuggingFace API token (optional, will prompt if not provided)'
    )
    
    args = parser.parse_args()
    
    # Determine upload mode
    if args.resnet_checkpoint and args.se_resnet_checkpoint:
        # Upload both models to single repo
        if not os.path.exists(args.resnet_checkpoint):
            raise FileNotFoundError(f"ResNet-50 checkpoint not found: {args.resnet_checkpoint}")
        if not os.path.exists(args.se_resnet_checkpoint):
            raise FileNotFoundError(f"SE-ResNet-50 checkpoint not found: {args.se_resnet_checkpoint}")
        
        upload_both_models(
            resnet_checkpoint=args.resnet_checkpoint,
            se_resnet_checkpoint=args.se_resnet_checkpoint,
            repo_id=args.repo_id,
            private=args.private,
            token=args.token
        )
    elif args.checkpoint:
        # Upload single model
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        
        upload_model_to_hf(
            checkpoint_path=args.checkpoint,
            repo_id=args.repo_id,
            private=args.private,
            token=args.token
        )
    else:
        parser.error("Either provide --checkpoint (single model) or both --resnet_checkpoint and --se_resnet_checkpoint (both models)")


if __name__ == '__main__':
    main()

