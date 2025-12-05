# Guide: Uploading Merged Dataset to HuggingFace

This guide explains how to upload the merged food recognition dataset to HuggingFace Hub as a private repository with train/validation splits in Parquet format.

## Prerequisites

1. **Install required packages**:
   ```bash
   pip install datasets pillow huggingface_hub pyarrow pandas scikit-learn tqdm
   ```

2. **Login to HuggingFace**:
   ```bash
   huggingface-cli login
   ```
   Or use Python:
   ```python
   from huggingface_hub import login
   login()
   ```

## Usage

### Basic Usage

```bash
python upload_dataset_to_hf.py --dataset_name "your-username/your-dataset-name"
```

### Full Example

```bash
python upload_dataset_to_hf.py \
    --dataset_name "your-username/food-recognition-merged" \
    --data_dir "merged_dataset" \
    --train_ratio 0.8 \
    --seed 42 \
    --readme "DATASET_README.md"
```

### Arguments

- `--dataset_name` (required): HuggingFace dataset name in format `username/dataset-name`
- `--data_dir` (optional): Path to merged_dataset directory (default: `merged_dataset`)
- `--train_ratio` (optional): Ratio for training split (default: `0.8` for 80/20 split)
- `--seed` (optional): Random seed for reproducibility (default: `42`)
- `--private` (default): Make repository private (default: True)
- `--public` (optional): Make repository public (overrides --private)
- `--readme` (optional): Path to README.md file (default: `DATASET_README.md`)

## What the Script Does

1. **Scans the dataset directory**: Finds all image files organized by class folders
2. **Creates label mapping**: Maps class names to integer IDs
3. **Stratified splitting**: Splits data into train/validation (80/20) maintaining class distribution
4. **Loads images**: Converts images to PIL Image format
5. **Creates HuggingFace Dataset**: Builds DatasetDict with train/validation splits
6. **Uploads to Hub**: Pushes dataset to HuggingFace Hub in Parquet format
7. **Uploads README**: Includes dataset documentation

## Output

After successful upload, you'll get:
- Dataset available at: `https://huggingface.co/datasets/your-username/your-dataset-name`
- `class_mapping.json` file saved locally with label mappings

## Dataset Structure on HuggingFace

The uploaded dataset will have:
- `train/` split: 80% of images
- `validation/` split: 20% of images
- Each sample contains:
  - `image`: PIL Image object
  - `label`: String label (e.g., "apple", "banana")
  - `label_id`: Integer ID (0 to num_classes-1)

## Loading the Dataset

After upload, you can load the dataset:

```python
from datasets import load_dataset

dataset = load_dataset("your-username/your-dataset-name")
train = dataset['train']
val = dataset['validation']
```

## Troubleshooting

### Error: Not logged in
```bash
huggingface-cli login
```

### Error: Dataset directory not found
Make sure the `merged_dataset` folder exists and contains class subfolders with images.

### Memory Issues
For very large datasets, consider processing in batches or using streaming mode.

### Upload Fails
- Check your internet connection
- Verify HuggingFace token is valid
- Ensure you have write access to the repository

## Notes

- The dataset will be uploaded in **Parquet format** automatically by HuggingFace
- The repository will be **private** by default (use `--public` to make it public)
- Stratified splitting ensures class distribution is maintained in train/val splits
- Images are converted to RGB format automatically

