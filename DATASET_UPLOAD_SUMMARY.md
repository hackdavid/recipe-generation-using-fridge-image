# Dataset Upload Summary

## Files Created

1. **`upload_dataset_to_hf.py`** - Main script to upload dataset to HuggingFace Hub
2. **`DATASET_README.md`** - README file for the HuggingFace dataset repository
3. **`UPLOAD_DATASET_GUIDE.md`** - Detailed guide on how to use the upload script

## Quick Start

### Step 1: Install Dependencies
```bash
pip install datasets pillow huggingface_hub pyarrow pandas scikit-learn tqdm
```

### Step 2: Login to HuggingFace
```bash
huggingface-cli login
```

### Step 3: Upload Dataset
```bash
python upload_dataset_to_hf.py --dataset_name "your-username/food-recognition-merged"
```

## What Gets Uploaded

- **Train Split**: 80% of images (~12,000 images)
- **Validation Split**: 20% of images (~3,000 images)
- **Format**: Parquet (automatically converted by HuggingFace)
- **Privacy**: Private repository (default)
- **README**: Dataset documentation with merging process details

## Dataset Features

- ✅ 90+ food categories
- ✅ Stratified train/validation split (maintains class distribution)
- ✅ Images stored in Parquet format for efficient access
- ✅ Includes label mappings (string labels and integer IDs)
- ✅ Comprehensive README with usage examples

## Source Datasets Documented

The README includes proper attribution to:
1. SunnyAgarwal4274/Food_and_Vegetables
2. Nattakarn/fruit-and-vegetable-image-recognition  
3. Kaggle Grocery Store Dataset

## Next Steps

1. Run the upload script with your HuggingFace username
2. Verify the dataset appears on HuggingFace Hub
3. Test loading the dataset using the examples in the README
4. Use the dataset for training your models!

## Notes

- The script automatically handles image loading, splitting, and format conversion
- Class mapping is saved locally as `class_mapping.json`
- The dataset will be private by default (use `--public` flag to make it public)
- All images are converted to RGB format automatically

