# Multi-Label Dataset Creation Guide

This document explains the approach and methodology for creating a multi-label food recognition dataset from single-class images.

## Overview

The multi-label dataset generation process creates composite images containing 2-5 different food items per image, enabling multi-label classification training. This approach allows us to leverage existing single-class datasets to create a comprehensive multi-label dataset.

## Approach

### 1. Dataset Generation Strategy

We use **synthetic image composition** to create multi-label images by combining multiple single-class food images into realistic composite images. This approach offers several advantages:

- **Scalability**: Can generate large datasets from existing single-class collections
- **Controlled Distribution**: Can balance class representation and label counts
- **Realistic Composition**: Uses natural arrangement methods for realistic appearance
- **Flexibility**: Supports different composition strategies (grid, overlay, natural)

### 2. Composition Methods

The generator supports three composition strategies:

#### **Grid Layout** (`grid`)
- Arranges images in a structured grid pattern
- Best for: Clear separation of items, easy annotation verification
- Use case: When you need distinct, non-overlapping items

#### **Overlay Layout** (`overlay`)
- Overlays images with blending and transparency
- Best for: More realistic appearance with overlapping items
- Use case: When you want items to appear naturally together

#### **Natural Layout** (`natural`) - **Recommended**
- Creates realistic "tabletop" arrangements with proper positioning
- Features:
  - Natural background (gradient, texture)
  - Realistic shadows and depth
  - Proper scaling and positioning
  - Natural transformations (rotation, brightness)
- Best for: Most realistic appearance, production use

### 3. Class Imbalance Handling

The generator addresses class imbalance through:

- **Inverse Frequency Weighting**: Rare classes are more likely to be sampled
- **Oversampling**: Ensures all classes appear in the dataset
- **Controlled Distribution**: Maintains target label count distribution:
  - 2 labels: 40%
  - 3 labels: 35%
  - 4 labels: 20%
  - 5 labels: 5%

### 4. Image Processing Pipeline

Each composite image goes through:

1. **Image Selection**: Randomly samples one image per selected class
2. **Preprocessing**: Resizes images to fit canvas while maintaining aspect ratio
3. **Composition**: Applies selected composition method
4. **Enhancement**: Adds realistic transformations (rotation, brightness, shadows)
5. **Annotation**: Generates multi-hot encoded labels

## Dataset Structure

The generated dataset follows this structure:

```
multilabel_dataset/
├── images/
│   ├── composite_000000.jpg
│   ├── composite_000001.jpg
│   └── ... (all images in single folder)
├── annotations/
│   └── labels.json  (single annotation file)
└── metadata/
    ├── dataset_info.json
    └── statistics.json
```

### Annotation Format

Each entry in `annotations/labels.json`:

```json
{
  "composite_000000.jpg": {
    "labels": [1, 5, 16, 29],
    "label_names": ["apple", "banana", "carrot", "grapes"],
    "num_labels": 4
  }
}
```

## Usage

### Prerequisites

Install required dependencies:

```bash
pip install pillow numpy matplotlib seaborn
```

### Basic Usage

Generate a multi-label dataset:

```bash
python utils/generate_multilabel_dataset.py \
    --source_dir ./merged_dataset \
    --output_dir ./multilabel_dataset \
    --num_images 13000 \
    --composition_method natural
```

### Parameters

- `--source_dir`: Path to directory containing single-class folders (required)
- `--output_dir`: Output directory for multi-label dataset (default: `./multilabel_dataset`)
- `--num_images`: Total number of images to generate (default: `13000`)
- `--min_labels`: Minimum labels per image (default: `2`)
- `--max_labels`: Maximum labels per image (default: `5`)
- `--composition_method`: Composition method - `grid`, `overlay`, or `natural` (default: `natural`)
- `--canvas_size`: Canvas size as width height (default: `512 512`)
- `--no_oversample`: Disable oversampling of rare classes

### Advanced Examples

**High-resolution dataset:**
```bash
python utils/generate_multilabel_dataset.py \
    --source_dir ./merged_dataset \
    --output_dir ./multilabel_dataset_hq \
    --num_images 15000 \
    --composition_method natural \
    --canvas_size 768 768
```

**Quick test dataset:**
```bash
python utils/generate_multilabel_dataset.py \
    --source_dir ./merged_dataset \
    --output_dir ./multilabel_dataset_test \
    --num_images 100 \
    --composition_method natural
```

**Uniform class distribution (no oversampling):**
```bash
python utils/generate_multilabel_dataset.py \
    --source_dir ./merged_dataset \
    --output_dir ./multilabel_dataset \
    --num_images 13000 \
    --no_oversample
```

## Workflow

### Step 1: Prepare Source Dataset

Ensure your source directory has this structure:

```
merged_dataset/
├── apple/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── banana/
│   └── ...
└── ... (one folder per class)
```

### Step 2: Generate Dataset

Run the generator:

```bash
python utils/generate_multilabel_dataset.py \
    --source_dir ./merged_dataset \
    --output_dir ./multilabel_dataset \
    --num_images 13000 \
    --composition_method natural
```

### Step 3: Verify Output

Check the generated files:

- `images/`: Contains all composite images
- `annotations/labels.json`: Contains all annotations
- `metadata/dataset_info.json`: Dataset metadata
- `metadata/statistics.json`: Dataset statistics

### Step 4: Upload to HuggingFace (Optional)

If you want to upload to HuggingFace:

```bash
python utils/upload_multilabel_dataset_to_hf.py \
    --dataset_dir ./multilabel_dataset \
    --repo_id your-username/multi-label-food-recognition \
    --split_ratio 0.8
```

## Technical Details

### Image Composition Algorithm

1. **Natural Layout Algorithm**:
   - Creates gradient background (light to dark)
   - Adds subtle texture for realism
   - Sorts images by size (larger first)
   - Places images with preferred positioning (lower 2/3 of canvas)
   - Applies realistic transformations
   - Adds shadows for depth perception

2. **Overlap Detection**:
   - Uses rectangle overlap detection
   - Attempts to find non-overlapping positions
   - Falls back to overlapping with transparency if needed

3. **Transformation Pipeline**:
   - Random rotation (-10° to +10°)
   - Brightness adjustment (±5%)
   - Contrast adjustment (±5%)
   - Optional blur for depth effect

### Statistics Generation

The generator automatically tracks:

- **Label Distribution**: Frequency of each class
- **Label Count Distribution**: Distribution of labels per image
- **Co-occurrence**: Which classes appear together
- **Top Patterns**: Most common class combinations

## Output Files

### `annotations/labels.json`
Single JSON file containing all image annotations with:
- Image filename as key
- Labels (integer IDs)
- Label names (strings)
- Number of labels

### `metadata/dataset_info.json`
Dataset metadata including:
- Number of classes
- Class names and mappings
- Generation parameters
- Generation date

### `metadata/statistics.json`
Comprehensive statistics:
- Total images
- Label distribution
- Label count distribution
- Co-occurrence patterns
- Top co-occurrences

## Best Practices

1. **Use Natural Composition**: The `natural` method produces the most realistic results
2. **Adequate Dataset Size**: Generate at least 10,000 images for training
3. **Verify Statistics**: Check `statistics.json` to ensure balanced distribution
4. **Test First**: Generate a small test dataset (100 images) before full generation
5. **Monitor Class Balance**: Review label distribution to ensure all classes are represented

## Troubleshooting

### Memory Issues
- Reduce batch size if processing large datasets
- Process in smaller chunks if needed

### Class Imbalance
- Enable oversampling (default)
- Check statistics.json for distribution
- Adjust source dataset if needed

### Image Quality
- Ensure source images are high quality
- Use appropriate canvas size (512x512 recommended)
- Check composition method (natural recommended)

## Citation

If you use this dataset generation approach, please cite:

```bibtex
@software{multi_label_food_dataset_generator,
  title={Multi-Label Food Recognition Dataset Generator},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

## License

MIT License - See LICENSE file for details.

