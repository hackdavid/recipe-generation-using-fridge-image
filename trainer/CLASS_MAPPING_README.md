# Class Mapping JSON Guide

This guide explains how to generate and use the class mapping JSON file for consistent label-to-ID mapping across training, validation, and inference.

## Overview

The class mapping JSON file stores the mapping between class names (strings) and their integer IDs. This ensures consistency across all stages of your workflow:
- **Training**: Uses the mapping to convert string labels to integers
- **Validation**: Uses the same mapping for consistent evaluation
- **Inference**: Uses the mapping to convert predictions back to class names

## Step 1: Generate Class Mapping JSON

**Before training**, generate the class mapping JSON file:

### For HuggingFace Datasets:

```bash
python trainer/generate_class_mapping.py \
    --dataset_name "ibrahimdaud/raw-food-recognition" \
    --output trainer/class_mapping.json
```

### For Local Folder Datasets:

```bash
python trainer/generate_class_mapping.py \
    --data_dir "./data" \
    --output trainer/class_mapping.json
```

### Options:

- `--dataset_name`: HuggingFace dataset identifier (e.g., "ibrahimdaud/raw-food-recognition")
- `--data_dir`: Local dataset directory path (for ImageFolder datasets)
- `--split`: Dataset split to use for HuggingFace datasets (default: "train")
- `--output`: Output path for JSON file (default: "trainer/class_mapping.json")

**Note**: You only need to run this once per dataset. The generated JSON file can be reused for all training runs.

## Step 2: Configure Training

Add the `class_mapping_path` to your experiment config file (e.g., `experiments/exp2.yaml`):

```yaml
data:
  data_source: "huggingface"
  dataset_name: "ibrahimdaud/raw-food-recognition"
  class_mapping_path: "trainer/class_mapping.json"  # Path to your generated mapping file
  # ... other data config
```

## Step 3: Training

The training script will automatically load and use the class mapping:

```bash
python trainer/train.py experiments/exp2.yaml
```

The training pipeline will:
1. Load the class mapping JSON file
2. Use it to convert string labels to integers during data loading
3. Return `class_names` for use in evaluation and logging

## JSON File Structure

The generated JSON file has the following structure:

```json
{
  "label_to_id": {
    "cauliflower": 0,
    "broccoli": 1,
    "carrot": 2,
    ...
  },
  "id_to_label": {
    "0": "cauliflower",
    "1": "broccoli",
    "2": "carrot",
    ...
  },
  "class_names": [
    "cauliflower",
    "broccoli",
    "carrot",
    ...
  ],
  "num_classes": 90,
  "dataset_name": "ibrahimdaud/raw-food-recognition",
  "source": "huggingface_classlabel",
  "metadata": {
    "description": "Class mapping for ingredient recognition model",
    "format": {
      "label_to_id": "Dictionary mapping class name (string) to integer ID",
      "id_to_label": "Dictionary mapping integer ID to class name (string)",
      "class_names": "List of class names in order (index corresponds to ID)"
    }
  }
}
```

## Using the Mapping in Code

### Load the Mapping:

```python
from trainer.hf_dataset import load_class_mapping

mapping = load_class_mapping('trainer/class_mapping.json')

# Access mappings
label_to_id = mapping['label_to_id']
id_to_label = mapping['id_to_label']
class_names = mapping['class_names']
num_classes = mapping['num_classes']
```

### Convert Predictions:

```python
# Convert prediction ID to class name
predicted_id = 5
predicted_class = id_to_label[str(predicted_id)]  # e.g., "cauliflower"

# Or use class_names list
predicted_class = class_names[predicted_id]  # Same result

# Convert class name to ID
class_name = "cauliflower"
class_id = label_to_id[class_name]  # e.g., 0
```

## Troubleshooting

### Error: "Class mapping file not found"

**Solution**: Generate the mapping file first using `generate_class_mapping.py`

### Error: "Label 'xyz' is a string but no label mapping is available"

**Solution**: Make sure you've generated the class mapping JSON and specified the correct path in your config file.

### Warning: "Class mapping file not found: trainer/class_mapping.json"

**Solution**: Either generate the file or update the `class_mapping_path` in your config to point to the correct location.

## Benefits

1. **Consistency**: Same mapping used across training, validation, and inference
2. **Reproducibility**: Fixed mapping ensures reproducible results
3. **Easy Inference**: Simple conversion between IDs and class names
4. **Version Control**: JSON file can be versioned with your code
5. **Pre-computed**: Generated once, used many times (faster training startup)

## Example Workflow

```bash
# 1. Generate mapping (one-time)
python trainer/generate_class_mapping.py \
    --dataset_name "ibrahimdaud/raw-food-recognition" \
    --output trainer/class_mapping.json

# 2. Train (uses the mapping automatically)
python trainer/train.py experiments/exp2.yaml

# 3. Use in inference/evaluation
python inference.py --checkpoint checkpoints/resnet50_best.pth \
    --class_mapping trainer/class_mapping.json
```

