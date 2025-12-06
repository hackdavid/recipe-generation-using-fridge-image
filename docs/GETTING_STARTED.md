# Getting Started Guide

This guide will help you get started with the Ingredient Recognition project quickly.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- Git
- 10GB+ free disk space

## Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd ai_coursework
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Generate Class Mapping

Before training, you need to generate the class mapping JSON file:

```bash
# For HuggingFace datasets
python trainer/generate_class_mapping.py \
    --dataset_name "ibrahimdaud/raw-food-recognition" \
    --output trainer/class_mapping.json

# For local folder datasets
python trainer/generate_class_mapping.py \
    --data_dir "./data" \
    --output trainer/class_mapping.json
```

**Note**: This step is required and only needs to be done once per dataset.

### 5. Configure Your Experiment

Edit a configuration file in `configs/` or `experiments/`:

```yaml
data:
  data_source: "huggingface"
  dataset_name: "ibrahimdaud/raw-food-recognition"
  class_mapping_path: "trainer/class_mapping.json"
  # ... other settings

model:
  architecture: "resnet50"
  num_classes: 90
  pretrained: true

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  # ... other settings
```

### 6. Run Training

```bash
# Train ResNet-50
python trainer/train.py configs/resnet50_config.yaml

# Train SE-ResNet-50
python trainer/train.py configs/se_resnet50_config.yaml

# Quick debug test
python trainer/train.py experiments/exp2_debug.yaml
```

### 7. Monitor Training

- **Console Output**: Real-time training progress
- **Log Files**: Saved in `logs/` directory
- **Wandb Dashboard**: If enabled, visit wandb.ai to view metrics

### 8. Evaluate Model

After training, evaluate your model:

```bash
python trainer/evaluate.py \
    --checkpoint ./checkpoints/resnet50_best.pth \
    --config configs/resnet50_config.yaml \
    --plot_cm
```

## Quick Test (Debug Mode)

To quickly verify everything works:

```bash
# 1. Generate class mapping
python trainer/generate_class_mapping.py \
    --dataset_name "ibrahimdaud/raw-food-recognition" \
    --output trainer/class_mapping.json

# 2. Run debug mode (trains 1 batch, validates 1 batch)
python trainer/train.py experiments/exp2_debug.yaml
```

This will:
- Load the dataset
- Train for 1 batch
- Validate for 1 batch
- Save a checkpoint
- Complete in ~2-5 minutes

## Common Issues

### Issue: "Class mapping file not found"

**Solution**: Generate the class mapping first:
```bash
python trainer/generate_class_mapping.py \
    --dataset_name "your-dataset-name" \
    --output trainer/class_mapping.json
```

### Issue: CUDA out of memory

**Solutions**:
- Reduce `batch_size` in config
- Reduce `image_size` in config
- Use CPU mode (set `device: "cpu"`)

### Issue: Dataset download fails

**Solutions**:
- Check internet connection
- Verify dataset name is correct
- Try using local folder mode instead

### Issue: Wandb authentication error

**Solutions**:
- Run `wandb login` and enter your API key
- Or set `wandb.use: false` in config to disable wandb

## Next Steps

- Read the [README.md](../README.md) for detailed documentation
- Check [CLASS_MAPPING_GUIDE.md](CLASS_MAPPING_GUIDE.md) for class mapping details
- Review [DATASET_GUIDE.md](DATASET_GUIDE.md) for dataset setup
- Explore experiment configs in `experiments/` directory

## Getting Help

- Check the documentation in `docs/` directory
- Review example configs in `configs/` and `experiments/`
- Check log files in `logs/` for error messages

