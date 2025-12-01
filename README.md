# Ingredient Recognition: ResNet-50 vs SE-ResNet-50

This project implements and compares two deep learning models for ingredient recognition from food images using HuggingFace datasets:
1. **ResNet-50**: Standard residual network architecture
2. **SE-ResNet-50**: ResNet-50 with Squeeze-and-Excitation attention mechanism

## Project Structure

```
.
├── models/
│   ├── __init__.py
│   ├── resnet50.py          # ResNet-50 implementation
│   └── se_resnet50.py        # SE-ResNet-50 implementation
├── trainer/
│   ├── __init__.py
│   ├── train.py              # Main training script
│   ├── config.py             # Configuration loader
│   ├── metrics.py            # Metrics calculation
│   ├── validation.py         # Validation functions
│   ├── hf_dataset.py         # HuggingFace dataset loader
│   └── evaluate.py           # Evaluation script
├── configs/
│   ├── resnet50_config.yaml  # ResNet-50 configuration
│   └── se_resnet50_config.yaml # SE-ResNet-50 configuration
├── train_colab.ipynb         # Google Colab training notebook
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd ai_coursework
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

This project uses **HuggingFace datasets** in streaming mode for memory efficiency. No local dataset download required!

Configure your dataset in the YAML config file:
```yaml
data:
  data_source: "huggingface"
  dataset_name: "SunnyAgarwal4274/Food_Ingredients"
  train_split: "train"
  val_split: "validation"
```

## Usage

### Local Training

Train a model using a YAML configuration file:

```bash
# Train ResNet-50
python trainer/train.py configs/resnet50_config.yaml

# Train SE-ResNet-50
python trainer/train.py configs/se_resnet50_config.yaml
```

### Google Colab Training

1. **Clone repository in Colab:**
```python
!git clone https://github.com/yourusername/your-repo.git
%cd your-repo
```

2. **Open `train_colab.ipynb`** and run all cells

3. **Enable GPU**: Runtime → Change runtime type → GPU

The notebook will automatically:
- Install dependencies
- Clone your repository
- Set up the environment
- Run training with GPU acceleration

### Configuration Files

All training parameters are specified in YAML config files. See `configs/` directory for examples:

**Key Configuration Sections:**
- `data`: Dataset source (HuggingFace), splits, image size
- `model`: Architecture (resnet50/se_resnet50), num_classes, pretrained
- `training`: Epochs, batch_size, learning_rate, optimizer, scheduler
- `wandb`: Experiment tracking (optional)
- `checkpoint`: Save directory, resume path

**Example Config:**
```yaml
data:
  data_source: "huggingface"
  dataset_name: "SunnyAgarwal4274/Food_Ingredients"
  train_split: "train"
  val_split: "validation"
  image_size: 224
  num_workers: 4

model:
  architecture: "resnet50"
  num_classes: 100
  pretrained: true

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  optimizer: "Adam"
  scheduler:
    type: "StepLR"
    step_size: 15
    gamma: 0.1

wandb:
  use: true
  project: "ingredient-recognition"
  api_key: "your-api-key-here"
```

### Evaluation

Evaluate a trained model:
```bash
python trainer/evaluate.py \
  --data_dir ./data \
  --checkpoint ./checkpoints/resnet50_best.pth \
  --plot_cm
```

## Model Architectures

### ResNet-50
- Standard residual network with bottleneck blocks
- 50 layers deep
- ImageNet pretrained weights
- ~25.6M parameters

### SE-ResNet-50
- ResNet-50 architecture with Squeeze-and-Excitation blocks
- SE blocks add channel attention mechanism
- ~26.0M parameters (slight increase)
- Improved feature representation through adaptive recalibration

## Key Features

- ✅ **HuggingFace Datasets**: Streaming mode for memory efficiency
- ✅ **Modular Structure**: Clean separation of concerns (trainer, models, configs)
- ✅ **YAML Configuration**: Easy experiment management
- ✅ **Transfer Learning**: Uses ImageNet pretrained weights
- ✅ **Data Augmentation**: Random crops, flips, color jitter
- ✅ **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-score
- ✅ **Visualization**: Confusion matrix plots
- ✅ **Checkpointing**: Save/restore training state
- ✅ **Experiment Tracking**: Integrated with Weights & Biases (wandb)
- ✅ **Flexible Optimizers**: Support for Adam and SGD
- ✅ **Learning Rate Scheduling**: StepLR, CosineAnnealingLR, ReduceLROnPlateau
- ✅ **Google Colab Support**: Ready-to-use notebook for GPU training

## Expected Outputs

Training produces:
- Model checkpoints (`checkpoints/`)
  - `{model}_best.pth` - Best validation accuracy model
  - `{model}_latest.pth` - Latest epoch checkpoint
- Training history (loss, accuracy) in JSON format
- Evaluation metrics (JSON format)
- Confusion matrices (if enabled)

## Google Colab Quick Start

1. Open `train_colab.ipynb` in Google Colab
2. Enable GPU: Runtime → Change runtime type → GPU
3. Run the first cell to clone repository:
```python
!git clone https://github.com/yourusername/your-repo.git
%cd your-repo
```
4. Run all cells sequentially
5. Training will start automatically with GPU acceleration

## Requirements

See `requirements.txt` for full list. Key dependencies:
- PyTorch (with CUDA support for GPU)
- torchvision
- datasets (HuggingFace)
- transformers
- wandb (optional, for experiment tracking)
- scikit-learn
- matplotlib, seaborn

## Citation

If you use this code, please cite:

**ResNet-50:**
```
He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. 
In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
```

**SE-ResNet-50:**
```
Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-excitation networks. 
In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7132-7141).
```
