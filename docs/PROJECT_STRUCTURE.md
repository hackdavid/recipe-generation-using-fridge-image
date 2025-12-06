# Project Structure Documentation

This document provides a detailed overview of the project structure and the purpose of each component.

## Directory Structure

```
ai_coursework/
│
├── models/                          # Model architecture implementations
│   ├── __init__.py                 # Package initialization
│   ├── resnet50.py                 # ResNet-50 model implementation
│   └── se_resnet50.py              # SE-ResNet-50 model implementation
│
├── trainer/                         # Training pipeline and utilities
│   ├── __init__.py                 # Package initialization
│   ├── train.py                    # Main training script
│   ├── config.py                   # YAML configuration loader
│   ├── hf_dataset.py               # HuggingFace dataset loader
│   ├── validation.py               # Validation functions
│   ├── metrics.py                  # Metrics calculation (accuracy, F1, etc.)
│   ├── evaluate.py                 # Model evaluation script
│   ├── generate_class_mapping.py   # Class mapping generator utility
│   └── class_mapping.json           # Generated class mapping (version controlled)
│
├── configs/                         # Base configuration files
│   ├── resnet50_config.yaml        # ResNet-50 training configuration
│   └── se_resnet50_config.yaml     # SE-ResNet-50 training configuration
│
├── experiments/                     # Experiment-specific configurations
│   ├── exp1.yaml                   # Experiment 1 configuration
│   ├── exp2.yaml                   # Experiment 2 configuration
│   ├── exp2_debug.yaml             # Debug mode configuration
│   └── se_resnet.yaml              # SE-ResNet experiment configuration
│
├── docs/                            # Documentation files
│   ├── GETTING_STARTED.md          # Quick start guide
│   ├── CLASS_MAPPING_GUIDE.md      # Class mapping documentation
│   ├── DATASET_GUIDE.md            # Dataset setup guide
│   ├── MEMORY_EFFICIENT_GUIDE.md   # Memory optimization guide
│   └── PROJECT_STRUCTURE.md        # This file
│
├── checkpoints/                     # Model checkpoints (gitignored)
│   └── .gitkeep                    # Keep directory in git
│
├── logs/                            # Training logs (gitignored)
│
├── wandb/                           # Weights & Biases runs (gitignored)
│
├── requirements.txt                 # Python dependencies
├── LICENSE                          # MIT License
├── README.md                        # Main project documentation
├── .gitignore                      # Git ignore rules
└── train_colab.ipynb               # Google Colab training notebook
```

## Component Descriptions

### Models (`models/`)

Contains the neural network architecture implementations:

- **`resnet50.py`**: Standard ResNet-50 implementation with ImageNet pretrained weights
- **`se_resnet50.py`**: SE-ResNet-50 implementation with Squeeze-and-Excitation blocks

Both models follow a consistent interface:
- `create_resnet50(num_classes, pretrained=True)` factory function
- `create_se_resnet50(num_classes, pretrained=True, reduction=16)` factory function

### Trainer (`trainer/`)

Core training and evaluation pipeline:

- **`train.py`**: Main training script that orchestrates the entire training process
  - Loads configuration
  - Sets up data loaders
  - Initializes model, optimizer, scheduler
  - Runs training loop with validation
  - Saves checkpoints and logs metrics

- **`config.py`**: Configuration loader that parses YAML files and flattens nested structures

- **`hf_dataset.py`**: HuggingFace dataset integration
  - Streaming dataset support for memory efficiency
  - Class mapping integration
  - Data augmentation pipeline

- **`validation.py`**: Validation functions used during training

- **`metrics.py`**: Comprehensive metrics calculation
  - Accuracy, Precision, Recall, F1-score
  - Confusion matrix generation

- **`evaluate.py`**: Standalone evaluation script for trained models

- **`generate_class_mapping.py`**: Utility to generate class mapping JSON files

### Configurations (`configs/` and `experiments/`)

YAML configuration files define all training parameters:

- **`configs/`**: Base configurations for each model architecture
- **`experiments/`**: Experiment-specific configurations for different runs

Configuration structure:
```yaml
data:          # Dataset configuration
model:         # Model architecture settings
training:      # Training hyperparameters
checkpoint:    # Checkpoint settings
wandb:         # Experiment tracking
logging:       # Logging configuration
```

### Documentation (`docs/`)

Comprehensive documentation:

- **`GETTING_STARTED.md`**: Quick start guide for new users
- **`CLASS_MAPPING_GUIDE.md`**: Detailed guide on class mapping
- **`DATASET_GUIDE.md`**: Dataset setup and upload instructions
- **`MEMORY_EFFICIENT_GUIDE.md`**: Memory optimization strategies
- **`PROJECT_STRUCTURE.md`**: This file

### Generated/Output Directories

These directories are gitignored but kept in structure:

- **`checkpoints/`**: Saved model checkpoints (`.pth` files)
- **`logs/`**: Training log files (`.log` files)
- **`wandb/`**: Weights & Biases experiment tracking data

## File Naming Conventions

- **Python files**: `snake_case.py`
- **Config files**: `snake_case.yaml`
- **Documentation**: `UPPER_SNAKE_CASE.md`
- **Checkpoints**: `{model}_{type}.pth` (e.g., `resnet50_best.pth`)
- **Logs**: `{model}_{timestamp}.log`

## Code Organization Principles

1. **Separation of Concerns**: Models, training, and evaluation are separate modules
2. **Configuration-Driven**: All parameters in YAML files for reproducibility
3. **Modular Design**: Each component can be used independently
4. **Documentation**: Comprehensive docstrings and comments
5. **Error Handling**: Robust error handling and validation

## Adding New Components

### Adding a New Model

1. Create `models/new_model.py`
2. Implement model class with `create_new_model()` factory function
3. Add config file in `configs/new_model_config.yaml`
4. Update `train.py` to handle new model type

### Adding a New Experiment

1. Copy existing experiment config from `experiments/`
2. Modify parameters as needed
3. Run: `python trainer/train.py experiments/new_exp.yaml`

### Adding New Metrics

1. Add metric calculation function to `trainer/metrics.py`
2. Call function in `trainer/train.py` or `trainer/evaluate.py`
3. Update documentation if needed

## Dependencies

See `requirements.txt` for full dependency list. Key dependencies:

- **PyTorch**: Deep learning framework
- **torchvision**: Computer vision utilities
- **datasets**: HuggingFace dataset library
- **wandb**: Experiment tracking (optional)
- **scikit-learn**: Metrics calculation
- **matplotlib/seaborn**: Visualization

## Version Control

- **Tracked**: Source code, configs, documentation, requirements
- **Ignored**: Checkpoints, logs, wandb runs, virtual environments, data

Use `.gitignore` to manage what gets committed to version control.

