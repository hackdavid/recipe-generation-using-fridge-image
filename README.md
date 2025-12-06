# Ingredient Recognition: ResNet-50 vs SE-ResNet-50

A comprehensive deep learning project comparing ResNet-50 and SE-ResNet-50 architectures for ingredient recognition from food images. This project implements state-of-the-art computer vision models with proper training pipelines, evaluation metrics, and experiment tracking.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Model Architectures](#model-architectures)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Results](#results)
- [Documentation](#documentation)
- [Citation](#citation)
- [License](#license)

## 🎯 Overview

This project implements and compares two deep learning architectures for ingredient recognition:

1. **ResNet-50**: Standard residual network with 50 layers
2. **SE-ResNet-50**: ResNet-50 enhanced with Squeeze-and-Excitation attention mechanism

Both models are trained on food ingredient datasets using transfer learning from ImageNet pretrained weights, with comprehensive evaluation metrics and experiment tracking.

## ✨ Features

- ✅ **HuggingFace Datasets Integration**: Streaming mode for memory-efficient training
- ✅ **Modular Architecture**: Clean separation of models, training, and evaluation
- ✅ **YAML Configuration**: Easy experiment management and reproducibility
- ✅ **Transfer Learning**: ImageNet pretrained weights for faster convergence
- ✅ **Data Augmentation**: Random crops, flips, color jitter, and normalization
- ✅ **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-score, Confusion Matrix
- ✅ **Checkpointing**: Save/restore training state for resuming experiments
- ✅ **Experiment Tracking**: Integrated Weights & Biases (wandb) support
- ✅ **Flexible Training**: Support for Adam and SGD optimizers
- ✅ **Learning Rate Scheduling**: StepLR, CosineAnnealingLR, ReduceLROnPlateau
- ✅ **Debug Mode**: Quick end-to-end testing with minimal batches
- ✅ **Class Mapping**: Consistent label-to-ID mapping across training/inference
- ✅ **Google Colab Support**: Ready-to-use notebook for GPU training

## 📁 Project Structure

```
.
├── models/                      # Model architectures
│   ├── __init__.py
│   ├── resnet50.py             # ResNet-50 implementation
│   └── se_resnet50.py          # SE-ResNet-50 implementation
│
├── trainer/                     # Training pipeline
│   ├── __init__.py
│   ├── train.py                # Main training script
│   ├── config.py               # Configuration loader
│   ├── hf_dataset.py           # HuggingFace dataset loader
│   ├── validation.py           # Validation functions
│   ├── metrics.py              # Metrics calculation
│   ├── evaluate.py             # Evaluation script
│   └── generate_class_mapping.py  # Class mapping generator
│
├── utils/                       # Utility scripts
│   ├── __init__.py
│   ├── generate_multilabel_dataset.py  # Multi-label dataset generator
│   ├── upload_model_to_hf.py   # Upload models to HuggingFace Hub
│   └── benchmark_models.py     # Model benchmarking script
│
├── configs/                     # Configuration files
│   ├── resnet50_config.yaml    # ResNet-50 configuration
│   └── se_resnet50_config.yaml # SE-ResNet-50 configuration
│
├── experiments/                 # Experiment configurations
│   ├── exp1.yaml               # Experiment 1 config
│   ├── exp2.yaml               # Experiment 2 config
│   ├── exp2_debug.yaml         # Debug mode config
│   └── se_resnet.yaml          # SE-ResNet experiment
│
├── checkpoints/                 # Model checkpoints (gitignored)
├── logs/                        # Training logs (gitignored)
├── requirements.txt            # Python dependencies
├── train_colab.ipynb           # Google Colab training notebook
├── MULTI_LABEL_DATASET_GUIDE.md  # Multi-label dataset creation guide
└── README.md                   # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- Git

### Step 1: Clone Repository

```bash
git clone <your-repo-url>
cd ai_coursework
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Generate Class Mapping (Required)

Before training, generate the class mapping JSON file:

```bash
# For the merged raw food recognition dataset
python trainer/generate_class_mapping.py \
    --dataset_name "ibrahimdaud/raw-food-recognition" \
    --output trainer/class_mapping.json
```

**Note**: This project uses a custom merged dataset combining three publicly available datasets. See [DATASET_CREATION.md](DATASET_CREATION.md) for details on the dataset creation process.

## 🏃 Quick Start

### 1. Train ResNet-50

```bash
python trainer/train.py configs/resnet50_config.yaml
```

### 2. Train SE-ResNet-50

```bash
python trainer/train.py configs/se_resnet50_config.yaml
```

### 3. Debug Mode (Quick Test)

```bash
python trainer/train.py experiments/exp2_debug.yaml
```

## 📖 Usage

### Training

Train a model using a YAML configuration file:

```bash
python trainer/train.py <config_file.yaml>
```

**Example:**
```bash
python trainer/train.py experiments/exp2.yaml
```

### Resuming Training

To resume from a checkpoint, set the `resume` path in your config:

```yaml
checkpoint:
  resume: "./checkpoints/resnet50_latest.pth"
```

### Evaluation

Evaluate a trained model:

```bash
python trainer/evaluate.py \
    --checkpoint ./checkpoints/resnet50_best.pth \
    --config configs/resnet50_config.yaml \
    --plot_cm
```

### Benchmarking Models

Benchmark both models for comprehensive performance analysis:

```bash
python utils/benchmark_models.py \
    --resnet_checkpoint ./checkpoints/resnet50_best.pth \
    --se_resnet_checkpoint ./checkpoints/se_resnet50_best.pth \
    --output_dir ./benchmark_results \
    --num_iterations 100 \
    --batch_sizes 1 4 8 16 32
```

This measures:
- Inference speed (latency, throughput, FPS)
- Memory usage (GPU/CPU)
- Model size and parameters
- Accuracy comparison
- Performance trade-offs

Results are saved in JSON format for easy integration into reports.

### Benchmarking on Standard Datasets

Compare your models against standard food classification benchmarks by evaluating on overlapping classes:

**Food-101 Benchmark:**
```bash
python utils/benchmark_food101.py \
    --resnet_checkpoint ./checkpoints/resnet50_best.pth \
    --se_resnet_checkpoint ./checkpoints/se_resnet50_best.pth \
    --class_mapping trainer/class_mapping.json \
    --output_dir ./benchmark_results/food101
```

**FoodX-251 Benchmark:**
```bash
python utils/benchmark_foodx251.py \
    --resnet_checkpoint ./checkpoints/resnet50_best.pth \
    --se_resnet_checkpoint ./checkpoints/se_resnet50_best.pth \
    --class_mapping trainer/class_mapping.json \
    --dataset_path ./data/foodx251 \
    --output_dir ./benchmark_results/foodx251
```

**UECFOOD-256 Benchmark:**
```bash
python utils/benchmark_uecfood256.py \
    --resnet_checkpoint ./checkpoints/resnet50_best.pth \
    --se_resnet_checkpoint ./checkpoints/se_resnet50_best.pth \
    --class_mapping trainer/class_mapping.json \
    --dataset_path ./data/uecfood256 \
    --output_dir ./benchmark_results/uecfood256
```

Each script:
- Automatically matches class names between datasets
- Filters to only overlapping classes
- Evaluates both ResNet-50 and SE-ResNet-50
- Generates JSON results and Markdown reports
- Shows class mapping and performance comparison

**Note**: Food-101 is available on HuggingFace and will be downloaded automatically. FoodX-251 and UECFOOD-256 may need to be downloaded manually.

### Upload Models to HuggingFace Hub

Upload trained models for sharing and inference:

**Upload both models to a single repository:**
```bash
python utils/upload_model_to_hf.py \
    --resnet_checkpoint ./checkpoints/resnet50_best.pth \
    --se_resnet_checkpoint ./checkpoints/se_resnet50_best.pth \
    --repo_id "your-username/raw-food-recognition-models" \
    --private
```

**Upload a single model:**
```bash
python utils/upload_model_to_hf.py \
    --checkpoint ./checkpoints/resnet50_best.pth \
    --repo_id "your-username/resnet50-raw-food-recognition"
```

### Google Colab

1. Open `train_colab.ipynb` in Google Colab
2. Enable GPU: Runtime → Change runtime type → GPU
3. Run all cells sequentially
4. Training will start automatically with GPU acceleration

## 🏗️ Model Architectures

### ResNet-50

- **Architecture**: Standard residual network with bottleneck blocks
- **Depth**: 50 layers
- **Parameters**: ~25.6M
- **Pretrained**: ImageNet weights
- **Key Features**: Skip connections, batch normalization, ReLU activation

### SE-ResNet-50

- **Architecture**: ResNet-50 with Squeeze-and-Excitation blocks
- **Depth**: 50 layers + SE attention
- **Parameters**: ~26.0M (slight increase)
- **Pretrained**: ImageNet weights (excluding SE blocks)
- **Key Features**: Channel attention mechanism, adaptive feature recalibration

**SE Block Mechanism:**
1. **Squeeze**: Global average pooling to capture global context
2. **Excitation**: Two FC layers with ReLU and Sigmoid to generate channel weights
3. **Scale**: Element-wise multiplication to recalibrate features

## ⚙️ Configuration

All training parameters are specified in YAML config files. See `configs/` and `experiments/` directories for examples.

### Configuration Structure

```yaml
# Dataset Configuration
data:
  data_source: "huggingface"  # or "folder"
  dataset_name: "ibrahimdaud/raw-food-recognition"
  train_split: "train"
  val_split: "validation"
  image_size: 224
  num_workers: 4
  class_mapping_path: "trainer/class_mapping.json"
  debug_mode: false  # Enable for quick testing
  debug_max_train_batches: 1
  debug_max_val_batches: 1

# Model Configuration
model:
  architecture: "resnet50"  # or "se_resnet50"
  num_classes: 90
  pretrained: true
  se_reduction: 16  # Only for SE-ResNet-50

# Training Configuration
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: "Adam"  # or "SGD"
  scheduler:
    type: "StepLR"  # or "CosineAnnealingLR", "ReduceLROnPlateau"
    step_size: 15
    gamma: 0.1

# Checkpoint Configuration
checkpoint:
  save_dir: "./checkpoints"
  resume: false  # Path to checkpoint or false
  save_best: true
  save_latest: true

# Weights & Biases Configuration
wandb:
  use: true
  project: "ingredient-recognition"
  entity: null
  run_name: "experiment_name"
  tags: ["resnet50", "baseline"]
  api_key: "your-api-key"  # Optional, can use wandb login
```

### Key Configuration Options

- **`data_source`**: `"huggingface"` for HuggingFace datasets, `"folder"` for local ImageFolder
- **`debug_mode`**: Enable to run minimal batches for quick testing
- **`se_reduction`**: Reduction ratio for SE blocks (default: 16, must be > 0)
- **`optimizer`**: `"Adam"` or `"SGD"` with momentum
- **`scheduler`**: Learning rate scheduling strategy

## 📊 Evaluation

The training pipeline automatically evaluates models and generates:

- **Metrics**: Accuracy, Precision, Recall, F1-score
- **Confusion Matrix**: Visual representation of predictions
- **Training History**: Loss and accuracy curves
- **Checkpoints**: Best and latest model states

### Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

## 📈 Results

Training produces:

- **Model Checkpoints**: `checkpoints/{model}_best.pth` and `checkpoints/{model}_latest.pth`
- **Results JSON**: `checkpoints/{model}_results.json` with metrics and history
- **Training Logs**: `logs/{model}_{timestamp}.log`
- **Wandb Dashboard**: Experiment tracking and visualization (if enabled)

## 📚 Documentation

- **[Dataset Creation](DATASET_CREATION.md)**: Comprehensive documentation on the merged raw food recognition dataset creation process, including methodology, source datasets, and publication details
- **[Multi-Label Dataset Guide](MULTI_LABEL_DATASET_GUIDE.md)**: Comprehensive guide for creating multi-label food recognition datasets using synthetic image composition. Includes approach explanation, usage commands, and technical details.
- **[Multi-Label Model Architecture](MULTI_LABEL_MODEL_ARCHITECTURE.md)**: Detailed explanation of the multi-label classification architecture, why it works, and the three training strategies (freeze encoder, full training, fine-tuning) with guidance on when to use each approach.

## 🔬 Technical Details

### Data Loading

- **Streaming Mode**: HuggingFace datasets loaded in streaming mode for memory efficiency
- **Data Augmentation**: Random resized crops, horizontal flips, color jitter, normalization
- **Class Mapping**: Pre-generated JSON mapping ensures consistency across training/inference

### Training Pipeline

- **Loss Function**: CrossEntropyLoss
- **Optimizers**: Adam (default) or SGD with momentum
- **Schedulers**: StepLR, CosineAnnealingLR, or ReduceLROnPlateau
- **Checkpointing**: Automatic saving of best and latest models
- **Resume Support**: Resume training from any checkpoint

### Evaluation Pipeline

- **Metrics Calculation**: Comprehensive classification metrics
- **Confusion Matrix**: Visual representation of model performance
- **Class-wise Analysis**: Per-class precision, recall, and F1-score

## 📝 Citation

If you use this code in your research, please cite:

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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Your Name**
- University: University of Roehampton
- Course: AI Coursework
- Email: your.email@example.com

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- HuggingFace for dataset infrastructure
- Weights & Biases for experiment tracking
- Original ResNet and SE-Net paper authors

---

**Note**: This project is part of academic coursework. Please ensure proper attribution and citation when using this code.
