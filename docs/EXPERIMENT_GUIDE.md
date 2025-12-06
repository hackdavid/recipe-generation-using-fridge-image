# Experiment Guide

This guide explains how to set up and run experiments with different configurations.

## Configuration Files

Experiments are configured using YAML files. There are two types:

1. **Base Configs** (`configs/`): Standard configurations for each model
2. **Experiment Configs** (`experiments/`): Specific experiment configurations

## Running Experiments

### Basic Training

```bash
# Train ResNet-50
python trainer/train.py configs/resnet50_config.yaml

# Train SE-ResNet-50
python trainer/train.py configs/se_resnet50_config.yaml
```

### Custom Experiments

```bash
# Run experiment 1
python trainer/train.py experiments/exp1.yaml

# Run experiment 2
python trainer/train.py experiments/exp2.yaml

# Run SE-ResNet experiment
python trainer/train.py experiments/se_resnet.yaml
```

### Debug Mode

Quick testing with minimal batches:

```bash
python trainer/train.py experiments/exp2_debug.yaml
```

Debug mode:
- Trains only 1 batch per epoch
- Validates only 1 batch per epoch
- Runs 1 epoch
- Useful for quick end-to-end testing

## Creating New Experiments

### Step 1: Copy Base Config

```bash
cp configs/resnet50_config.yaml experiments/my_experiment.yaml
```

### Step 2: Modify Parameters

Edit `experiments/my_experiment.yaml`:

```yaml
data:
  batch_size: 64  # Increase batch size
  image_size: 256  # Larger images

training:
  epochs: 100  # More epochs
  learning_rate: 0.0005  # Different learning rate
  optimizer: "SGD"  # Use SGD instead of Adam
  scheduler:
    type: "CosineAnnealingLR"  # Different scheduler

wandb:
  run_name: "my_experiment_v1"  # Unique run name
  tags: ["experiment", "large_batch"]
```

### Step 3: Run Experiment

```bash
python trainer/train.py experiments/my_experiment.yaml
```

## Experiment Tracking

### Weights & Biases (wandb)

Enable wandb in your config:

```yaml
wandb:
  use: true
  project: "ingredient-recognition"
  run_name: "experiment_name"
  tags: ["resnet50", "baseline"]
```

**Benefits:**
- Real-time metrics visualization
- Hyperparameter tracking
- Model checkpoint storage
- Experiment comparison

### Local Logging

All experiments generate:
- **Log files**: `logs/{model}_{timestamp}.log`
- **Checkpoints**: `checkpoints/{model}_best.pth` and `checkpoints/{model}_latest.pth`
- **Results**: `checkpoints/{model}_results.json`

## Hyperparameter Tuning

### Learning Rate

Common learning rates:
- **Adam**: 0.001 (default), 0.0005, 0.0001
- **SGD**: 0.01, 0.001, 0.0001

### Batch Size

Considerations:
- **GPU Memory**: Larger batch = more memory
- **Training Speed**: Larger batch = faster training
- **Gradient Quality**: Smaller batch = noisier gradients

Common sizes: 16, 32, 64, 128

### Image Size

Options:
- **224x224**: Standard, faster training
- **256x256**: Better detail, slower training
- **384x384**: High detail, requires more memory

### Optimizers

**Adam** (default):
- Good default choice
- Adaptive learning rates
- Works well for most cases

**SGD with Momentum**:
- Often better for fine-tuning
- Requires tuning learning rate
- Can achieve better final accuracy

### Learning Rate Schedulers

**StepLR** (default):
- Reduces LR at fixed intervals
- Simple and effective

**CosineAnnealingLR**:
- Smooth LR decay
- Good for longer training

**ReduceLROnPlateau**:
- Reduces LR when validation loss plateaus
- Adaptive scheduling

## Resuming Experiments

To resume from a checkpoint:

```yaml
checkpoint:
  resume: "./checkpoints/resnet50_latest.pth"
```

The training will:
- Load model weights
- Load optimizer state
- Resume from the saved epoch
- Continue training for remaining epochs

## Comparing Experiments

### Using Wandb

1. Open wandb dashboard
2. Compare runs side-by-side
3. Analyze metrics and hyperparameters

### Using Logs

Compare log files:
```bash
# View training history
cat logs/resnet50_*.log | grep "Epoch"
```

### Using Results JSON

```python
import json

# Load results
with open('checkpoints/resnet50_results.json') as f:
    results = json.load(f)

print(f"Final Accuracy: {results['final_accuracy']}")
print(f"F1-Score: {results['final_f1']}")
```

## Best Practices

1. **Use descriptive run names**: `"resnet50_baseline_v1"` not `"test"`
2. **Tag experiments**: Use tags to organize experiments
3. **Save configs**: Keep experiment configs in version control
4. **Document changes**: Note what changed between experiments
5. **Compare systematically**: Change one thing at a time
6. **Use debug mode**: Test configs quickly before full training

## Example Experiment Workflow

```bash
# 1. Generate class mapping (one-time)
python trainer/generate_class_mapping.py \
    --dataset_name "ibrahimdaud/raw-food-recognition" \
    --output trainer/class_mapping.json

# 2. Test config with debug mode
python trainer/train.py experiments/exp2_debug.yaml

# 3. Run full experiment
python trainer/train.py experiments/exp2.yaml

# 4. Monitor training (wandb or logs)
# Check wandb dashboard or tail logs/resnet50_*.log

# 5. Evaluate best model
python trainer/evaluate.py \
    --checkpoint ./checkpoints/resnet50_best.pth \
    --config experiments/exp2.yaml

# 6. Compare with other experiments
# Use wandb dashboard or compare results JSON files
```

## Troubleshooting

### Experiment fails to start

- Check config file syntax (YAML)
- Verify class mapping file exists
- Check dataset name/path

### Out of memory

- Reduce batch_size
- Reduce image_size
- Use gradient accumulation (future feature)

### Poor results

- Check learning rate (might be too high/low)
- Verify data augmentation
- Check class mapping
- Review training logs for issues

