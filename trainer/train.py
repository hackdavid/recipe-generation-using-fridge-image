"""
Training Script for Ingredient Recognition Models
Supports both ResNet-50 and SE-ResNet-50
Integrated with Weights & Biases (wandb) for experiment tracking

Usage:
    python trainer/train.py configs/resnet50_config.yaml
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from tqdm import tqdm
import json
import logging
from datetime import datetime
from itertools import islice

# Add project root to Python path to allow imports from any directory
# Get the directory containing this file (trainer/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root (parent of trainer/)
project_root = os.path.dirname(script_dir)
# Add project root to sys.path if not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    # Print warning before logger is set up
    print("Warning: wandb not installed. Install with: pip install wandb")

from models import create_resnet50, create_se_resnet50
from trainer.config import load_config
from trainer.metrics import calculate_metrics
from trainer.validation import validate
from trainer.hf_dataset import get_hf_data_loaders


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, use_wandb=False, max_batches=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Limit batches in debug mode
    loader_iter = iter(train_loader)
    if max_batches is not None:
        loader_iter = islice(loader_iter, max_batches)
        total_batches = max_batches
    else:
        total_batches = len(train_loader) if hasattr(train_loader, '__len__') else None
    
    pbar = tqdm(loader_iter, desc=f'Epoch {epoch} [Train]', total=total_batches)
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Log batch metrics to wandb
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'train/batch_loss': loss.item(),
                'train/batch_acc': 100 * (predicted == labels).sum().item() / labels.size(0),
                'train/epoch': epoch,
                'train/batch': batch_idx
            })
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss / (pbar.n + 1):.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    # Calculate average loss and accuracy
    num_batches = batch_idx + 1
    epoch_loss = running_loss / num_batches
    epoch_acc = 100 * correct / total if total > 0 else 0.0
    
    return epoch_loss, epoch_acc




def setup_logger(cfg):
    """Setup logger with file and console handlers"""
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(project_root, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create timestamped log filename (human-readable format)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = cfg.get('model', 'model')
    log_filename = f"{model_name}_{timestamp}.log"
    log_filepath = os.path.join(logs_dir, log_filename)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # File handler - writes to timestamped log file
    file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler - writes to stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging to file: {log_filepath}")
    return logger, log_filepath


def main():
    # Get config file path from command line
    if len(sys.argv) < 2:
        print("Usage: python train.py <config_file.yaml>")
        print("\nExample:")
        print("  python train.py configs/resnet50_config.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    # Load configuration first (needed for logger setup)
    cfg = load_config(config_path)
    
    # Setup logger with timestamped log file
    logger, log_filepath = setup_logger(cfg)
    
    logger.info(f"Loading configuration from: {config_path}")
    
    # Log configuration summary
    logger.info("\n" + "="*50)
    logger.info("Configuration Summary")
    logger.info("="*50)
    logger.info(f"Config file: {config_path}")
    logger.info(f"Model: {cfg['model']}")
    logger.info(f"Data directory: {cfg.get('data_dir', 'N/A')}")
    logger.info(f"Dataset name: {cfg.get('dataset_name', 'N/A')}")
    logger.info(f"Epochs: {cfg['epochs']}")
    logger.info(f"Batch size: {cfg['batch_size']}")
    logger.info(f"Learning rate: {cfg['lr']}")
    logger.info(f"Optimizer: {cfg['optimizer']}")
    logger.info(f"Scheduler: {cfg['scheduler'].get('type', 'StepLR')}")
    logger.info(f"Wandb: {'Enabled' if cfg['use_wandb'] else 'Disabled'}")
    logger.info("="*50)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize wandb
    use_wandb = cfg.get('use_wandb', False) and WANDB_AVAILABLE
    if use_wandb:
        # Set wandb API key from config if provided
        wandb_api_key = cfg.get('wandb_api_key')
        if wandb_api_key:
            os.environ['WANDB_API_KEY'] = wandb_api_key
            logger.info("✓ Wandb API key set from config")
        elif not os.environ.get('WANDB_API_KEY'):
            logger.warning("No wandb API key found in config or environment. Wandb login may be required.")
        # Generate run name if not provided
        run_name = cfg.get('wandb_run_name')
        if run_name is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_name = f"{cfg['model']}_{timestamp}"
        
        wandb.init(
            project=cfg.get('wandb_project', 'ingredient-recognition'),
            entity=cfg.get('wandb_entity'),
            name=run_name,
            tags=cfg.get('wandb_tags', []),
            config={
                'model': cfg['model'],
                'epochs': cfg['epochs'],
                'batch_size': cfg['batch_size'],
                'learning_rate': cfg['lr'],
                'weight_decay': cfg['weight_decay'],
                'image_size': cfg['image_size'],
                'num_workers': cfg['num_workers'],
                'se_reduction': cfg.get('se_reduction') if cfg['model'] == 'se_resnet50' else None,
                'device': str(device),
            }
        )
        logger.info(f"✓ Wandb initialized: {wandb.run.url}")
    elif cfg.get('use_wandb', False) and not WANDB_AVAILABLE:
        logger.warning("wandb enabled in config but wandb not installed. Continuing without wandb.")
    
    # Create save directory
    os.makedirs(cfg['save_dir'], exist_ok=True)
    
    # Debug mode settings - check early and adjust num_workers
    debug_mode = cfg.get('debug_mode', False)
    if debug_mode:
        logger.info("\n" + "="*50)
        logger.info("DEBUG MODE ENABLED")
        logger.info("="*50)
        logger.info("Setting num_workers=0 for debug mode (avoids multiprocessing issues)")
        logger.info("="*50 + "\n")
        # Override num_workers in debug mode BEFORE creating data loaders
        cfg['num_workers'] = 0
    
    # Load data
    logger.info("\nLoading datasets...")
    
    # Get class mapping path (default: trainer/class_mapping.json)
    class_mapping_path = cfg.get('class_mapping_path', 'trainer/class_mapping.json')
    if class_mapping_path and os.path.exists(class_mapping_path):
        logger.info(f"Using class mapping from: {class_mapping_path}")
    elif class_mapping_path:
        logger.warning(f"Class mapping file not found: {class_mapping_path}")
        logger.warning("Training will proceed without mapping (may fail if labels are strings)")
        class_mapping_path = None
    
    train_loader, val_loader, num_classes, class_names = get_hf_data_loaders(
        data_source=cfg['data_source'],
        data_dir=cfg.get('data_dir') if cfg['data_source'] == 'folder' else cfg.get('dataset_name'),
        num_classes=cfg['num_classes'],
        train_split=cfg.get('train_split'),
        val_split=cfg.get('val_split'),
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        image_size=cfg['image_size'],
        shuffle_train=True,
        seed=42,
        train_ratio=cfg.get('train_ratio', 0.8),
        class_mapping_path=class_mapping_path
    )
    # Ensure num_classes is from config
    num_classes = cfg['num_classes']
    
    logger.info(f"Number of classes: {num_classes}")
    # Note: IterableDataset doesn't have len(), so we can't log sample counts
    if hasattr(train_loader.dataset, '__len__') and train_loader.dataset.__len__() is not None:
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
    else:
        logger.info("Training/validation samples: Streaming dataset (size unknown)")
    
    # Create model
    logger.info(f"\nCreating {cfg['model']} model...")
    if cfg['model'] == 'resnet50':
        model = create_resnet50(num_classes=num_classes, pretrained=cfg.get('pretrained', True))
    else:
        # Handle se_reduction: if False/None/0, use default 16; otherwise use the provided value
        se_reduction = cfg.get('se_reduction', 16)
        if se_reduction is False or se_reduction is None or se_reduction == 0:
            se_reduction = 16
            logger.warning(f"se_reduction was set to {cfg.get('se_reduction')} (invalid). Using default value: 16")
        model = create_se_resnet50(num_classes=num_classes, pretrained=cfg.get('pretrained', True), 
                                   reduction=se_reduction)
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Log model info to wandb
    if use_wandb:
        wandb_config_update = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_classes': num_classes,
        }
        # Only add sample counts if dataset has length
        if hasattr(train_loader.dataset, '__len__') and train_loader.dataset.__len__() is not None:
            wandb_config_update['train_samples'] = len(train_loader.dataset)
            wandb_config_update['val_samples'] = len(val_loader.dataset)
        wandb.config.update(wandb_config_update)
        # Log model architecture
        wandb.watch(model, log='all', log_freq=100)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Create optimizer based on config
    optimizer_type = cfg.get('optimizer', 'Adam').lower()
    if optimizer_type == 'sgd':
        sgd_cfg = cfg.get('sgd', {})
        optimizer = optim.SGD(
            model.parameters(), 
            lr=cfg['lr'], 
            weight_decay=cfg['weight_decay'],
            momentum=sgd_cfg.get('momentum', 0.9),
            nesterov=sgd_cfg.get('nesterov', False)
        )
    else:  # Default to Adam
        optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    
    # Create scheduler based on config
    scheduler_cfg = cfg.get('scheduler', {})
    scheduler_type = scheduler_cfg.get('type', 'StepLR').lower()
    if scheduler_type == 'cosineannealinglr':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg['epochs']
        )
    elif scheduler_type == 'reducelronplateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=scheduler_cfg.get('gamma', 0.1), patience=5
        )
    else:  # Default to StepLR
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=scheduler_cfg.get('step_size', 15), 
            gamma=scheduler_cfg.get('gamma', 0.1)
        )
    
    # Store scheduler type for later use
    cfg['scheduler_type'] = scheduler_type
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    if cfg.get('resume'):
        logger.info(f"\nResuming from checkpoint: {cfg['resume']}")
        checkpoint = torch.load(cfg['resume'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        checkpoint_epoch = checkpoint['epoch']
        best_val_acc = checkpoint['best_val_acc']
        history = checkpoint['history']
        
        # When resuming, start from the checkpoint epoch (which is the next epoch to train)
        # Note: checkpoint['epoch'] is the epoch number that was completed (e.g., epoch 1 means epoch 0 was done)
        # So we start training from checkpoint_epoch (which is the next epoch)
        start_epoch = checkpoint_epoch
        
        logger.info(f"Checkpoint was saved at epoch {checkpoint_epoch} (completed {checkpoint_epoch} epochs)")
        logger.info(f"Resuming training from epoch {start_epoch} to {cfg['epochs']}")
        
        # If checkpoint epoch >= total epochs, extend epochs to allow continuing training
        # This ensures that when resuming, we can always train at least one more epoch
        if start_epoch >= cfg['epochs']:
            logger.info(f"Checkpoint epoch ({start_epoch}) >= total epochs ({cfg['epochs']})")
            logger.info("Extending epochs to allow continuing training from checkpoint")
            # Extend epochs to allow at least one more epoch
            cfg['epochs'] = start_epoch + 1
    
    # Training loop
    logger.info("\n" + "="*50)
    logger.info("Starting Training")
    logger.info("="*50)
    
    # Debug mode batch limits (num_workers already set above)
    debug_max_train_batches = cfg.get('debug_max_train_batches', 1)
    debug_max_val_batches = cfg.get('debug_max_val_batches', 1)
    
    if debug_mode:
        logger.info(f"Debug batch limits: train={debug_max_train_batches}, val={debug_max_val_batches}")
    
    # Log training range
    if start_epoch < cfg['epochs']:
        epochs_to_train = cfg['epochs'] - start_epoch
        logger.info(f"Training epochs: {start_epoch} to {cfg['epochs']-1} (total: {epochs_to_train} epoch(s))")
        if debug_mode:
            logger.info(f"Each epoch will process: {debug_max_train_batches} train batch(es) + {debug_max_val_batches} val batch(es)")
    else:
        logger.warning(f"start_epoch ({start_epoch}) >= total epochs ({cfg['epochs']}). No training will occur.")
    
    for epoch in range(start_epoch, cfg['epochs']):
        # Train
        max_train_batches = debug_max_train_batches if debug_mode else None
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, 
            optimizer, device, epoch, 
            use_wandb=use_wandb,
            max_batches=max_train_batches
        )
        
        # Validate
        max_val_batches = debug_max_val_batches if debug_mode else None
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, 
            criterion, device, 
            use_wandb=use_wandb,
            max_batches=max_val_batches
        )
        
        # Update learning rate
        if scheduler_type == 'reducelronplateau':
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
        else:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Log epoch metrics to wandb
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/epoch_loss': train_loss,
                'train/epoch_acc': train_acc,
                'val/epoch_loss': val_loss,
                'val/epoch_acc': val_acc,
                'learning_rate': current_lr,
            })
        
        # Log epoch summary
        logger.info(f"\nEpoch {epoch+1}/{cfg['epochs']}")
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        logger.info(f"Learning Rate: {current_lr:.6f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'history': history,
            'num_classes': num_classes,
            'class_names': class_names,  # May be None for HF datasets
            'model_type': cfg['model']
        }
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint['best_val_acc'] = best_val_acc
            best_model_path = os.path.join(cfg['save_dir'], f"{cfg['model']}_best.pth")
            torch.save(checkpoint, best_model_path)
            logger.info(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")
            
            # Log best model to wandb
            if use_wandb:
                wandb.run.summary['best_val_acc'] = best_val_acc
                wandb.run.summary['best_epoch'] = epoch + 1
                wandb.save(best_model_path)
        
        # Save latest checkpoint
        latest_model_path = os.path.join(cfg['save_dir'], f"{cfg['model']}_latest.pth")
        torch.save(checkpoint, latest_model_path)
        
        # Log checkpoint to wandb
        if use_wandb:
            wandb.save(latest_model_path)
    
    # Final evaluation with detailed metrics
    logger.info("\n" + "="*50)
    logger.info("Final Evaluation")
    logger.info("="*50)
    
    # Final evaluation (also respect debug mode)
    max_final_val_batches = debug_max_val_batches if debug_mode else None
    final_val_loss, final_val_acc, final_preds, final_labels = validate(
        model, val_loader, criterion, device, max_batches=max_final_val_batches
    )
    
    metrics = calculate_metrics(final_preds, final_labels, num_classes)
    
    logger.info(f"\nFinal Validation Results:")
    logger.info(f"Accuracy: {final_val_acc:.2f}%")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1-Score: {metrics['f1']:.4f}")
    
    # Log final metrics to wandb
    if use_wandb:
        wandb.run.summary.update({
            'final_accuracy': final_val_acc,
            'final_precision': metrics['precision'],
            'final_recall': metrics['recall'],
            'final_f1': metrics['f1'],
        })
        
        # Log confusion matrix
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
            
            cm = np.array(metrics['confusion_matrix'])
            # Plot top 20 classes for readability
            if class_names and len(class_names) > 20:
                class_counts = cm.sum(axis=1)
                top_indices = np.argsort(class_counts)[-20:]
                cm_plot = cm[np.ix_(top_indices, top_indices)]
                class_names_plot = [class_names[i] for i in top_indices]
            else:
                cm_plot = cm
                class_names_plot = class_names if class_names else [f'Class {i}' for i in range(num_classes)]
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm_plot, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names_plot, yticklabels=class_names_plot)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            wandb.log({'confusion_matrix': wandb.Image(plt)})
            plt.close()
        except Exception as e:
            logger.warning(f"Could not log confusion matrix to wandb: {e}")
    
    # Save final metrics
    results = {
        'model': cfg['model'],
        'num_classes': num_classes,
        'final_accuracy': final_val_acc,
        'final_precision': metrics['precision'],
        'final_recall': metrics['recall'],
        'final_f1': metrics['f1'],
        'history': history,
        'timestamp': datetime.now().isoformat(),
        'config': cfg  # Save config used for this run
    }
    
    results_path = os.path.join(cfg['save_dir'], f"{cfg['model']}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Log results file to wandb
    wandb_url = None
    if use_wandb:
        wandb.save(results_path)
        # Store URL before finishing
        if wandb.run is not None:
            wandb_url = wandb.run.url
        wandb.finish()
    
    logger.info(f"\n✓ Training complete!")
    logger.info(f"✓ Results saved to {cfg['save_dir']}")
    logger.info(f"✓ Log file saved to: {log_filepath}")
    if use_wandb and wandb_url:
        logger.info(f"✓ Wandb run: {wandb_url}")


if __name__ == '__main__':
    """
    How to run this training script:
    
    From the project root directory:
    
    1. Using local folder dataset:
       python trainer/train.py configs/resnet50_config.yaml
       
    2. Using HuggingFace dataset:
       python trainer/train.py configs/resnet50_config.yaml
       
    3. Using experiment configs:
       python trainer/train.py experiments/exp1.yaml
       python trainer/train.py experiments/exp2.yaml
    
    Examples:
        # Train ResNet-50 with local folder
        python trainer/train.py configs/resnet50_config.yaml
        
        # Train SE-ResNet-50 with HuggingFace dataset
        python trainer/train.py configs/se_resnet50_config.yaml
        
        # Run experiment with custom config
        python trainer/train.py experiments/exp1.yaml
    
    Note: Make sure your YAML config file specifies:
    - data_source: 'folder' or 'huggingface'
    - data_dir: Local path (for folder) or HuggingFace dataset name
    - num_classes: Number of classes (required)
    - All other training parameters
    """
    main()

