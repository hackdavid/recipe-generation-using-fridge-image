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
from datetime import datetime

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")

from models import create_resnet50, create_se_resnet50
from trainer.config import load_config
from trainer.metrics import calculate_metrics
from trainer.validation import validate
from trainer.hf_dataset import get_hf_data_loaders


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, use_wandb=False):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
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
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc




def main():
    # Get config file path from command line
    if len(sys.argv) < 2:
        print("Usage: python train.py <config_file.yaml>")
        print("\nExample:")
        print("  python train.py configs/resnet50_config.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    # Load configuration
    print(f"Loading configuration from: {config_path}")
    cfg = load_config(config_path)
    
    # Print configuration summary
    print("\n" + "="*50)
    print("Configuration Summary")
    print("="*50)
    print(f"Config file: {config_path}")
    print(f"Model: {cfg['model']}")
    print(f"Data directory: {cfg['data_dir']}")
    print(f"Epochs: {cfg['epochs']}")
    print(f"Batch size: {cfg['batch_size']}")
    print(f"Learning rate: {cfg['lr']}")
    print(f"Optimizer: {cfg['optimizer']}")
    print(f"Scheduler: {cfg['scheduler'].get('type', 'StepLR')}")
    print(f"Wandb: {'Enabled' if cfg['use_wandb'] else 'Disabled'}")
    print("="*50)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb
    use_wandb = cfg.get('use_wandb', False) and WANDB_AVAILABLE
    if use_wandb:
        # Set wandb API key from config if provided
        wandb_api_key = cfg.get('wandb_api_key')
        if wandb_api_key:
            os.environ['WANDB_API_KEY'] = wandb_api_key
            print("✓ Wandb API key set from config")
        elif not os.environ.get('WANDB_API_KEY'):
            print("Warning: No wandb API key found in config or environment. Wandb login may be required.")
        # Generate run name if not provided
        run_name = cfg.get('wandb_run_name')
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
        print(f"✓ Wandb initialized: {wandb.run.url}")
    elif cfg.get('use_wandb', False) and not WANDB_AVAILABLE:
        print("Warning: wandb enabled in config but wandb not installed. Continuing without wandb.")
    
    # Create save directory
    os.makedirs(cfg['save_dir'], exist_ok=True)
    
    # Load data
    print("\nLoading datasets...")
    if cfg['data_source'] == 'huggingface':
        print(f"Using HuggingFace dataset: {cfg['dataset_name']}")
        train_loader, val_loader, num_classes, class_names = get_hf_data_loaders(
            dataset_name=cfg['dataset_name'],
            num_classes=cfg['num_classes'],
            train_split=cfg['train_split'],
            val_split=cfg['val_split'],
            batch_size=cfg['batch_size'],
            num_workers=cfg['num_workers'],
            image_size=cfg['image_size']
        )
        # Override num_classes from config (since we're using config value)
        num_classes = cfg['num_classes']
    else:
        raise ValueError(f"Unsupported data_source: {cfg['data_source']}. Only 'huggingface' is supported.")
    
    print(f"Number of classes: {num_classes}")
    # Note: IterableDataset doesn't have len(), so we can't print sample counts
    if hasattr(train_loader.dataset, '__len__') and train_loader.dataset.__len__() is not None:
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
    else:
        print("Training/validation samples: Streaming dataset (size unknown)")
    
    # Create model
    print(f"\nCreating {cfg['model']} model...")
    if cfg['model'] == 'resnet50':
        model = create_resnet50(num_classes=num_classes, pretrained=cfg.get('pretrained', True))
    else:
        model = create_se_resnet50(num_classes=num_classes, pretrained=cfg.get('pretrained', True), 
                                   reduction=cfg.get('se_reduction', 16))
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
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
        print(f"\nResuming from checkpoint: {cfg['resume']}")
        checkpoint = torch.load(cfg['resume'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint['best_val_acc']
        history = checkpoint['history']
    
    # Training loop
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50)
    
    for epoch in range(start_epoch, cfg['epochs']):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                           optimizer, device, epoch, use_wandb=use_wandb)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, 
                                                           criterion, device, use_wandb=use_wandb)
        
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
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{cfg['epochs']}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        
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
            print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")
            
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
    print("\n" + "="*50)
    print("Final Evaluation")
    print("="*50)
    
    final_val_loss, final_val_acc, final_preds, final_labels = validate(
        model, val_loader, criterion, device
    )
    
    metrics = calculate_metrics(final_preds, final_labels, num_classes)
    
    print(f"\nFinal Validation Results:")
    print(f"Accuracy: {final_val_acc:.2f}%")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    
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
            if len(class_names) > 20:
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
            print(f"Warning: Could not log confusion matrix to wandb: {e}")
    
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
    if use_wandb:
        wandb.save(results_path)
        wandb.finish()
    
    print(f"\n✓ Training complete!")
    print(f"✓ Results saved to {cfg['save_dir']}")
    if use_wandb:
        print(f"✓ Wandb run: {wandb.run.url}")


if __name__ == '__main__':
    main()

