"""
Multi-Label Classification Training Pipeline

Supports three training modes:
1. Freeze encoder: Only train classifier head
2. Full training: Train encoder + classifier
3. Fine-tuning: Train with different learning rates
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
import argparse
import yaml
from pathlib import Path
from datetime import datetime
import wandb
from tqdm import tqdm
import json
import sys
import os

# Add project root to Python path to allow imports from any directory
# Get the directory containing this file (trainer/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root (parent of trainer/)
project_root = os.path.dirname(script_dir)
# Add project root to sys.path if not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.multilabel_resnet50 import create_multilabel_resnet50
from trainer.multilabel_dataset import get_multilabel_data_loaders
from trainer.multilabel_metrics import compute_multilabel_metrics, print_metrics


class MultiLabelTrainer:
    """Multi-label classification trainer"""
    
    def __init__(self, config: dict):
        """
        Initialize trainer
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Training mode
        self.training_mode = config.get('training_mode', 'full')  # freeze, full, finetune
        
        # Create output directories
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb
        if config.get('use_wandb', True):
            wandb_config = config.get('wandb', {})
            run_name = wandb_config.get('run_name') or config.get('experiment_name') or f"multilabel_{self.training_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            wandb.init(
                project=wandb_config.get('project', config.get('wandb_project', 'multilabel-food-recognition')),
                name=run_name,
                config=config
            )
        
        # Load data
        print("\n" + "="*70)
        print("Loading Dataset")
        print("="*70)
        self.train_loader, self.val_loader, self.num_classes = get_multilabel_data_loaders(
            dataset_name=config['data']['dataset_name'],
            split_ratio=config['data'].get('split_ratio', 0.8),
            batch_size=config['data']['batch_size'],
            num_workers=config['data'].get('num_workers', 4),
            image_size=config['data'].get('image_size', 224),
            seed=config.get('seed', 42),
            debug_mode=config.get('debug_mode', False),
            debug_max_samples=config.get('debug_max_samples', 10)
        )
        
        # Create model
        print("\n" + "="*70)
        print("Creating Model")
        print("="*70)
        freeze_encoder = (self.training_mode == 'freeze')
        
        # Get checkpoint path if specified
        checkpoint_path = config['model'].get('checkpoint_path', None)
        pretrained = config['model'].get('pretrained', True) if checkpoint_path is None else False
        
        self.model = create_multilabel_resnet50(
            num_classes=self.num_classes,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
            dropout=config['model'].get('dropout', 0.5),
            freeze_encoder=freeze_encoder
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer
        self._setup_optimizer()
        
        # Learning rate scheduler
        self._setup_scheduler()
        
        # Training state
        self.start_epoch = 0
        self.best_map = 0.0
        self.train_history = []
        
        # Load checkpoint if resuming (read from config)
        resume_checkpoint = config.get('resume_checkpoint')
        if resume_checkpoint:
            self.load_checkpoint(resume_checkpoint)
    
    def _setup_optimizer(self):
        """Setup optimizer based on training mode"""
        if self.training_mode == 'freeze':
            # Only optimize classifier
            self.optimizer = optim.Adam(
                self.model.classifier.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training'].get('weight_decay', 1e-4)
            )
            print("✓ Optimizer: Adam (classifier only)")
        
        elif self.training_mode == 'finetune':
            # Different learning rates for encoder and classifier
            encoder_lr = self.config['training'].get('encoder_lr', self.config['training']['learning_rate'] * 0.1)
            classifier_lr = self.config['training']['learning_rate']
            
            self.optimizer = optim.Adam([
                {'params': self.model.encoder.parameters(), 'lr': encoder_lr},
                {'params': self.model.classifier.parameters(), 'lr': classifier_lr}
            ], weight_decay=self.config['training'].get('weight_decay', 1e-4))
            print(f"✓ Optimizer: Adam (encoder lr={encoder_lr:.2e}, classifier lr={classifier_lr:.2e})")
        
        else:  # full training
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training'].get('weight_decay', 1e-4)
            )
            print("✓ Optimizer: Adam (all parameters)")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        scheduler_type = self.config['training'].get('scheduler', 'step')
        
        if scheduler_type == 'step':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=self.config['training'].get('step_size', 10),
                gamma=self.config['training'].get('gamma', 0.1)
            )
        elif scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs']
            )
        elif scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            self.scheduler = None
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['training']['epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Accumulate metrics
            running_loss += loss.item()
            probs = torch.sigmoid(logits)
            all_preds.append(probs.detach())
            all_targets.append(labels.detach())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log to wandb
            if self.config.get('use_wandb', True) and batch_idx % 10 == 0:
                wandb.log({
                    'train/batch_loss': loss.item(),
                    'train/learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        # Compute epoch metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = compute_multilabel_metrics(all_preds, all_targets)
        
        avg_loss = running_loss / len(self.train_loader)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def validate(self, epoch: int = None):
        """Validate model"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validating")
            for batch in pbar:
                images = batch['image'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                
                # Accumulate
                running_loss += loss.item()
                probs = torch.sigmoid(logits)
                all_preds.append(probs)
                all_targets.append(labels)
        
        # Compute metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = compute_multilabel_metrics(all_preds, all_targets)
        
        avg_loss = running_loss / len(self.val_loader)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save checkpoint (only best and latest)"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_map': self.best_map,
            'metrics': metrics,
            'config': self.config,
            'num_classes': self.num_classes,
            'training_mode': self.training_mode,
            'model_type': 'multilabel_resnet50'
        }
        
        # Save latest checkpoint (always overwrites previous latest)
        checkpoint_path = self.checkpoint_dir / 'multilabel_latest.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Saved latest checkpoint (epoch {epoch+1}, mAP: {metrics['map']:.4f})")
        
        # Save best checkpoint (only if this is the best so far)
        if is_best:
            best_path = self.checkpoint_dir / 'multilabel_best.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best checkpoint (epoch {epoch+1}, mAP: {metrics['map']:.4f})")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        print(f"\nLoading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_map = checkpoint.get('best_map', 0.0)
        
        print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']+1}")
        print(f"  Best mAP: {self.best_map:.4f}")
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*70)
        print("Starting Training")
        print("="*70)
        print(f"Training mode: {self.training_mode}")
        print(f"Epochs: {self.config['training']['epochs']}")
        print(f"Starting from epoch: {self.start_epoch}")
        
        for epoch in range(self.start_epoch, self.config['training']['epochs']):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{self.config['training']['epochs']}")
            print(f"{'='*70}")
            
            # Train
            train_metrics = self.train_epoch(epoch)
            print_metrics(train_metrics, prefix="Train - ")
            
            # Validate
            val_metrics = self.validate(epoch)
            print_metrics(val_metrics, prefix="Val - ")
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['map'])
                else:
                    self.scheduler.step()
            
            # Check if best
            is_best = val_metrics['map'] > self.best_map
            if is_best:
                self.best_map = val_metrics['map']
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best=is_best)
            
            # Log to wandb
            if self.config.get('use_wandb', True):
                log_dict = {
                    'epoch': epoch + 1,
                    'train/loss': train_metrics['loss'],
                    'train/f1_macro': train_metrics['f1_macro'],
                    'train/map': train_metrics['map'],
                    'val/loss': val_metrics['loss'],
                    'val/f1_macro': val_metrics['f1_macro'],
                    'val/map': val_metrics['map'],
                    'val/subset_accuracy': val_metrics['subset_accuracy'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }
                wandb.log(log_dict)
            
            # Save history
            self.train_history.append({
                'epoch': epoch + 1,
                'train': train_metrics,
                'val': val_metrics
            })
        
        print("\n" + "="*70)
        print("Training Complete!")
        print("="*70)
        print(f"Best mAP: {self.best_map:.4f}")
        
        # Save training history
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        print(f"✓ Saved training history to {history_path}")
        
        if self.config.get('use_wandb', True):
            wandb.finish()


def load_config(config_path: str) -> dict:
    """Load YAML config file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train multi-label classification model')
    parser.add_argument('config', type=str, help='Path to config YAML file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (overrides config)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override debug mode if provided via command line
    if args.debug:
        config['debug_mode'] = True
        config['debug_max_samples'] = 10
        if 'data' not in config:
            config['data'] = {}
        config['data']['batch_size'] = 2
        config['data']['num_workers'] = 0
        print("⚠ DEBUG MODE ENABLED (command line override)")
    
    # Create trainer
    trainer = MultiLabelTrainer(config)
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()

