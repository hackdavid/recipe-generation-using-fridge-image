"""
Validation Module

Provides functions for model validation during training.
"""

import torch
from tqdm import tqdm
from itertools import islice


def validate(model, val_loader, criterion, device, use_wandb=False, max_batches=None):
    """
    Validate the model
    
    Args:
        model: PyTorch model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to run validation on
        use_wandb: Whether to log to wandb
        max_batches: Maximum number of batches to process (for debug mode)
        
    Returns:
        epoch_loss, epoch_acc, all_preds, all_labels
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        # Limit batches in debug mode
        loader_iter = iter(val_loader)
        if max_batches is not None:
            loader_iter = islice(loader_iter, max_batches)
            total_batches = max_batches
        else:
            total_batches = len(val_loader) if hasattr(val_loader, '__len__') else None
        
        pbar = tqdm(loader_iter, desc='Validation', total=total_batches)
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{running_loss / (pbar.n + 1):.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    # Calculate average loss and accuracy
    num_batches = batch_idx + 1
    epoch_loss = running_loss / num_batches if num_batches > 0 else 0.0
    epoch_acc = 100 * correct / total if total > 0 else 0.0
    
    return epoch_loss, epoch_acc, all_preds, all_labels

