"""
Validation Module

Provides functions for model validation during training.
"""

import torch
from tqdm import tqdm


def validate(model, val_loader, criterion, device, use_wandb=False):
    """
    Validate the model
    
    Args:
        model: PyTorch model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to run validation on
        use_wandb: Whether to log to wandb
        
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
        pbar = tqdm(val_loader, desc='Validation')
        for images, labels in pbar:
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
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels

