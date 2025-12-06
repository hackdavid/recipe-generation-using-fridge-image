"""
Multi-Label Classification Metrics

Computes metrics specific to multi-label classification:
- Hamming Loss
- Subset Accuracy (Exact Match Ratio)
- F1-Score (micro, macro, per-class)
- Precision/Recall (micro, macro, per-class)
- Mean Average Precision (mAP)
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import (
    hamming_loss,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score
)


def compute_multilabel_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute comprehensive multi-label classification metrics
    
    Args:
        predictions: Predicted probabilities (B, C)
        targets: Ground truth labels (B, C)
        threshold: Threshold for binary predictions
    
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy
    pred_probs = predictions.cpu().numpy()
    target_labels = targets.cpu().numpy()
    
    # Binary predictions
    pred_binary = (pred_probs >= threshold).astype(int)
    
    # Flatten for some metrics
    pred_flat = pred_binary.flatten()
    target_flat = target_labels.flatten()
    
    metrics = {}
    
    # Hamming Loss (lower is better)
    metrics['hamming_loss'] = float(hamming_loss(target_labels, pred_binary))
    
    # Subset Accuracy / Exact Match Ratio
    metrics['subset_accuracy'] = float(accuracy_score(target_labels, pred_binary))
    
    # F1 Scores
    metrics['f1_micro'] = float(f1_score(target_labels, pred_binary, average='micro', zero_division=0))
    metrics['f1_macro'] = float(f1_score(target_labels, pred_binary, average='macro', zero_division=0))
    metrics['f1_weighted'] = float(f1_score(target_labels, pred_binary, average='weighted', zero_division=0))
    
    # Per-class F1
    f1_per_class = f1_score(target_labels, pred_binary, average=None, zero_division=0)
    metrics['f1_per_class'] = f1_per_class.tolist()
    metrics['f1_mean'] = float(np.nanmean(f1_per_class))
    
    # Precision
    metrics['precision_micro'] = float(precision_score(target_labels, pred_binary, average='micro', zero_division=0))
    metrics['precision_macro'] = float(precision_score(target_labels, pred_binary, average='macro', zero_division=0))
    metrics['precision_weighted'] = float(precision_score(target_labels, pred_binary, average='weighted', zero_division=0))
    
    # Recall
    metrics['recall_micro'] = float(recall_score(target_labels, pred_binary, average='micro', zero_division=0))
    metrics['recall_macro'] = float(recall_score(target_labels, pred_binary, average='macro', zero_division=0))
    metrics['recall_weighted'] = float(recall_score(target_labels, pred_binary, average='weighted', zero_division=0))
    
    # Mean Average Precision (mAP)
    try:
        # Compute AP for each class, then average
        ap_per_class = []
        for c in range(target_labels.shape[1]):
            if target_labels[:, c].sum() > 0:  # Only compute if class exists
                ap = average_precision_score(target_labels[:, c], pred_probs[:, c])
                ap_per_class.append(ap)
        metrics['map'] = float(np.mean(ap_per_class)) if ap_per_class else 0.0
    except Exception:
        metrics['map'] = 0.0
    
    # Label-wise accuracy
    label_accuracy = []
    for c in range(target_labels.shape[1]):
        if target_labels[:, c].sum() > 0:
            acc = accuracy_score(target_labels[:, c], pred_binary[:, c])
            label_accuracy.append(acc)
    metrics['label_accuracy_mean'] = float(np.mean(label_accuracy)) if label_accuracy else 0.0
    
    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """Print metrics in a formatted way"""
    print(f"\n{prefix}Multi-Label Classification Metrics:")
    print(f"  Hamming Loss: {metrics['hamming_loss']:.4f} (lower is better)")
    print(f"  Subset Accuracy: {metrics['subset_accuracy']:.4f}")
    print(f"  F1-Score (Micro): {metrics['f1_micro']:.4f}")
    print(f"  F1-Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"  F1-Score (Mean): {metrics['f1_mean']:.4f}")
    print(f"  Precision (Macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (Macro): {metrics['recall_macro']:.4f}")
    print(f"  mAP: {metrics['map']:.4f}")

