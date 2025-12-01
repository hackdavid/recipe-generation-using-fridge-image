"""
Metrics Calculation Module

Provides functions for calculating evaluation metrics.
"""

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


def calculate_metrics(all_preds, all_labels, num_classes):
    """
    Calculate precision, recall, and F1-score
    
    Args:
        all_preds: List/array of predicted labels
        all_labels: List/array of true labels
        num_classes: Number of classes
        
    Returns:
        Dictionary containing precision, recall, F1-score, and confusion matrix
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'confusion_matrix': cm.tolist()
    }

