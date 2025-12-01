"""
Evaluation Script for Ingredient Recognition Models
Calculates accuracy, precision, recall, F1-score, and confusion matrix
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import argparse
import os
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from models import create_resnet50, create_se_resnet50
from trainer.metrics import calculate_metrics


def get_test_loader(data_dir, batch_size=32, num_workers=4, image_size=224):
    """Create test data loader"""
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = ImageFolder(os.path.join(data_dir, 'test'), transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return test_loader, test_dataset.classes


def evaluate_model(model, test_loader, device, class_names):
    """Evaluate model and return predictions and labels"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def calculate_detailed_metrics(all_preds, all_labels, class_names):
    """Calculate comprehensive metrics"""
    # Overall metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Weighted averages
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # Macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Classification report
    report = classification_report(all_labels, all_preds, 
                                  target_names=class_names, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'support_per_class': support_per_class.tolist(),
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }


def plot_confusion_matrix(cm, class_names, save_path, top_n=20):
    """Plot confusion matrix (top N classes for readability)"""
    if len(class_names) > top_n:
        # Show top N most frequent classes
        class_counts = cm.sum(axis=1)
        top_indices = np.argsort(class_counts)[-top_n:]
        cm = cm[np.ix_(top_indices, top_indices)]
        class_names = [class_names[i] for i in top_indices]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved to {save_path}")


def plot_metrics_comparison(metrics_resnet, metrics_se, save_path):
    """Plot comparison between ResNet-50 and SE-ResNet-50"""
    metrics_names = ['Accuracy', 'Precision\n(Weighted)', 'Recall\n(Weighted)', 'F1-Score\n(Weighted)']
    resnet_values = [
        metrics_resnet['accuracy'],
        metrics_resnet['precision_weighted'],
        metrics_resnet['recall_weighted'],
        metrics_resnet['f1_weighted']
    ]
    se_values = [
        metrics_se['accuracy'],
        metrics_se['precision_weighted'],
        metrics_se['recall_weighted'],
        metrics_se['f1_weighted']
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, resnet_values, width, label='ResNet-50', alpha=0.8)
    bars2 = ax.bar(x + width/2, se_values, width, label='SE-ResNet-50', alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison: ResNet-50 vs SE-ResNet-50', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.set_ylim([0, 1])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Comparison plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Ingredient Recognition Model')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory (should contain test folder)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Directory to save evaluation results')
    parser.add_argument('--plot_cm', action='store_true',
                       help='Plot confusion matrix')
    parser.add_argument('--se_reduction', type=int, default=16,
                       help='SE block reduction ratio (for SE-ResNet-50)')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    num_classes = checkpoint['num_classes']
    class_names = checkpoint['class_names']
    model_type = checkpoint.get('model_type', 'resnet50')
    
    print(f"Model type: {model_type}")
    print(f"Number of classes: {num_classes}")
    
    # Create model
    if model_type == 'resnet50':
        model = create_resnet50(num_classes=num_classes, pretrained=False)
    else:
        model = create_se_resnet50(num_classes=num_classes, pretrained=False,
                                   reduction=args.se_reduction)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    
    # Load test data
    print("\nLoading test dataset...")
    test_loader, test_class_names = get_test_loader(
        args.data_dir, args.batch_size, args.num_workers, args.image_size
    )
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Evaluate
    print("\nEvaluating model...")
    all_preds, all_labels, all_probs = evaluate_model(model, test_loader, device, class_names)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_detailed_metrics(all_preds, all_labels, class_names)
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision:   {metrics['precision_weighted']:.4f} (weighted)")
    print(f"  Recall:      {metrics['recall_weighted']:.4f} (weighted)")
    print(f"  F1-Score:    {metrics['f1_weighted']:.4f} (weighted)")
    print(f"\nMacro Averages:")
    print(f"  Precision:   {metrics['precision_macro']:.4f}")
    print(f"  Recall:      {metrics['recall_macro']:.4f}")
    print(f"  F1-Score:    {metrics['f1_macro']:.4f}")
    
    print("\n" + "="*50)
    print("Classification Report")
    print("="*50)
    print(metrics['classification_report'])
    
    # Save results
    results = {
        'model_type': model_type,
        'checkpoint': args.checkpoint,
        'num_classes': num_classes,
        'test_samples': len(test_loader.dataset),
        'metrics': metrics
    }
    
    results_path = os.path.join(args.output_dir, f'{model_type}_evaluation.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {results_path}")
    
    # Plot confusion matrix
    if args.plot_cm:
        cm_path = os.path.join(args.output_dir, f'{model_type}_confusion_matrix.png')
        plot_confusion_matrix(np.array(metrics['confusion_matrix']), 
                            class_names, cm_path)
    
    print("\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()

