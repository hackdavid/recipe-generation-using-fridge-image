"""
Comparison Script: ResNet-50 vs SE-ResNet-50
Loads both models and compares their performance side-by-side
"""

import torch
import argparse
import json
import os
from evaluate import evaluate_model, calculate_detailed_metrics, get_test_loader, plot_confusion_matrix, plot_metrics_comparison
from models import create_resnet50, create_se_resnet50


def main():
    parser = argparse.ArgumentParser(description='Compare ResNet-50 and SE-ResNet-50')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--resnet_checkpoint', type=str, required=True,
                       help='Path to ResNet-50 checkpoint')
    parser.add_argument('--se_resnet_checkpoint', type=str, required=True,
                       help='Path to SE-ResNet-50 checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--output_dir', type=str, default='./comparison_results',
                       help='Directory to save comparison results')
    parser.add_argument('--se_reduction', type=int, default=16,
                       help='SE block reduction ratio')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test data
    print("\nLoading test dataset...")
    test_loader, test_class_names = get_test_loader(
        args.data_dir, args.batch_size, args.num_workers
    )
    print(f"Test samples: {len(test_loader.dataset)}")
    
    results = {}
    
    # Evaluate ResNet-50
    print("\n" + "="*50)
    print("Evaluating ResNet-50")
    print("="*50)
    
    resnet_checkpoint = torch.load(args.resnet_checkpoint, map_location=device)
    resnet_model = create_resnet50(
        num_classes=resnet_checkpoint['num_classes'], 
        pretrained=False
    )
    resnet_model.load_state_dict(resnet_checkpoint['model_state_dict'])
    resnet_model = resnet_model.to(device)
    resnet_model.eval()
    
    resnet_preds, resnet_labels, resnet_probs = evaluate_model(
        resnet_model, test_loader, device, resnet_checkpoint['class_names']
    )
    resnet_metrics = calculate_detailed_metrics(
        resnet_preds, resnet_labels, resnet_checkpoint['class_names']
    )
    results['resnet50'] = resnet_metrics
    
    print(f"ResNet-50 Accuracy: {resnet_metrics['accuracy']:.4f}")
    print(f"ResNet-50 F1-Score: {resnet_metrics['f1_weighted']:.4f}")
    
    # Evaluate SE-ResNet-50
    print("\n" + "="*50)
    print("Evaluating SE-ResNet-50")
    print("="*50)
    
    se_checkpoint = torch.load(args.se_resnet_checkpoint, map_location=device)
    se_model = create_se_resnet50(
        num_classes=se_checkpoint['num_classes'], 
        pretrained=False,
        reduction=args.se_reduction
    )
    se_model.load_state_dict(se_checkpoint['model_state_dict'])
    se_model = se_model.to(device)
    se_model.eval()
    
    se_preds, se_labels, se_probs = evaluate_model(
        se_model, test_loader, device, se_checkpoint['class_names']
    )
    se_metrics = calculate_detailed_metrics(
        se_preds, se_labels, se_checkpoint['class_names']
    )
    results['se_resnet50'] = se_metrics
    
    print(f"SE-ResNet-50 Accuracy: {se_metrics['accuracy']:.4f}")
    print(f"SE-ResNet-50 F1-Score: {se_metrics['f1_weighted']:.4f}")
    
    # Comparison
    print("\n" + "="*50)
    print("Comparison Summary")
    print("="*50)
    
    print(f"\n{'Metric':<20} {'ResNet-50':<15} {'SE-ResNet-50':<15} {'Difference':<15}")
    print("-" * 65)
    
    metrics_to_compare = [
        ('Accuracy', 'accuracy'),
        ('Precision (W)', 'precision_weighted'),
        ('Recall (W)', 'recall_weighted'),
        ('F1-Score (W)', 'f1_weighted'),
        ('Precision (M)', 'precision_macro'),
        ('Recall (M)', 'recall_macro'),
        ('F1-Score (M)', 'f1_macro')
    ]
    
    for metric_name, metric_key in metrics_to_compare:
        resnet_val = resnet_metrics[metric_key]
        se_val = se_metrics[metric_key]
        diff = se_val - resnet_val
        diff_pct = (diff / resnet_val * 100) if resnet_val > 0 else 0
        
        print(f"{metric_name:<20} {resnet_val:<15.4f} {se_val:<15.4f} {diff:+.4f} ({diff_pct:+.2f}%)")
    
    # Plot comparison
    comparison_path = os.path.join(args.output_dir, 'model_comparison.png')
    plot_metrics_comparison(resnet_metrics, se_metrics, comparison_path)
    
    # Save results
    results_path = os.path.join(args.output_dir, 'comparison_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Comparison complete!")
    print(f"✓ Results saved to {results_path}")
    print(f"✓ Comparison plot saved to {comparison_path}")


if __name__ == '__main__':
    main()

