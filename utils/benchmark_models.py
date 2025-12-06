"""
Comprehensive Model Benchmarking Script

This script benchmarks ResNet-50 and SE-ResNet-50 models on various metrics:
- Inference Speed (Latency, Throughput, FPS)
- Memory Usage (GPU/CPU)
- Model Size
- Parameter Count
- Accuracy Metrics
- Comparison Analysis

Usage:
    python utils/benchmark_models.py --resnet_checkpoint resnet50_best.pth --se_resnet_checkpoint se_resnet50_best.pth --output_dir ./benchmark_results --num_iterations 100 --batch_sizes 1 4 8 16 32
"""

import torch
import torch.nn as nn
import argparse
import json
import os
import time
import statistics
from pathlib import Path
from datetime import datetime
import numpy as np
from typing import Dict, List, Tuple
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet50 import create_resnet50
from models.se_resnet50 import create_se_resnet50


def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def get_model_size(checkpoint_path):
    """Get model checkpoint size in MB"""
    size_bytes = os.path.getsize(checkpoint_path)
    size_mb = size_bytes / (1024 * 1024)
    return {
        'bytes': size_bytes,
        'mb': size_mb,
        'gb': size_mb / 1024
    }


def measure_inference_speed(model, device, input_shape=(1, 3, 224, 224), 
                           num_iterations=100, warmup=10, batch_size=1):
    """
    Measure inference speed
    
    Args:
        model: PyTorch model
        device: torch device
        input_shape: Input tensor shape (batch, channels, height, width)
        num_iterations: Number of inference iterations
        warmup: Number of warmup iterations
        batch_size: Batch size for inference
    
    Returns:
        Dictionary with timing metrics
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, input_shape[1], input_shape[2], input_shape[3]).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    # Synchronize GPU if CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Measure inference time
    inference_times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            _ = model(dummy_input)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            inference_times.append((end_time - start_time) * 1000)  # Convert to ms
    
    # Calculate statistics
    avg_time = statistics.mean(inference_times)
    median_time = statistics.median(inference_times)
    min_time = min(inference_times)
    max_time = max(inference_times)
    std_time = statistics.stdev(inference_times) if len(inference_times) > 1 else 0
    
    # Calculate throughput
    fps = (1000 / avg_time) * batch_size  # Frames per second
    throughput = batch_size / (avg_time / 1000)  # Samples per second
    
    return {
        'latency_ms': {
            'mean': avg_time,
            'median': median_time,
            'min': min_time,
            'max': max_time,
            'std': std_time,
            'p50': median_time,
            'p95': np.percentile(inference_times, 95),
            'p99': np.percentile(inference_times, 99)
        },
        'throughput': {
            'fps': fps,
            'samples_per_second': throughput
        },
        'batch_size': batch_size,
        'num_iterations': num_iterations
    }


def measure_memory_usage(model, device, input_shape=(1, 3, 224, 224)):
    """
    Measure memory usage
    
    Args:
        model: PyTorch model
        device: torch device
        input_shape: Input tensor shape
    
    Returns:
        Dictionary with memory metrics
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape[0], input_shape[1], input_shape[2], input_shape[3]).to(device)
    
    memory_stats = {}
    
    if device.type == 'cuda':
        # GPU Memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        initial_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)  # MB
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
        current_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)  # MB
        
        memory_stats['gpu'] = {
            'initial_mb': initial_memory,
            'peak_mb': peak_memory,
            'current_mb': current_memory,
            'used_mb': peak_memory - initial_memory
        }
    
    # CPU Memory (approximate)
    try:
        import psutil
        process = psutil.Process(os.getpid())
        cpu_memory = process.memory_info().rss / (1024 ** 2)  # MB
        memory_stats['cpu'] = {
            'rss_mb': cpu_memory
        }
    except ImportError:
        # psutil not available, skip CPU memory measurement
        memory_stats['cpu'] = {
            'rss_mb': None,
            'note': 'psutil not installed, CPU memory not measured'
        }
    
    return memory_stats


def benchmark_model(checkpoint_path, model_type, device, num_iterations=100, 
                   batch_sizes=[1, 4, 8, 16, 32], input_size=224):
    """
    Comprehensive benchmark for a single model
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_type: 'resnet50' or 'se_resnet50'
        device: torch device
        num_iterations: Number of inference iterations
        batch_sizes: List of batch sizes to test
        input_size: Input image size
    
    Returns:
        Dictionary with all benchmark results
    """
    print(f"\n{'='*70}")
    print(f"Benchmarking {model_type.upper()}")
    print(f"{'='*70}")
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    num_classes = checkpoint.get('num_classes', 90)
    se_reduction = checkpoint.get('se_reduction', 16) if model_type == 'se_resnet50' else None
    
    # Create model
    if model_type == 'resnet50':
        model = create_resnet50(num_classes=num_classes, pretrained=False)
    else:
        model = create_se_resnet50(num_classes=num_classes, pretrained=False, reduction=se_reduction)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Model characteristics
    print("\nCalculating model characteristics...")
    params = count_parameters(model)
    model_size = get_model_size(checkpoint_path)
    
    # Accuracy metrics from checkpoint
    accuracy_metrics = {
        'best_val_acc': checkpoint.get('best_val_acc', None),
        'epoch': checkpoint.get('epoch', None)
    }
    
    # Benchmark inference speed for different batch sizes
    print("\nMeasuring inference speed...")
    inference_results = {}
    for batch_size in batch_sizes:
        print(f"  Testing batch size: {batch_size}")
        inference_results[f'batch_{batch_size}'] = measure_inference_speed(
            model, device, 
            input_shape=(batch_size, 3, input_size, input_size),
            num_iterations=num_iterations,
            batch_size=batch_size
        )
    
    # Measure memory usage
    print("\nMeasuring memory usage...")
    memory_results = measure_memory_usage(
        model, device,
        input_shape=(1, 3, input_size, input_size)
    )
    
    # Compile results
    results = {
        'model_type': model_type,
        'checkpoint_path': checkpoint_path,
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'input_size': input_size,
        'num_classes': num_classes,
        'model_characteristics': {
            'parameters': params,
            'model_size': model_size
        },
        'accuracy_metrics': accuracy_metrics,
        'inference_speed': inference_results,
        'memory_usage': memory_results
    }
    
    print(f"\n✓ Benchmarking complete for {model_type}")
    return results


def compare_models(resnet_results, se_resnet_results):
    """
    Compare ResNet-50 and SE-ResNet-50 benchmark results
    
    Args:
        resnet_results: ResNet-50 benchmark results
        se_resnet_results: SE-ResNet-50 benchmark results
    
    Returns:
        Dictionary with comparison metrics
    """
    comparison = {
        'parameter_comparison': {
            'resnet50_total': resnet_results['model_characteristics']['parameters']['total'],
            'se_resnet50_total': se_resnet_results['model_characteristics']['parameters']['total'],
            'overhead': se_resnet_results['model_characteristics']['parameters']['total'] - 
                       resnet_results['model_characteristics']['parameters']['total'],
            'overhead_percent': ((se_resnet_results['model_characteristics']['parameters']['total'] - 
                                resnet_results['model_characteristics']['parameters']['total']) /
                               resnet_results['model_characteristics']['parameters']['total']) * 100
        },
        'model_size_comparison': {
            'resnet50_mb': resnet_results['model_characteristics']['model_size']['mb'],
            'se_resnet50_mb': se_resnet_results['model_characteristics']['model_size']['mb'],
            'size_increase_mb': se_resnet_results['model_characteristics']['model_size']['mb'] - 
                               resnet_results['model_characteristics']['model_size']['mb'],
            'size_increase_percent': ((se_resnet_results['model_characteristics']['model_size']['mb'] - 
                                     resnet_results['model_characteristics']['model_size']['mb']) /
                                    resnet_results['model_characteristics']['model_size']['mb']) * 100
        },
        'speed_comparison': {},
        'memory_comparison': {}
    }
    
    # Compare inference speed for each batch size
    for batch_key in resnet_results['inference_speed'].keys():
        resnet_latency = resnet_results['inference_speed'][batch_key]['latency_ms']['mean']
        se_resnet_latency = se_resnet_results['inference_speed'][batch_key]['latency_ms']['mean']
        
        comparison['speed_comparison'][batch_key] = {
            'resnet50_latency_ms': resnet_latency,
            'se_resnet50_latency_ms': se_resnet_latency,
            'latency_increase_ms': se_resnet_latency - resnet_latency,
            'latency_increase_percent': ((se_resnet_latency - resnet_latency) / resnet_latency) * 100,
            'speedup_ratio': resnet_latency / se_resnet_latency if se_resnet_latency > 0 else 0,
            'resnet50_fps': resnet_results['inference_speed'][batch_key]['throughput']['fps'],
            'se_resnet50_fps': se_resnet_results['inference_speed'][batch_key]['throughput']['fps']
        }
    
    # Compare memory usage
    if 'gpu' in resnet_results['memory_usage'] and 'gpu' in se_resnet_results['memory_usage']:
        comparison['memory_comparison']['gpu'] = {
            'resnet50_peak_mb': resnet_results['memory_usage']['gpu']['peak_mb'],
            'se_resnet50_peak_mb': se_resnet_results['memory_usage']['gpu']['peak_mb'],
            'memory_increase_mb': se_resnet_results['memory_usage']['gpu']['peak_mb'] - 
                                 resnet_results['memory_usage']['gpu']['peak_mb'],
            'memory_increase_percent': ((se_resnet_results['memory_usage']['gpu']['peak_mb'] - 
                                       resnet_results['memory_usage']['gpu']['peak_mb']) /
                                      resnet_results['memory_usage']['gpu']['peak_mb']) * 100
        }
    
    # CPU memory comparison (only if both have valid measurements)
    resnet_cpu = resnet_results['memory_usage']['cpu'].get('rss_mb')
    se_resnet_cpu = se_resnet_results['memory_usage']['cpu'].get('rss_mb')
    if resnet_cpu is not None and se_resnet_cpu is not None:
        comparison['memory_comparison']['cpu'] = {
            'resnet50_rss_mb': resnet_cpu,
            'se_resnet50_rss_mb': se_resnet_cpu,
            'memory_increase_mb': se_resnet_cpu - resnet_cpu
        }
    else:
        comparison['memory_comparison']['cpu'] = {
            'note': 'CPU memory measurement not available (psutil not installed)'
        }
    
    # Accuracy comparison
    comparison['accuracy_comparison'] = {
        'resnet50_val_acc': resnet_results['accuracy_metrics'].get('best_val_acc'),
        'se_resnet50_val_acc': se_resnet_results['accuracy_metrics'].get('best_val_acc'),
        'accuracy_improvement': (se_resnet_results['accuracy_metrics'].get('best_val_acc', 0) - 
                                resnet_results['accuracy_metrics'].get('best_val_acc', 0)) if 
                               resnet_results['accuracy_metrics'].get('best_val_acc') and 
                               se_resnet_results['accuracy_metrics'].get('best_val_acc') else None
    }
    
    return comparison


def print_benchmark_summary(resnet_results, se_resnet_results, comparison):
    """Print formatted benchmark summary"""
    
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    
    # Model Characteristics
    print("\n📊 Model Characteristics:")
    print(f"  ResNet-50 Parameters:     {resnet_results['model_characteristics']['parameters']['total']:,}")
    print(f"  SE-ResNet-50 Parameters:  {se_resnet_results['model_characteristics']['parameters']['total']:,}")
    print(f"  Parameter Overhead:      {comparison['parameter_comparison']['overhead']:,} "
          f"({comparison['parameter_comparison']['overhead_percent']:.2f}%)")
    
    print(f"\n  ResNet-50 Model Size:    {resnet_results['model_characteristics']['model_size']['mb']:.2f} MB")
    print(f"  SE-ResNet-50 Model Size:  {se_resnet_results['model_characteristics']['model_size']['mb']:.2f} MB")
    print(f"  Size Increase:           {comparison['model_size_comparison']['size_increase_mb']:.2f} MB "
          f"({comparison['model_size_comparison']['size_increase_percent']:.2f}%)")
    
    # Inference Speed
    print("\n⚡ Inference Speed (Batch Size 1):")
    batch_1 = comparison['speed_comparison']['batch_1']
    print(f"  ResNet-50 Latency:       {batch_1['resnet50_latency_ms']:.2f} ms")
    print(f"  SE-ResNet-50 Latency:    {batch_1['se_resnet50_latency_ms']:.2f} ms")
    print(f"  Latency Increase:        {batch_1['latency_increase_ms']:.2f} ms "
          f"({batch_1['latency_increase_percent']:.2f}%)")
    print(f"  ResNet-50 FPS:           {batch_1['resnet50_fps']:.2f}")
    print(f"  SE-ResNet-50 FPS:        {batch_1['se_resnet50_fps']:.2f}")
    
    # Memory Usage
    if 'gpu' in comparison['memory_comparison']:
        print("\n💾 GPU Memory Usage:")
        gpu_mem = comparison['memory_comparison']['gpu']
        print(f"  ResNet-50 Peak Memory:   {gpu_mem['resnet50_peak_mb']:.2f} MB")
        print(f"  SE-ResNet-50 Peak Memory: {gpu_mem['se_resnet50_peak_mb']:.2f} MB")
        print(f"  Memory Increase:         {gpu_mem['memory_increase_mb']:.2f} MB "
              f"({gpu_mem['memory_increase_percent']:.2f}%)")
    
    # Accuracy
    if comparison['accuracy_comparison']['resnet50_val_acc'] and comparison['accuracy_comparison']['se_resnet50_val_acc']:
        print("\n🎯 Validation Accuracy:")
        acc_comp = comparison['accuracy_comparison']
        print(f"  ResNet-50 Accuracy:      {acc_comp['resnet50_val_acc']:.2f}%")
        print(f"  SE-ResNet-50 Accuracy:   {acc_comp['se_resnet50_val_acc']:.2f}%")
        if acc_comp['accuracy_improvement']:
            print(f"  Accuracy Improvement:    {acc_comp['accuracy_improvement']:.2f}%")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark ResNet-50 and SE-ResNet-50 models'
    )
    parser.add_argument(
        '--resnet_checkpoint',
        type=str,
        required=True,
        help='Path to ResNet-50 checkpoint'
    )
    parser.add_argument(
        '--se_resnet_checkpoint',
        type=str,
        required=True,
        help='Path to SE-ResNet-50 checkpoint'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./benchmark_results',
        help='Directory to save benchmark results'
    )
    parser.add_argument(
        '--num_iterations',
        type=int,
        default=100,
        help='Number of inference iterations for speed measurement'
    )
    parser.add_argument(
        '--batch_sizes',
        type=int,
        nargs='+',
        default=[1, 4, 8, 16, 32],
        help='Batch sizes to test'
    )
    parser.add_argument(
        '--input_size',
        type=int,
        default=224,
        help='Input image size'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu). Auto-detected if not specified'
    )
    
    args = parser.parse_args()
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Benchmark ResNet-50
    resnet_results = benchmark_model(
        checkpoint_path=args.resnet_checkpoint,
        model_type='resnet50',
        device=device,
        num_iterations=args.num_iterations,
        batch_sizes=args.batch_sizes,
        input_size=args.input_size
    )
    
    # Benchmark SE-ResNet-50
    se_resnet_results = benchmark_model(
        checkpoint_path=args.se_resnet_checkpoint,
        model_type='se_resnet50',
        device=device,
        num_iterations=args.num_iterations,
        batch_sizes=args.batch_sizes,
        input_size=args.input_size
    )
    
    # Compare models
    print("\n" + "="*70)
    print("Comparing Models")
    print("="*70)
    comparison = compare_models(resnet_results, se_resnet_results)
    
    # Print summary
    print_benchmark_summary(resnet_results, se_resnet_results, comparison)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save individual results
    resnet_path = os.path.join(args.output_dir, f'resnet50_benchmark_{timestamp}.json')
    with open(resnet_path, 'w') as f:
        json.dump(resnet_results, f, indent=2)
    print(f"\n✓ ResNet-50 results saved to: {resnet_path}")
    
    se_resnet_path = os.path.join(args.output_dir, f'se_resnet50_benchmark_{timestamp}.json')
    with open(se_resnet_path, 'w') as f:
        json.dump(se_resnet_results, f, indent=2)
    print(f"✓ SE-ResNet-50 results saved to: {se_resnet_path}")
    
    # Save comparison
    comparison_path = os.path.join(args.output_dir, f'model_comparison_{timestamp}.json')
    comparison_data = {
        'timestamp': timestamp,
        'device': str(device),
        'resnet50_results': resnet_results,
        'se_resnet50_results': se_resnet_results,
        'comparison': comparison
    }
    with open(comparison_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    print(f"✓ Comparison results saved to: {comparison_path}")
    
    print(f"\n✓ All benchmark results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

