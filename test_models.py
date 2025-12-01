"""
Test script to verify ResNet-50 and SE-ResNet-50 models work correctly
"""

import torch
from models import create_resnet50, create_se_resnet50


def test_resnet50():
    """Test ResNet-50 model"""
    print("\n" + "="*50)
    print("Testing ResNet-50")
    print("="*50)
    
    model = create_resnet50(num_classes=357, pretrained=True)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.2f}, {output.max().item():.2f}]")
    
    # Test feature extraction
    features = model.get_features(dummy_input)
    print(f"Feature shape: {features.shape}")
    
    print("✓ ResNet-50 test passed!\n")
    return model


def test_se_resnet50():
    """Test SE-ResNet-50 model"""
    print("\n" + "="*50)
    print("Testing SE-ResNet-50")
    print("="*50)
    
    model = create_se_resnet50(num_classes=357, pretrained=True, reduction=16)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.2f}, {output.max().item():.2f}]")
    
    # Test feature extraction
    features = model.get_features(dummy_input)
    print(f"Feature shape: {features.shape}")
    
    print("✓ SE-ResNet-50 test passed!\n")
    return model


def compare_models():
    """Compare model sizes"""
    print("\n" + "="*50)
    print("Model Comparison")
    print("="*50)
    
    resnet = create_resnet50(num_classes=357, pretrained=False)
    se_resnet = create_se_resnet50(num_classes=357, pretrained=False, reduction=16)
    
    resnet_params = sum(p.numel() for p in resnet.parameters())
    se_resnet_params = sum(p.numel() for p in se_resnet.parameters())
    
    print(f"ResNet-50 parameters:     {resnet_params:,}")
    print(f"SE-ResNet-50 parameters: {se_resnet_params:,}")
    print(f"Difference:              {se_resnet_params - resnet_params:,}")
    print(f"Overhead:                 {(se_resnet_params - resnet_params) / resnet_params * 100:.2f}%")
    
    # Test inference time (rough estimate)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    import time
    
    resnet.eval()
    se_resnet.eval()
    
    # Warmup
    with torch.no_grad():
        _ = resnet(dummy_input)
        _ = se_resnet(dummy_input)
    
    # Time ResNet-50
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = resnet(dummy_input)
    resnet_time = (time.time() - start) / 100
    
    # Time SE-ResNet-50
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = se_resnet(dummy_input)
    se_resnet_time = (time.time() - start) / 100
    
    print(f"\nInference time (100 runs, average):")
    print(f"ResNet-50:     {resnet_time*1000:.2f} ms")
    print(f"SE-ResNet-50: {se_resnet_time*1000:.2f} ms")
    print(f"Overhead:      {(se_resnet_time - resnet_time)*1000:.2f} ms ({(se_resnet_time/resnet_time - 1)*100:.2f}%)")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Model Testing Script")
    print("="*60)
    
    # Test both models
    resnet_model = test_resnet50()
    se_resnet_model = test_se_resnet50()
    
    # Compare models
    compare_models()
    
    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)

