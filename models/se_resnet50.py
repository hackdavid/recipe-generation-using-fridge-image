"""
SE-ResNet-50 Implementation for Ingredient Recognition
Based on: "Squeeze-and-Excitation Networks" (Hu et al., 2018)
Combines ResNet-50 architecture with Squeeze-and-Excitation attention mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    
    The SE block adaptively recalibrates channel-wise feature responses by
    explicitly modeling interdependencies between channels.
    
    Process:
    1. Squeeze: Global average pooling (HxW -> 1x1)
    2. Excitation: Two FC layers with ReLU and Sigmoid
    3. Scale: Channel-wise multiplication
    """
    
    def __init__(self, channels, reduction=16):
        """
        Initialize SE Block
        
        Args:
            channels: Number of input channels
            reduction: Reduction ratio for the bottleneck (default: 16)
        """
        super(SEBlock, self).__init__()
        self.channels = channels
        self.reduction = reduction
        
        # Squeeze: Global Average Pooling
        # This reduces spatial dimensions to 1x1
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        
        # Excitation: Two fully connected layers
        # First FC: channels -> channels//reduction (with ReLU)
        # Second FC: channels//reduction -> channels (with Sigmoid)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass of SE Block
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Scaled tensor with same shape as input
        """
        batch_size, channels, height, width = x.size()
        
        # Squeeze: Global average pooling
        # Output shape: (batch, channels, 1, 1)
        y = self.squeeze(x)
        
        # Reshape for FC layers: (batch, channels, 1, 1) -> (batch, channels)
        y = y.view(batch_size, channels)
        
        # Excitation: Generate channel weights
        # Output shape: (batch, channels)
        y = self.excitation(y)
        
        # Reshape back: (batch, channels) -> (batch, channels, 1, 1)
        y = y.view(batch_size, channels, 1, 1)
        
        # Scale: Element-wise multiplication
        # Broadcasting: (batch, channels, 1, 1) * (batch, channels, H, W)
        return x * y.expand_as(x)


class SEBottleneck(nn.Module):
    """
    Bottleneck ResNet block with Squeeze-and-Excitation attention
    Combines ResNet bottleneck structure with SE block
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, reduction=16):
        """
        Initialize SE-Bottleneck block
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (before expansion)
            stride: Stride for the 3x3 convolution
            downsample: Downsample layer for residual connection
            reduction: SE block reduction ratio
        """
        super(SEBottleneck, self).__init__()
        
        # Standard ResNet bottleneck layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
        # Add SE block after the third convolution
        # SE block operates on the expanded channels
        self.se = SEBlock(out_channels * self.expansion, reduction=reduction)

    def forward(self, x):
        """
        Forward pass of SE-Bottleneck
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with residual connection and SE attention
        """
        identity = x

        # Standard ResNet forward pass
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        # Apply SE attention
        out = self.se(out)

        # Residual connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SEResNet50(nn.Module):
    """
    SE-ResNet-50 Architecture for Ingredient Recognition
    
    This model extends ResNet-50 by adding Squeeze-and-Excitation (SE) blocks
    to each bottleneck block. The SE blocks enable the model to adaptively
    recalibrate channel-wise feature responses, improving feature representation.
    
    Architecture:
    - Input: 224x224 RGB images
    - Output: num_classes ingredient predictions
    - Uses SE-Bottleneck blocks with residual connections and SE attention
    """
    
    def __init__(self, num_classes=357, pretrained=True, reduction=16):
        """
        Initialize SE-ResNet-50 model
        
        Args:
            num_classes: Number of ingredient classes (default: 357)
            pretrained: Whether to use ImageNet pretrained weights
            reduction: SE block reduction ratio (default: 16)
        """
        super(SEResNet50, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        
        # Initial convolution layer (same as ResNet-50)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # SE-ResNet layers (using SE-Bottleneck blocks)
        self.layer1 = self._make_layer(SEBottleneck, 64, 64, 3, stride=1, reduction=reduction)
        self.layer2 = self._make_layer(SEBottleneck, 256, 128, 4, stride=2, reduction=reduction)
        self.layer3 = self._make_layer(SEBottleneck, 512, 256, 6, stride=2, reduction=reduction)
        self.layer4 = self._make_layer(SEBottleneck, 1024, 512, 3, stride=2, reduction=reduction)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * SEBottleneck.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
        # Load pretrained weights if specified
        if pretrained:
            self._load_pretrained_weights()
    
    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1, reduction=16):
        """
        Create a layer of SE-ResNet blocks
        
        Args:
            block: Block type (SEBottleneck)
            in_channels: Input channels
            out_channels: Output channels (before expansion)
            blocks: Number of blocks in the layer
            stride: Stride for the first block
            reduction: SE reduction ratio
            
        Returns:
            Sequential container of blocks
        """
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample, reduction))
        in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(in_channels, out_channels, reduction=reduction))

        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _load_pretrained_weights(self):
        """Load ImageNet pretrained weights (excluding SE blocks)"""
        try:
            pretrained_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            
            # Copy weights from standard ResNet-50
            # Note: SE blocks don't have pretrained weights, they're initialized randomly
            self.conv1.load_state_dict(pretrained_model.conv1.state_dict())
            self.bn1.load_state_dict(pretrained_model.bn1.state_dict())
            
            # Copy weights from each layer (excluding SE blocks)
            self._copy_layer_weights(self.layer1, pretrained_model.layer1)
            self._copy_layer_weights(self.layer2, pretrained_model.layer2)
            self._copy_layer_weights(self.layer3, pretrained_model.layer3)
            self._copy_layer_weights(self.layer4, pretrained_model.layer4)
            
            print("✓ Loaded ImageNet pretrained weights (SE blocks initialized randomly)")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
    
    def _copy_layer_weights(self, se_layer, resnet_layer):
        """
        Copy weights from ResNet layer to SE-ResNet layer
        (excluding SE block weights)
        """
        for se_block, resnet_block in zip(se_layer, resnet_layer):
            se_block.conv1.load_state_dict(resnet_block.conv1.state_dict())
            se_block.bn1.load_state_dict(resnet_block.bn1.state_dict())
            se_block.conv2.load_state_dict(resnet_block.conv2.state_dict())
            se_block.bn2.load_state_dict(resnet_block.bn2.state_dict())
            se_block.conv3.load_state_dict(resnet_block.conv3.state_dict())
            se_block.bn3.load_state_dict(resnet_block.bn3.state_dict())
            if se_block.downsample is not None and resnet_block.downsample is not None:
                se_block.downsample.load_state_dict(resnet_block.downsample.state_dict())
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # SE-ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Classification head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def get_features(self, x):
        """
        Extract features before classification layer
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor of shape (batch_size, 2048)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x


def create_se_resnet50(num_classes=357, pretrained=True, reduction=16):
    """
    Factory function to create SE-ResNet-50 model
    
    Args:
        num_classes: Number of ingredient classes
        pretrained: Whether to use pretrained weights
        reduction: SE block reduction ratio
        
    Returns:
        SEResNet50 model instance
    """
    model = SEResNet50(num_classes=num_classes, pretrained=pretrained, reduction=reduction)
    return model


if __name__ == "__main__":
    # Test model
    model = create_se_resnet50(num_classes=357, pretrained=True, reduction=16)
    
    # Print model architecture
    print("\n" + "="*50)
    print("SE-ResNet-50 Model Summary")
    print("="*50)
    print(f"Number of classes: {model.num_classes}")
    print(f"SE reduction ratio: {model.reduction}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Compare with standard ResNet-50
    from models.resnet50 import create_resnet50
    resnet50_model = create_resnet50(num_classes=357, pretrained=False)
    resnet50_params = sum(p.numel() for p in resnet50_model.parameters())
    print(f"\nResNet-50 parameters: {resnet50_params:,}")
    print(f"SE-ResNet-50 parameters: {total_params:,}")
    print(f"Additional parameters: {total_params - resnet50_params:,}")
    print(f"Overhead: {(total_params - resnet50_params) / resnet50_params * 100:.2f}%")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("="*50)

