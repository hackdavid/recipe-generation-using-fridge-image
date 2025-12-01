"""
ResNet-50 Implementation for Ingredient Recognition
Based on: "Deep Residual Learning for Image Recognition" (He et al., 2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class BasicBlock(nn.Module):
    """Basic ResNet block with 3x3 convolutions"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck ResNet block with 1x1, 3x3, 1x1 convolutions"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
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

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    """
    ResNet-50 Architecture for Ingredient Recognition
    
    Architecture:
    - Input: 224x224 RGB images
    - Output: num_classes ingredient predictions
    - Uses Bottleneck blocks with residual connections
    """
    
    def __init__(self, num_classes=357, pretrained=True):
        """
        Initialize ResNet-50 model
        
        Args:
            num_classes: Number of ingredient classes (default: 357)
            pretrained: Whether to use ImageNet pretrained weights
        """
        super(ResNet50, self).__init__()
        self.num_classes = num_classes
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers (using Bottleneck blocks)
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 3, stride=1)
        self.layer2 = self._make_layer(Bottleneck, 256, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 512, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 1024, 512, 3, stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
        # Load pretrained weights if specified
        if pretrained:
            self._load_pretrained_weights()
    
    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        """Create a layer of ResNet blocks"""
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(in_channels, out_channels))

        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def _load_pretrained_weights(self):
        """Load ImageNet pretrained weights"""
        try:
            pretrained_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            # Copy weights except the final classifier layer
            self.conv1.load_state_dict(pretrained_model.conv1.state_dict())
            self.bn1.load_state_dict(pretrained_model.bn1.state_dict())
            self.layer1.load_state_dict(pretrained_model.layer1.state_dict())
            self.layer2.load_state_dict(pretrained_model.layer2.state_dict())
            self.layer3.load_state_dict(pretrained_model.layer3.state_dict())
            self.layer4.load_state_dict(pretrained_model.layer4.state_dict())
            print("✓ Loaded ImageNet pretrained weights")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
    
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
        
        # ResNet layers
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


def create_resnet50(num_classes=357, pretrained=True):
    """
    Factory function to create ResNet-50 model
    
    Args:
        num_classes: Number of ingredient classes
        pretrained: Whether to use pretrained weights
        
    Returns:
        ResNet50 model instance
    """
    model = ResNet50(num_classes=num_classes, pretrained=pretrained)
    return model


if __name__ == "__main__":
    # Test model
    model = create_resnet50(num_classes=357, pretrained=True)
    
    # Print model architecture
    print("\n" + "="*50)
    print("ResNet-50 Model Summary")
    print("="*50)
    print(f"Number of classes: {model.num_classes}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("="*50)

