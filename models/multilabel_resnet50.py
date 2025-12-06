"""
Multi-Label ResNet-50 Model

This module implements a multi-label classification model using ResNet-50 as the encoder.
Supports three training modes:
1. Freeze encoder: Only train the classifier head
2. Full training: Train both encoder and classifier
3. Fine-tuning: Train with lower learning rate for encoder
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50 as tv_resnet50, ResNet50_Weights
from .resnet50 import ResNet50


class MultiLabelResNet50(nn.Module):
    """
    Multi-label classification model using ResNet-50 as encoder
    
    Architecture:
    - ResNet-50 encoder (ImageNet pretrained)
    - Multi-label classifier head with sigmoid activation
    """
    
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        checkpoint_path: str = None,
        dropout: float = 0.5,
        freeze_encoder: bool = False
    ):
        """
        Initialize multi-label ResNet-50 model
        
        Args:
            num_classes: Number of classes (labels)
            pretrained: Whether to use ImageNet pretrained weights (if checkpoint_path is None)
            checkpoint_path: Path to custom checkpoint file (overrides pretrained)
            dropout: Dropout rate for classifier head
            freeze_encoder: Whether to freeze encoder weights
        """
        super(MultiLabelResNet50, self).__init__()
        
        self.num_classes = num_classes
        self.freeze_encoder = freeze_encoder
        
        # Load ResNet-50 encoder
        if checkpoint_path:
            # Load custom checkpoint using our own ResNet-50 architecture
            print(f"Loading encoder from checkpoint: {checkpoint_path}")
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Get num_classes from checkpoint if available (for compatibility)
            checkpoint_num_classes = checkpoint.get('num_classes', None)
            if checkpoint_num_classes is None and 'model_state_dict' in checkpoint:
                # Try to infer from checkpoint
                state_dict = checkpoint['model_state_dict']
                # Look for fc layer to infer num_classes
                for key in state_dict.keys():
                    if 'fc.weight' in key or 'classifier.weight' in key:
                        checkpoint_num_classes = state_dict[key].shape[0]
                        break
            
            # Initialize our custom ResNet-50 architecture
            # Use checkpoint num_classes if available, otherwise use a placeholder
            temp_num_classes = checkpoint_num_classes if checkpoint_num_classes else 1000
            self.encoder = ResNet50(num_classes=temp_num_classes, pretrained=False)
            
            # Extract encoder weights from checkpoint
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Filter encoder weights (remove classifier/fc layer)
            encoder_state_dict = {}
            for key, value in state_dict.items():
                # Skip classifier/fc layer
                if 'fc' in key or 'classifier' in key:
                    continue
                # Remove 'encoder.' prefix if present
                new_key = key.replace('encoder.', '') if key.startswith('encoder.') else key
                encoder_state_dict[new_key] = value
            
            # Load encoder weights
            try:
                missing_keys, unexpected_keys = self.encoder.load_state_dict(encoder_state_dict, strict=False)
                if missing_keys:
                    print(f"⚠ Missing keys (will use random init): {len(missing_keys)}")
                if unexpected_keys:
                    print(f"⚠ Unexpected keys (ignored): {len(unexpected_keys)}")
                print("✓ Loaded encoder weights from checkpoint")
            except Exception as e:
                print(f"⚠ Warning: Could not fully load encoder weights: {e}")
                print("  Attempting partial load...")
                self.encoder.load_state_dict(encoder_state_dict, strict=False)
        elif pretrained:
            # Use ImageNet pretrained weights via torchvision
            self.encoder = tv_resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            print("✓ Loaded ImageNet pretrained weights")
        else:
            # Random initialization using our custom architecture
            self.encoder = ResNet50(num_classes=1000, pretrained=False)
            print("✓ Using randomly initialized encoder (custom architecture)")
        
        # Remove the original classifier (fc layer)
        self.encoder.fc = nn.Identity()
        
        # Get feature dimension (2048 for ResNet-50)
        self.feature_dim = 2048
        
        # Multi-label classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        # Freeze encoder if requested
        if freeze_encoder:
            self.freeze_encoder_weights()
    
    def freeze_encoder_weights(self):
        """Freeze all encoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.freeze_encoder = True
        print("✓ Encoder weights frozen")
    
    def unfreeze_encoder_weights(self):
        """Unfreeze all encoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.freeze_encoder = False
        print("✓ Encoder weights unfrozen")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            logits: Raw logits before sigmoid (B, num_classes)
        """
        # Extract features using encoder
        features = self.encoder(x)
        
        # Classify
        logits = self.classifier(features)
        
        return logits
    
    def predict_proba(self, x, threshold: float = 0.5):
        """
        Predict probabilities and binary labels
        
        Args:
            x: Input tensor
            threshold: Threshold for binary classification
        
        Returns:
            probabilities: Sigmoid probabilities (B, num_classes)
            predictions: Binary predictions (B, num_classes)
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= threshold).long()
        
        return probabilities, predictions


def create_multilabel_resnet50(
    num_classes: int,
    pretrained: bool = True,
    checkpoint_path: str = None,
    dropout: float = 0.5,
    freeze_encoder: bool = False
) -> MultiLabelResNet50:
    """
    Factory function to create multi-label ResNet-50 model
    
    Args:
        num_classes: Number of classes
        pretrained: Whether to use ImageNet pretrained weights (if checkpoint_path is None)
        checkpoint_path: Path to custom checkpoint file (overrides pretrained)
        dropout: Dropout rate
        freeze_encoder: Whether to freeze encoder
    
    Returns:
        MultiLabelResNet50 model
    """
    model = MultiLabelResNet50(
        num_classes=num_classes,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
        dropout=dropout,
        freeze_encoder=freeze_encoder
    )
    return model

