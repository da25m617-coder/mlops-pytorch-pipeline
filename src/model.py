import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_model(architecture: str = "resnet18", num_classes: int = 10) -> nn.Module:
    """
    Returns ResNet-18 adapted for 28x28 single-channel (grayscale) input.
    """
    if architecture != "resnet18":
        raise ValueError(f"Unsupported architecture: {architecture}")
        
    model = resnet18(weights=None)
    
    # 1. Modify input conv layer for 1-channel grayscale input.
    #    Use kernel_size=3, stride=1 to prevent aggressive downsampling on 28x28 images.
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )
    
    # 2. Bypass initial maxpool so features aren't shrunk to 3x3 too early
    model.maxpool = nn.Identity()
    
    # 3. Modify final linear head for 10 Fashion-MNIST classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model