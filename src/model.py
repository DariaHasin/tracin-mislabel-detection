import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet18(num_classes=100):
    """
    Build a ResNet-18 for CIFAR-100.
    """
    model = models.resnet18(weights=None)
    
    # Replace the final layer to match CIFAR-100 (default is ImageNet 1000 classes)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model