#!/usr/bin/env python3

import torch
import torch.nn as nn
import torchvision.models as models

class ResNetArch:
    def __init__(self, pretrained=True):
        self.model = models.resnet18(pretrained=pretrained)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 10)
    
    def get_model(self):
        return self.model
    
    def get_num_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

if __name__ == "__main__":
    resnet = ResNetArch(pretrained=False)
    model = resnet.get_model()
    
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    
    print(f"Parameters: {resnet.get_num_params():,}")
    print(f"Input: {x.shape}, Output: {output.shape}")