#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from resnet_arch import ResNetArch
import numpy as np
import os

class AdversarialTrainer:
    def __init__(self, model_path=None, learning_rate=0.001):
        self.device = 'cpu'
        
        self.resnet = ResNetArch()
        self.model = self.resnet.get_model().to(self.device)
        
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded pretrained model from {model_path}")
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
    def load_data(self):
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=test_transform
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=128, shuffle=True, num_workers=0
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=100, shuffle=False, num_workers=0
        )
        return self.train_loader, self.test_loader
    
    def center_focused_fgsm(self, images, labels, epsilon=0.03, center_weight=2.0):
        images_copy = images.clone().detach().requires_grad_(True)
        outputs = self.model(images_copy)
        loss = self.criterion(outputs, labels)
        self.model.zero_grad()
        loss.backward()
        gradients = images_copy.grad.data
        _, _, height, _ = gradients.shape
        center_mask = torch.ones_like(gradients)
        center_start = height // 4
        center_end = 3 * height // 4
        center_mask[:, :, center_start:center_end, center_start:center_end] *= center_weight
        weighted_gradients = gradients * center_mask
        gradient_signs = weighted_gradients.sign()
        perturbation = epsilon * gradient_signs
        adversarial_images = images_copy + perturbation
        return torch.clamp(adversarial_images, -3, 3).detach()
    
    def pgd_attack(self, images, labels, epsilon=0.03, step_size=0.007, steps=10):
        adv_images = images.clone().detach()
        for _ in range(steps):
            adv_images = adv_images.detach().requires_grad_(True)
            outputs = self.model(adv_images)
            loss = self.criterion(outputs, labels)
            self.model.zero_grad()
            loss.backward()
            with torch.no_grad():
                adv_images = adv_images + step_size * adv_images.grad.sign()
                perturbation = torch.clamp(adv_images - images, -epsilon, epsilon)
                adv_images = torch.clamp(images + perturbation, -3, 3)
        
        return adv_images.detach()
    
    def frequency_aware_loss(self, images, adv_images, labels):
        adv_outputs = self.model(adv_images)
        adv_loss = self.criterion(adv_outputs, labels)
        
        def high_pass_filter(x):
            kernel = torch.tensor([[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]).float()
            kernel = kernel.expand(3, 1, 3, 3).to(x.device)
            return F.conv2d(x, kernel, padding=1, groups=3)
        
        original_edges = high_pass_filter(images)
        adv_edges = high_pass_filter(adv_images)
        edge_preservation_loss = F.mse_loss(adv_edges, original_edges)
        
        return adv_loss + 0.1 * edge_preservation_loss
    
    def train_adversarial(self, epochs=10, adversarial_ratio=0.6):
        print("Starting adversarial training")
        print(f"Adversarial ratio: {adversarial_ratio:.1%}")
        
        self.model.train()
        
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                if batch_idx >= 100:
                    break
                    
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                if np.random.random() < adversarial_ratio:
                    if np.random.random() < 0.7:
                        adv_images = self.center_focused_fgsm(images, labels, epsilon=np.random.uniform(0.01, 0.03))
                    else:
                        adv_images = self.pgd_attack(images, labels, epsilon=np.random.uniform(0.02, 0.04))
                    loss = self.frequency_aware_loss(images, adv_images, labels)
                    with torch.no_grad():
                        outputs = self.model(adv_images)
                        _, predicted = outputs.max(1)
                        correct += predicted.eq(labels).sum().item()
                        total += labels.size(0)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    with torch.no_grad():
                        _, predicted = outputs.max(1)
                        correct += predicted.eq(labels).sum().item()
                        total += labels.size(0)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                
                if batch_idx % 50 == 49:
                    accuracy = 100.0 * correct / total
                    print(f'Epoch {epoch+1}, Batch {batch_idx+1:3d}, Loss: {running_loss/50:.3f}, Acc: {accuracy:.1f}%')
                    running_loss = 0.0
                    correct = 0
                    total = 0
    
    def save_model(self, filepath):
        checkpoint = {'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")

def main():
    pretrained_path = '/Users/rithvikharinath/Documents/CSC321/Mastery/resnet18_cifar10_pytorch.pth'
    trainer = AdversarialTrainer(model_path=pretrained_path, learning_rate=0.0001)
    
    _, _ = trainer.load_data()
    
    trainer.train_adversarial(epochs=5, adversarial_ratio=0.6)
    trainer.save_model('resnet18_adversarial_trained.pth')

if __name__ == "__main__":
    main()