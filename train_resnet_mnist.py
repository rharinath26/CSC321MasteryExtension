#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from resnet_arch import ResNetArch


def create_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.1307, 0.1307, 0.1307], [0.3081, 0.3081, 0.3081])
    ])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


def train():
    device = 'cpu'
    train_loader, test_loader = create_loaders()
    resnet = ResNetArch(pretrained=False)
    model = resnet.get_model().to(device)
    print(f"Parameters: {resnet.get_num_params():,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 8], gamma=0.1)

    best_acc = 0.0

    for epoch in range(10):
        model.train()
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if batch_idx % 200 == 0:
                print(f'Epoch {epoch+1}/10, Batch {batch_idx}, Loss: {loss.item():.4f}')

        train_acc = 100. * correct / total

        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()

        test_acc = 100. * test_correct / test_total
        scheduler.step()

        print(f'Epoch {epoch+1}: Train {train_acc:.1f}%, Test {test_acc:.1f}%')

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_accuracy': best_acc,
                'epoch': epoch,
            }, 'resnet18_mnist.pth')
            print(f'New best model saved: {best_acc:.2f}%')

    print(f'Best accuracy: {best_acc:.2f}%')


if __name__ == "__main__":
    train()
