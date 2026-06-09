import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from resnet_arch import ResNetArch

MNIST_PATH = '/Users/rithvikharinath/Documents/CSC321/Mastery/resnet18_mnist.pth'


def fgsm_attack(model, images, labels, epsilon):
    images = images.clone().detach().requires_grad_(True)
    loss = nn.CrossEntropyLoss()(model(images), labels)
    model.zero_grad()
    loss.backward()
    return torch.clamp(images + epsilon * images.grad.sign(), -3, 3).detach()


def pgd_attack(model, images, labels, epsilon, steps=20):
    step_size = epsilon / 4
    adv = images.clone().detach()
    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        loss = nn.CrossEntropyLoss()(model(adv), labels)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            adv = adv + step_size * adv.grad.sign()
            adv = torch.clamp(images + torch.clamp(adv - images, -epsilon, epsilon), -3, 3)
    return adv.detach()


def load_model(path=MNIST_PATH):
    device = 'cpu'
    model = ResNetArch(pretrained=False).get_model()
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device).eval(), device


def load_data():
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.1307, 0.1307, 0.1307], [0.3081, 0.3081, 0.3081])
    ])
    dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=0)
    return loader, dataset


def test_attacks():
    model, device = load_model()
    loader, _ = load_data()
    epsilons = [0, 0.05, 0.10, 0.20, 0.30]

    print("Testing FGSM:")
    for eps in epsilons:
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(loader):
            if i >= 5:
                break
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                preds = model(images).argmax(1)
            images = images[preds == labels]
            labels = labels[preds == labels]
            adv = images if eps == 0 else fgsm_attack(model, images, labels, eps)
            with torch.no_grad():
                correct += (model(adv).argmax(1) == labels).sum().item()
                total += labels.size(0)
        if total > 0:
            acc = 100.0 * correct / total
            if eps == 0:
                print(f"Clean: {acc:.1f}%")
            else:
                print(f"ε={eps:.3f}: {acc:.1f}% accuracy, {100-acc:.1f}% attack success")

    print("Testing PGD:")
    for eps in epsilons:
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(loader):
            if i >= 5:
                break
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                preds = model(images).argmax(1)
            images = images[preds == labels]
            labels = labels[preds == labels]
            adv = images if eps == 0 else pgd_attack(model, images, labels, eps)
            with torch.no_grad():
                correct += (model(adv).argmax(1) == labels).sum().item()
                total += labels.size(0)
        if total > 0:
            acc = 100.0 * correct / total
            if eps == 0:
                print(f"Clean: {acc:.1f}%")
            else:
                print(f"ε={eps:.3f}: {acc:.1f}% accuracy, {100-acc:.1f}% attack success")


if __name__ == "__main__":
    test_attacks()
