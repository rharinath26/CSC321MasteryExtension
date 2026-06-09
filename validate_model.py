import torch
import torchvision
import torchvision.transforms as transforms
from resnet_arch import ResNetArch

MODEL_PATH = '/Users/rithvikharinath/Documents/CSC321/Mastery/resnet18_cifar10_adversarial_trained.pth'
CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def load_model(path=MODEL_PATH):
    device = 'cpu'
    model = ResNetArch().get_model()
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, device


def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=0)
    return loader, dataset


def validate(path=MODEL_PATH):
    model, device = load_model(path)
    loader, dataset = load_data()

    correct = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(1)
            correct += (preds == labels).sum().item()
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += (preds[i] == labels[i]).item()
                class_total[label] += 1

    print(f"Overall accuracy: {100. * correct / len(dataset):.2f}%")
    print("Per-class accuracy:")
    for i, name in enumerate(CLASSES):
        if class_total[i] > 0:
            print(f"  {name:8s}: {100. * class_correct[i] / class_total[i]:.1f}%")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else MODEL_PATH
    validate(path)
