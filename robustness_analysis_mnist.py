import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from resnet_arch import ResNetArch

MODEL_PATH = '/Users/rithvikharinath/Documents/CSC321/Mastery/resnet18_mnist.pth'


def load_model():
    device = 'cpu'
    model = ResNetArch(pretrained=False).get_model()
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, device


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


def get_correct(model, images, labels):
    with torch.no_grad():
        preds = model(images).argmax(1)
    mask = preds == labels
    return images[mask], labels[mask], preds[mask]


def feature_occlusion_robustness(model, device, loader, patch_size=4):
    print("Feature Occlusion Robustness:")

    region_scores = {}
    total = 0

    for i, (images, labels) in enumerate(loader):
        if i >= 5:
            break
        images, labels = images.to(device), labels.to(device)
        images, labels, _ = get_correct(model, images, labels)
        if len(images) == 0:
            continue

        for y in range(0, 32 - patch_size, patch_size):
            for x in range(0, 32 - patch_size, patch_size):
                occluded = images.clone()
                occluded[:, :, y:y+patch_size, x:x+patch_size] = 0
                with torch.no_grad():
                    preds = model(occluded).argmax(1)
                score = (preds == labels).float().mean().item()
                region_scores.setdefault((y, x), []).append(score)

        total += len(images)

    avg = {k: np.mean(v) for k, v in region_scores.items()}
    ranked = sorted(avg.items(), key=lambda x: x[1], reverse=True)

    print(f"Tested {total} images, patch size: {patch_size}x{patch_size}")
    print("Most robust regions:")
    for (y, x), score in ranked[:5]:
        print(f"  ({y},{x}): {score:.3f}")
    print("Least robust regions:")
    for (y, x), score in ranked[-5:]:
        print(f"  ({y},{x}): {score:.3f}")

    return avg


def frequency_robustness(model, device, loader):
    print("Frequency Component Robustness:")

    def gaussian_blur(images, kernel_size=5):
        sigma = kernel_size / 6
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        g = torch.exp(-x**2 / (2 * sigma**2))
        g = g / g.sum()
        kernel = (g[:, None] * g[None, :]).expand(3, 1, kernel_size, kernel_size).to(images.device)
        return F.conv2d(images, kernel, padding=kernel_size // 2, groups=3)

    scores = {'edges_removed': [], 'edges_amplified': []}

    for i, (images, labels) in enumerate(loader):
        if i >= 5:
            break
        images, labels = images.to(device), labels.to(device)
        images, labels, _ = get_correct(model, images, labels)
        if len(images) == 0:
            continue

        blurred = gaussian_blur(images)
        test_cases = {
            'edges_removed': blurred,
            'edges_amplified': torch.clamp(images + 0.1 * (images - blurred), -3, 3),
        }

        for freq_type, modified in test_cases.items():
            with torch.no_grad():
                preds = model(modified).argmax(1)
            scores[freq_type].append((preds == labels).float().mean().item())

    for freq_type, vals in scores.items():
        print(f"  {freq_type}: {np.mean(vals):.3f}")

    return scores


def visualize_robustness_map(data, title, filename):
    grid = np.zeros((8, 8))

    for (y, x), val in data.items():
        gy, gx = y // 4, x // 4
        if 0 <= gy < 8 and 0 <= gx < 8:
            grid[gy, gx] = val

    plt.figure(figsize=(8, 6))
    plt.imshow(grid, cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(label='Prediction Stability')
    plt.title(title)
    plt.xlabel('X Position (Grid)')
    plt.ylabel('Y Position (Grid)')
    for i in range(8):
        for j in range(8):
            if grid[i, j] > 0:
                plt.text(j, i, f'{grid[i, j]:.2f}', ha='center', va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def main():
    model, device = load_model()
    loader, _ = load_data()

    occlusion = feature_occlusion_robustness(model, device, loader)
    frequency_robustness(model, device, loader)

    visualize_robustness_map(occlusion, "MNIST Occlusion Robustness Map", "mnist_occlusion_robustness_map.png")


if __name__ == "__main__":
    main()
