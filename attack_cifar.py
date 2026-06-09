import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from resnet_arch import ResNetArch

STD_PATH = '/Users/rithvikharinath/Documents/CSC321/Mastery/resnet18_cifar10.pth'
ADV_PATH = '/Users/rithvikharinath/Documents/CSC321/Mastery/resnet18_cifar10_adversarial_trained.pth'

def fgsm_attack(model, image, target, epsilon):
    image.requires_grad = True
    output = model(image)
    loss = nn.CrossEntropyLoss()(output, target)
    model.zero_grad()
    loss.backward()
    gradient_sign = image.grad.data.sign()
    adv_image = image + epsilon * gradient_sign
    return torch.clamp(adv_image, -3, 3)

def pgd_attack(model, image, target, epsilon=0.03, step_size=0.01, steps=10):
    adv_image = image.clone().detach()
    for _ in range(steps):
        adv_image = adv_image.detach().requires_grad_(True)
        output = model(adv_image)
        loss = nn.CrossEntropyLoss()(output, target)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            adv_image = adv_image + step_size * adv_image.grad.sign()
            perturbation = torch.clamp(adv_image - image, -epsilon, epsilon)
            adv_image = torch.clamp(image + perturbation, -3, 3)
    return adv_image.detach()

def targeted_fgsm_attack(model, image, target_class, epsilon):
    image_copy = image.clone().detach().requires_grad_(True)
    output = model(image_copy)
    target = torch.full((image.shape[0],), target_class, dtype=torch.long).to(image.device)
    loss = nn.CrossEntropyLoss()(output, target)
    model.zero_grad()
    loss.backward()
    gradient_sign = image_copy.grad.data.sign()
    adv_image = image_copy - epsilon * gradient_sign
    return torch.clamp(adv_image, -3, 3).detach()

def targeted_pgd_attack(model, image, target_class, epsilon=0.03, step_size=0.01, steps=20):
    adv_image = image.clone().detach()
    target = torch.full((image.shape[0],), target_class, dtype=torch.long).to(image.device)
    for _ in range(steps):
        adv_image = adv_image.detach().requires_grad_(True)
        output = model(adv_image)
        loss = nn.CrossEntropyLoss()(output, target)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            adv_image = adv_image - step_size * adv_image.grad.sign()
            perturbation = torch.clamp(adv_image - image, -epsilon, epsilon)
            adv_image = torch.clamp(image + perturbation, -3, 3)
    
    return adv_image.detach()

def load_model(path=STD_PATH):
    device = 'cpu'
    resnet = ResNetArch()
    model = resnet.get_model()
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
    loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)
    return loader, dataset

def test_attacks(model=None, device=None):
    if model is None:
        model, device = load_model()
    loader, _ = load_data()
    epsilons = [0, 0.01, 0.03, 0.05, 0.1]
    results = {}
    
    print("Testing FGSM:")
    for eps in epsilons:
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(loader):
            if i >= 5:
                break
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                og_outputs = model(images)
                og_predictions = og_outputs.argmax(1)
                
            images = images[og_predictions == labels]
            labels = labels[og_predictions == labels]
            if eps == 0:
                adv_images = images
            else:
                adv_images = fgsm_attack(model, images, labels, eps)
            with torch.no_grad():
                adv_outputs = model(adv_images)
                adv_preds = adv_outputs.argmax(1)
            correct += (adv_preds == labels).sum().item()
            total += labels.size(0)
        if total > 0:
            accuracy = 100.0 * correct / total
            results[('fgsm', eps)] = accuracy
            if eps == 0:
                print(f"Clean: {accuracy:.1f}%")
            else:
                print(f"ε={eps:.3f}: {accuracy:.1f}% accuracy, {100-accuracy:.1f}% attack success")

    print("Testing PGD:")
    for eps in epsilons:
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(loader):
            if i >= 5:
                break  
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                og_outputs = model(images)
                og_predictions = og_outputs.argmax(1)
                
            images = images[og_predictions == labels]
            labels = labels[og_predictions == labels]
            if eps == 0:
                adv_images = images
            else:
                adv_images = pgd_attack(model, images, labels, eps)
            with torch.no_grad():
                adv_outputs = model(adv_images)
                adv_preds = adv_outputs.argmax(1)
            
            correct += (adv_preds == labels).sum().item()
            total += labels.size(0)
        if total > 0:
            accuracy = 100.0 * correct / total
            results[('pgd', eps)] = accuracy
            if eps == 0:
                print(f"Clean: {accuracy:.1f}%")
            else:
                print(f"ε={eps:.3f}: {accuracy:.1f}% accuracy, {100-accuracy:.1f}% attack success")
    
    return results


def compare_models():
    standard_model, device = load_model(STD_PATH)
    adv_model, _ = load_model(ADV_PATH)

    standard_results = test_attacks(standard_model, device)
    adv_results = test_attacks(adv_model, device)

    epsilons = [0.01, 0.03, 0.05, 0.1]
    print("Comparison (accuracy):")
    for eps in epsilons:
        fs = standard_results[('fgsm', eps)]
        fa = adv_results[('fgsm', eps)]
        ps = standard_results[('pgd', eps)]
        pa = adv_results[('pgd', eps)]
        print(f"ε={eps:.2f}  FGSM std: {fs:.1f}%  FGSM adv: {fa:.1f}%  PGD std: {ps:.1f}%  PGD adv: {pa:.1f}%")
def black_box_transfer():
    surrogate, device = load_model(STD_PATH)
    target_model, _ = load_model(ADV_PATH)
    loader, _ = load_data()
    epsilons = [0.01, 0.03, 0.05, 0.1]

    for epsilon in epsilons:
        bb_fgsm = 0
        bb_pgd = 0
        total = 0

        for i, (images, labels) in enumerate(loader):
            if i >= 5:
                break

            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                std_preds = surrogate(images).argmax(1)
                adv_preds = target_model(images).argmax(1)
            both_correct = (std_preds == labels) & (adv_preds == labels)
            if both_correct.sum() == 0:
                continue

            imgs = images[both_correct]
            lbls = labels[both_correct]

            fgsm_adv = fgsm_attack(surrogate, imgs.clone(), lbls, epsilon)
            with torch.no_grad():
                bb_fgsm += (target_model(fgsm_adv).argmax(1) == lbls).sum().item()

            pgd_adv = pgd_attack(surrogate, imgs.clone(), lbls, epsilon)
            with torch.no_grad():
                bb_pgd += (target_model(pgd_adv).argmax(1) == lbls).sum().item()

            total += lbls.size(0)

        if total > 0:
            print(f"ε={epsilon:.2f}  FGSM: {100*bb_fgsm/total:.1f}%  PGD: {100*bb_pgd/total:.1f}%")


def targeted_attacks():
    target_class = 0
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    std_model, device = load_model(STD_PATH)
    adv_model, _ = load_model(ADV_PATH)
    loader, _ = load_data()
    epsilons = [0.01, 0.03, 0.05, 0.1]

    print(f"Targeted attacks (target class: '{classes[target_class]}', attack success rate):")

    for epsilon in epsilons:
        fgsm_std_success = 0
        fgsm_adv_success = 0
        pgd_std_success = 0
        pgd_adv_success = 0
        total = 0

        for i, (images, labels) in enumerate(loader):
            if i >= 5:
                break

            images, labels = images.to(device), labels.to(device)

            not_target = (labels != target_class)
            if not_target.sum() == 0:
                continue

            imgs = images[not_target]
            lbls = labels[not_target]

            fgsm_std_adv = targeted_fgsm_attack(std_model, imgs.clone(), target_class, epsilon)
            with torch.no_grad():
                fgsm_std_success += (std_model(fgsm_std_adv).argmax(1) == target_class).sum().item()

            fgsm_adv_adv = targeted_fgsm_attack(adv_model, imgs.clone(), target_class, epsilon)
            with torch.no_grad():
                fgsm_adv_success += (adv_model(fgsm_adv_adv).argmax(1) == target_class).sum().item()

            pgd_std_adv = targeted_pgd_attack(std_model, imgs.clone(), target_class, epsilon)
            with torch.no_grad():
                pgd_std_success += (std_model(pgd_std_adv).argmax(1) == target_class).sum().item()

            pgd_adv_adv = targeted_pgd_attack(adv_model, imgs.clone(), target_class, epsilon)
            with torch.no_grad():
                pgd_adv_success += (adv_model(pgd_adv_adv).argmax(1) == target_class).sum().item()

            total += lbls.size(0)

        if total > 0:
            fs = 100.0 * fgsm_std_success / total
            fa = 100.0 * fgsm_adv_success / total
            ps = 100.0 * pgd_std_success / total
            pa = 100.0 * pgd_adv_success / total
            print(f"ε={epsilon:.2f}  FGSM std: {fs:.1f}%  FGSM adv: {fa:.1f}%  PGD std: {ps:.1f}%  PGD adv: {pa:.1f}%")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "compare":
            compare_models()
        elif mode == "blackbox":
            black_box_transfer()
        elif mode == "targeted":
            targeted_attacks()
        else:
            test_attacks()
    else:
        test_attacks()