import torch
import torchvision.transforms as transforms

CLASSES = ["cat", "dog", "car", "tree"]


def predict_class(image):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    tensor = transform(image)
    avg = tensor.mean().item()
    index = int(avg * len(CLASSES)) % len(CLASSES)
    return CLASSES[index]