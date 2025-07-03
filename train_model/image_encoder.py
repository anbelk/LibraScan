import os
import torch
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
from config import IMAGE_ENCODER_WEIGHTS_PATH, BATCH_SIZE, DEVICE

class ImageEncoder:
    def __init__(self, weights_path):
        self.model = resnet50(weights=None)
        self.model.fc = torch.nn.Identity()

        self.model.load_state_dict(torch.load(weights_path, map_location=DEVICE))

        self.model.to(DEVICE)
        self.model.eval()

    def get_image_emb(self, images):
        with torch.no_grad():
            return self.model(images)