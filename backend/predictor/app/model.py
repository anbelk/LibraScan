import torch
import torchvision.transforms as transforms


class Classifier:
    def predict(self, image_vec, text_vec):
        return int((sum(image_vec) + sum(text_vec)) % 4)

