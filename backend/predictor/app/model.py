import torch
import torchvision.transforms as transforms
from train_model.predictor import Predictor


class Classifier:
    def __init__(self):
        self.model = Predictor()

    def predict(self, image_vec, text_vec):
        text_tensor = torch.tensor(text_vec, dtype=torch.float32).unsqueeze(0)  # [1, 768]
        image_tensor = torch.tensor(image_vec, dtype=torch.float32).unsqueeze(0)  # [1, 2048]

        preds = self.model.predict_from_embedding(image_tensor, text_tensor)
        print(preds, preds.type())
        return preds

