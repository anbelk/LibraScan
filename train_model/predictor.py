import torch
import os
import torch.nn as nn
from train_model.text_encoder import TextEncoder
from train_model.image_encoder import ImageEncoder
from train_model.config import BEST_FNN_WEIGHTS_PATH, IMAGE_ENCODER_WEIGHTS_PATH, TEXT_ENCODER_WEIGHTS_PATH, FNN_WEIGHTS_PATH, BATCH_SIZE, DEVICE

class FNN(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


class Predictor:
    def __init__(self, use_text=True, use_image=True, device=DEVICE):
        self.device = device
        self.use_text = use_text
        self.use_image = use_image

        self.text_encoder = TextEncoder(TEXT_ENCODER_WEIGHTS_PATH) if use_text else None
        # self.image_encoder = ImageEncoder(IMAGE_ENCODER_WEIGHTS_PATH) if use_image else None
        self.image_encoder = ImageEncoder() if use_image else None
        
        input_dim = 0
        if use_text:
            input_dim += 768  # BERT output size
        if use_image:
            input_dim += 2048  # ResNet output size

        self.model = FNN(input_dim).to(device)
        if os.path.exists(FNN_WEIGHTS_PATH):
            print(f"Loading FNN weights from {BEST_FNN_WEIGHTS_PATH}")
            self.model.load_state_dict(torch.load(BEST_FNN_WEIGHTS_PATH, map_location=device))
            self.model.eval()
        else:
            print(f"FNN weights not found at {BEST_FNN_WEIGHTS_PATH}, using random initialization")

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def extract_features(self, images, texts):
        features = []
        if self.use_text:
            text_embs = self.text_encoder.get_text_emb(texts)  # texts уже список строк [B]
            features.append(text_embs)
        if self.use_image:
            image_embs = self.image_encoder.get_image_emb(images)  # images — пути к картинкам [B]
            features.append(image_embs)
        combined = torch.cat(features, dim=1)  # [B, input_dim]
        return combined

    def predict(self, images, texts):
        self.model.eval()
        with torch.no_grad():
            features = self.extract_features(images, texts).to(self.device)  # [B, input_dim]
            logits = self.model(features)
            probs = torch.softmax(logits, dim=1)
        return probs

    def train_step(self, images, texts, labels):
        self.model.train()
        self.optimizer.zero_grad()
        batch_features = self.extract_features(images, texts).to(self.device)  # [B, input_dim]
        labels = labels.to(self.device).long()  # Убедитесь, что это torch.long
        outputs = self.model(batch_features)  # Получаем логиты от модели
        loss = self.criterion(outputs, labels)  # Вычисляем потерю
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict_from_embedding(self, text_vector, image_vector):
        self.model.eval()
        with torch.no_grad():
            features = torch.cat([text_vector, image_vector], dim=1) .to(self.device)
            logits = self.model(features)
            probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds