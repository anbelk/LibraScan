import os
import pytesseract
from transformers import BertTokenizer, BertModel
import torch
from PIL import Image
from config import TEXT_ENCODER_WEIGHTS_PATH, BATCH_SIZE, DEVICE

class TextEncoder:
    def __init__(self, weights_path):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.model = BertModel.from_pretrained('bert-base-multilingual-cased')

        self.model.load_state_dict(torch.load(weights_path, map_location=DEVICE))

        self.model.to(DEVICE)
        self.model.eval()

    def ocr_images(self, image_paths):
        texts = []
        for path in image_paths:
            img = Image.open(path)
            text = pytesseract.image_to_string(img)
            texts.append(text)
        return texts

    def get_text_emb(self, texts):
        all_embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i+BATCH_SIZE]
            encoded_input = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                output = self.model(**encoded_input)
            emb = output.last_hidden_state[:, 0, :].cpu()
            all_embeddings.append(emb)
        return torch.cat(all_embeddings, dim=0)

    def get_ocr_text_emb(self, image_paths):
        texts = self.ocr_images(image_paths)
        return self.get_text_emb(texts)