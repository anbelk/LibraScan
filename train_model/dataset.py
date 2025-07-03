import os
import json
from tqdm import tqdm
from PIL import Image
import pytesseract
from torch.utils.data import Dataset
from torchvision import transforms


def read_labels(labels_file, images_dir):
    with open(labels_file, 'r') as f:
        lines = f.readlines()
    files, labels = [], []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        *filename_parts, label = line.rsplit(' ', 1)
        filename = ' '.join(filename_parts)
        files.append(os.path.join(images_dir, filename))
        labels.append(int(label))
    return files, labels

def cache_ocr_texts(image_paths, cache_path):
    if os.path.exists(cache_path):
        print(f'Loading OCR cache from {cache_path}')
        with open(cache_path, 'r', encoding='utf-8') as f:
            ocr_texts = json.load(f)
        return [ocr_texts.get(os.path.basename(p), '') for p in image_paths]
    else:
        print(f'Creating OCR cache and saving to {cache_path}')
        ocr_texts = {}
        for path in tqdm(image_paths, desc='Extracting OCR text'):
            img = Image.open(path)
            text = pytesseract.image_to_string(img)
            ocr_texts[os.path.basename(path)] = text
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(ocr_texts, f, ensure_ascii=False, indent=2)
        return [ocr_texts.get(os.path.basename(p), '') for p in image_paths]

class BookPagesDataset(Dataset):
    def __init__(self, image_files, labels, ocr_texts=None, transform=None):
        self.image_files = image_files
        self.labels = labels
        self.ocr_texts = ocr_texts
        self.transform = transform

        # Если трансформация не передана, то добавим изменение размера и ToTensor()
        self.default_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Изменяем размер до 224x224
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert('RGB')

        # Применяем трансформации, если они есть
        if self.transform:
            img = self.transform(img)
        else:
            img = self.default_transform(img)

        text = self.ocr_texts[idx] if self.ocr_texts is not None else None
        label = self.labels[idx]
        return img, text, label
