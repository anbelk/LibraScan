import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import pytesseract
from config import (
    TRAIN_IMAGES_DIR, TRAIN_LABELS_FILE,
    AUGMENTED_TRAIN_IMAGES_DIR, AUGMENTED_TRAIN_LABELS_FILE,
    TEST_IMAGES_DIR, TEST_LABELS_FILE,
    OCR_CACHE_PATH, OCR_CACHE_TEST_PATH,
    BATCH_SIZE, EPOCHS, DEVICE, EARLY_STOPPING_PATIENCE,
    MODEL_SAVE_PATH, RESULTS_SAVE_PATH
)
from predictor import Predictor
from dataset import BookPagesDataset, read_labels, cache_ocr_texts

def collate_fn(batch):
    images, texts, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels).long()
    return images, texts, labels

def train_and_evaluate(train_files, train_labels, test_files, test_labels,
                       use_text=True, use_image=True):
    print(f'Training with use_text={use_text}, use_image={use_image}')

    train_ocr_texts = cache_ocr_texts(train_files, OCR_CACHE_PATH) if use_text else None
    test_ocr_texts = cache_ocr_texts(test_files, OCR_CACHE_TEST_PATH) if use_text else None

    transform = None
    if use_image:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    train_dataset = BookPagesDataset(train_files, train_labels, train_ocr_texts, transform)
    test_dataset = BookPagesDataset(test_files, test_labels, test_ocr_texts, transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    predictor = Predictor(use_text=use_text, use_image=use_image, device=DEVICE)
    best_val_loss = float('inf')
    patience = 0
    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        predictor.model.train()
        running_loss = 0
        for images, texts, labels in tqdm(train_loader, desc=f'Training Epoch {epoch+1}'):
            loss = predictor.train_step(images, texts, labels)
            running_loss += loss
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        predictor.model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, texts, labels in test_loader:
                outputs = predictor.predict(images, texts)
                labels = labels.to(DEVICE)
                loss = predictor.criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

        # Вывод потерь
        print(f'Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # Вычисление и вывод метрик после каждой эпохи
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
        micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro')
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
        cm = confusion_matrix(all_labels, all_preds)

        print(f'Epoch {epoch+1} - Accuracy: {accuracy:.4f}')
        for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
            print(f'Class {i}: Precision={p:.4f}, Recall={r:.4f}, F1={f:.4f}')
        print(f'Micro avg: Precision={micro_p:.4f}, Recall={micro_r:.4f}, F1={micro_f1:.4f}')
        print(f'Macro avg: Precision={macro_p:.4f}, Recall={macro_r:.4f}, F1={macro_f1:.4f}')
        print('Confusion Matrix:')
        print(cm)

        # Раннее прекращение
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience = 0
            os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
            torch.save(predictor.model.state_dict(), os.path.join(MODEL_SAVE_PATH, f'best_model_text_{use_text}_image_{use_image}.pth'))
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                print('Early stopping triggered')
                break

    # Сохранение метрик
    metrics = {
        'accuracy': accuracy,
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1_score': f1.tolist(),
        'micro_avg': {'precision': micro_p, 'recall': micro_r, 'f1_score': micro_f1},
        'macro_avg': {'precision': macro_p, 'recall': macro_r, 'f1_score': macro_f1},
        'confusion_matrix': cm.tolist(),
    }
    os.makedirs(RESULTS_SAVE_PATH, exist_ok=True)
    with open(os.path.join(RESULTS_SAVE_PATH, f'metrics_{use_text}_{use_image}.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    # График потерь
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Loss Curve use_text={use_text} use_image={use_image}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_SAVE_PATH, f'loss_curve_{use_text}_{use_image}.png'))
    plt.close()

if __name__ == '__main__':
    os.makedirs('cache', exist_ok=True)

    train_files, train_labels = read_labels(TRAIN_LABELS_FILE, TRAIN_IMAGES_DIR)
    augmented_train_files, augmented_train_labels = read_labels(AUGMENTED_TRAIN_LABELS_FILE, AUGMENTED_TRAIN_IMAGES_DIR)
    test_files, test_labels = read_labels(TEST_LABELS_FILE, TEST_IMAGES_DIR)

    # Пример запуска — обучить на аугментированных данных с текстом и изображениями
    train_and_evaluate(augmented_train_files, augmented_train_labels, test_files, test_labels, use_text=True, use_image=True)