import os
import random
import shutil
from PIL import Image
import albumentations as A
import cv2
from config import TRAIN_IMAGES_DIR, TRAIN_LABELS_FILE, AUGMENTED_TRAIN_IMAGES_DIR, AUGMENTED_TRAIN_LABELS_FILE

os.makedirs(AUGMENTED_TRAIN_IMAGES_DIR, exist_ok=True)

# Читаем метки оригинального датасета
with open(TRAIN_LABELS_FILE, 'r') as f:
    lines = [line.strip() for line in f if line.strip()]

# Формируем словарь filename -> label
filename_to_label = {}
for line in lines:
    *filename_parts, label = line.rsplit(' ', maxsplit=1)
    filename = ' '.join(filename_parts)
    filename_to_label[filename] = label

# Список аугментаций — каждая будет применена по отдельности
augmentations_list = [
    A.Rotate(limit=15, border_mode=cv2.BORDER_REPLICATE, p=1.0),                    # Поворот ±15°
    A.RandomScale(scale_limit=0.15, p=1.0),                                        # Масштаб ±15%
    A.ToGray(p=1.0),                                                               # Черно-белое
    A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.0), contrast_limit=0, p=1.0)  # Понижение яркости
]

def save_image(np_img, path):
    # Albumentations возвращает numpy в формате HWC BGR
    # Конвертируем обратно в RGB для сохранения через PIL
    img_rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    pil_img.save(path)

augmented_lines = []

print(f"Создаём аугментированный датасет в {AUGMENTED_TRAIN_IMAGES_DIR} ...")

for filename, label in filename_to_label.items():
    orig_path = os.path.join(TRAIN_IMAGES_DIR, filename)
    
    # Копируем оригинальное изображение без изменений
    new_orig_path = os.path.join(AUGMENTED_TRAIN_IMAGES_DIR, filename)
    shutil.copy2(orig_path, new_orig_path)
    augmented_lines.append(f"{filename} {label}")

    # Загружаем изображение для аугментации
    image = cv2.imread(orig_path)
    if image is None:
        print(f"Warning: не удалось загрузить {orig_path}")
        continue

    # Выбираем 2 разные аугментации из списка
    chosen_augs = random.sample(augmentations_list, 2)

    for i, aug in enumerate(chosen_augs, 1):
        augmented_img = aug(image=image)['image']

        base, ext = os.path.splitext(filename)
        aug_filename = f"{base}_aug{i}{ext}"
        aug_path = os.path.join(AUGMENTED_TRAIN_IMAGES_DIR, aug_filename)

        save_image(augmented_img, aug_path)
        augmented_lines.append(f"{aug_filename} {label}")

# Записываем новый файл с метками для аугментированного датасета
with open(AUGMENTED_TRAIN_LABELS_FILE, 'w') as f:
    f.write('\n'.join(augmented_lines))

print(f"Аугментированный датасет создан: {len(augmented_lines)} записей")
print(f"Изображения сохранены в: {AUGMENTED_TRAIN_IMAGES_DIR}")
print(f"Метки сохранены в: {AUGMENTED_TRAIN_LABELS_FILE}")
