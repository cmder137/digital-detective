"""
Аугментации для задачи бинарной сегментации подделок.

- Train: геометрические + цветовые + имитация JPEG-артефактов.
- Val/Test: только ресайз + нормализация (никаких аугментаций!).
- Нормализация — ImageNet mean/std, обязательна для предобученных энкодеров.
"""
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ImageNet статистика — совместима с предобученными энкодерами SMP
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _image_compression(p: float = 0.3):
    """
    Совместимость с разными версиями albumentations.
    - >= 1.4.0: quality_range=(70, 100)
    - 1.3.x:    quality_lower=70, quality_upper=100
    """
    try:
        return A.ImageCompression(quality_range=(70, 100), p=p)
    except (TypeError, ValueError):
        return A.ImageCompression(quality_lower=70, quality_upper=100, p=p)


def get_train_transforms(img_size: int) -> A.Compose:
    """
    Аугментации для обучающей выборки.

    Все геометрические преобразования применяются СИНХРОННО к image и mask.
    Цветовые (яркость, контраст, hue) — только к image.
    """
    return A.Compose([
        # --- Геометрические ---
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.Affine(
            translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5,
        ),

        # --- Ресайз ---
        A.Resize(img_size, img_size),

        # --- Цветовые (имитация разных условий съёмки) ---
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3,
        ),

        # --- Имитация артефактов сжатия (важно для борьбы с FPR) ---
        _image_compression(p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),

        # --- Нормализация под ImageNet ---
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int) -> A.Compose:
    """
    Для валидации и теста — только ресайз и нормализация.
    Никаких аугментаций: мы хотим измерять реальное качество модели.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])