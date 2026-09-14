import numpy as np
from typing import Union, List, Optional
from PIL import Image

def calculate_aic(
    pred_masks: Union[np.ndarray, List[str]],
    gt_masks: Union[np.ndarray, List[str]],
    is_positive: Optional[Union[np.ndarray, List[bool]]] = None,
    pred_threshold: float = 0.5,
    gt_threshold: float = 127.5
) -> float:
    """
    Вычисляет метрику AIC Score для задачи сегментации изменений.
    """
    # 1. Загрузка данных
    if isinstance(pred_masks, list) and len(pred_masks) > 0 and isinstance(pred_masks[0], str):
        pred_masks = np.array([np.array(Image.open(p).convert('L')) for p in pred_masks])
    if isinstance(gt_masks, list) and len(gt_masks) > 0 and isinstance(gt_masks[0], str):
        gt_masks = np.array([np.array(Image.open(p).convert('L')) for p in gt_masks])
        
    pred = np.array(pred_masks).astype(np.float32)
    gt = np.array(gt_masks).astype(np.float32)
    
    # Приведение к (N, H, W)
    if pred.ndim == 4:
        pred = pred.squeeze(axis=1)
    if gt.ndim == 4:
        gt = gt.squeeze(axis=1)
        
    # Нормализация к [0, 1], если значения в [0, 255]
    if pred.max() > 1.0:
        pred = pred / 255.0
    if gt.max() > 1.0:
        gt = gt / 255.0
        
    # Бинаризация с раздельными порогами
    # gt_threshold делим на 255, т.к. мы уже нормализовали gt к [0, 1]
    pred_binary = (pred >= pred_threshold).astype(np.float32)
    gt_binary = (gt >= gt_threshold / 255.0).astype(np.float32)
    
    # Определение Positive / Negative примеров
    if is_positive is None:
        is_positive = np.sum(gt_binary, axis=(1, 2)) > 0
    else:
        is_positive = np.array(is_positive)
        
    # Компонент 1: Mean Dice для Positive-примеров
    positive_mask = is_positive
    if np.sum(positive_mask) > 0:
        pred_pos = pred_binary[positive_mask]
        gt_pos = gt_binary[positive_mask]
        
        intersection = np.sum(pred_pos * gt_pos, axis=(1, 2))
        denominator = np.sum(pred_pos, axis=(1, 2)) + np.sum(gt_pos, axis=(1, 2))
        dice_scores = np.where(denominator > 0, 2.0 * intersection / denominator, 1.0)
        mean_dice = float(np.mean(dice_scores))
    else:
        mean_dice = 1.0
        
    # Компонент 2: 1 - FPR для Negative-примеров (с порогом площади 1%)
    negative_mask = ~is_positive
    if np.sum(negative_mask) > 0:
        pred_neg = pred_binary[negative_mask]
        h, w = pred_neg.shape[1], pred_neg.shape[2]
        
        # Доля белых пикселей от площади картинки
        areas = np.sum(pred_neg, axis=(1, 2)) / (h * w)
        
        # Изображение - ложная тревога, только если площадь >= 1% (0.01)
        false_alarms = areas >= 0.01  
        fpr_image_level = float(np.mean(false_alarms))
        component2 = 1.0 - fpr_image_level
    else:
        component2 = 1.0
        
    # Итоговый AIC Score (гармоническое среднее)
    if mean_dice + component2 > 0:
        aic_score = 2.0 * (mean_dice * component2) / (mean_dice + component2)
    else:
        aic_score = 0.0
        
    return float(aic_score)

if __name__ == "__main__":
    print("Запуск тестов метрики (edge cases)...")
    
    H, W = 100, 100 # Для удобства подсчета: площадь = 10,000, 1% = 100 пикселей
    
    # Тест 1: Negative с 1 белым пикселем -> Не ложная тревога (площадь < 1%)
    pred_1 = np.zeros((1, H, W))
    pred_1[0, 0, 0] = 1 # всего 1 пиксель
    gt_1 = np.zeros((1, H, W))
    score_1 = calculate_aic(pred_1, gt_1)
    print(f"Тест 1 (Negative, 1 шумный пиксель): {score_1} (Ожидаем: 1.0)")
    
    # Тест 2: Negative с >1% белых пикселей -> Ложная тревога
    pred_2 = np.zeros((1, H, W))
    pred_2[0, 0:11, 0:10] = 1 # 110 пикселей (1.1%)
    gt_2 = np.zeros((1, H, W))
    score_2 = calculate_aic(pred_2, gt_2)
    print(f"Тест 2 (Negative, >1% шума): {score_2} (Ожидаем: 0.0)")
    
    # Тест 3: Positive с пустым предсказанием -> Dice 0 -> AIC 0
    pred_3 = np.zeros((1, H, W))
    gt_3 = np.zeros((1, H, W))
    gt_3[0, 50:60, 50:60] = 1 # Есть подделка в GT
    score_3 = calculate_aic(pred_3, gt_3)
    print(f"Тест 3 (Positive, пустое предсказание): {score_3} (Ожидаем: 0.0)")