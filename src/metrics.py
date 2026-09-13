import numpy as np
import torch

from typing import Union, List, Optional
from PIL import Image
import os

def calculate_aic(
    pred_masks: Union[np.ndarray, List[str]],
    gt_masks: Union[np.ndarray, List[str]],
    is_positive: Optional[Union[np.ndarray, List[bool]]] = None,
    threshold: float = 127.5
) -> float:
    """
    Вычисляет кастомную метрику AIC Score для задачи сегментации изменений.
    
    Метрика является гармоническим средним между:
    1. Средним Dice коэффициентом для Positive-примеров (где в GT есть изменения).
    2. Долей корректно предсказанных Negative-примеров (1 - FPR на уровне изображения), 
       где изображение считается ложной тревогой, если площадь предсказанной маски > 0.
    
    Args:
        pred_masks: Массив numpy формы (N, H, W) или (N, 1, H, W) с предсказаниями модели 
                    (логиты, вероятности или бинарные значения 0/255), 
                    ИЛИ список путей к PNG-файлам предсказаний.
        gt_masks: Массив numpy формы (N, H, W) или (N, 1, H, W) с GT-масками 
                  (значения 0 или 255), ИЛИ список путей к PNG-файлам GT.
        is_positive: (Опционально) Массив или список булевых значений. Если None, 
                     определяется автоматически: True, если сумма пикселей в GT > 0.
        threshold: Порог бинаризации предсказаний. По умолчанию 127.5 (корректно работает 
                   как для вероятностей [0, 1] с порогом 0.5, так и для масок [0, 255]).
                     
    Returns:
        float: Значение метрики AIC Score от 0.0 до 1.0.
    """
    
    # 1. Загрузка данных, если переданы списки путей к файлам
    if isinstance(pred_masks, list) and len(pred_masks) > 0 and isinstance(pred_masks[0], str):
        pred_masks = np.array([np.array(Image.open(p).convert('L')) for p in pred_masks])
    if isinstance(gt_masks, list) and len(gt_masks) > 0 and isinstance(gt_masks[0], str):
        gt_masks = np.array([np.array(Image.open(p).convert('L')) for p in gt_masks])
        
    # 2. Приведение к единому формату (N, H, W) и бинаризация
    pred_binary = (np.array(pred_masks) > threshold).astype(np.float32)
    gt_binary = (np.array(gt_masks) > threshold).astype(np.float32)
    
    # Убираем канальное измерение, если модель вернула (N, 1, H, W)
    if pred_binary.ndim == 4:
        pred_binary = pred_binary.squeeze(axis=1)
        gt_binary = gt_binary.squeeze(axis=1)
        
    # 3. Определение Positive / Negative примеров
    if is_positive is None:
        # Изображение считается Positive, если в GT есть хотя бы один измененный пиксель
        is_positive = np.sum(gt_binary, axis=(1, 2)) > 0
    else:
        is_positive = np.array(is_positive)
        
    # ==============================================================================
    # Компонент 1: Mean Dice для Positive-примеров
    # ==============================================================================
    positive_mask = is_positive
    n_positive = np.sum(positive_mask)
    
    if n_positive > 0:
        pred_pos = pred_binary[positive_mask]
        gt_pos = gt_binary[positive_mask]
        
        intersection = np.sum(pred_pos * gt_pos, axis=(1, 2))
        pred_sum = np.sum(pred_pos, axis=(1, 2))
        gt_sum = np.sum(gt_pos, axis=(1, 2))
        
        denominator = pred_sum + gt_sum
        # Защита от деления на ноль: если обе маски пустые, считаем Dice = 1.0
        dice_scores = np.where(denominator > 0, 2.0 * intersection / denominator, 1.0)
        mean_dice = float(np.mean(dice_scores))
    else:
        mean_dice = 1.0  # Если нет Positive-примеров, считаем компонент идеальным
        
    # ==============================================================================
    # Компонент 2: 1 - FPR для Negative-примеров (на уровне изображения)
    # ==============================================================================
    negative_mask = ~is_positive
    n_negative = np.sum(negative_mask)
    
    if n_negative > 0:
        pred_neg = pred_binary[negative_mask]
        
        # Изображение является ложной тревогой, если площадь предсказанной маски > 0
        false_alarms = np.sum(pred_neg, axis=(1, 2)) > 0
        fpr_image_level = float(np.mean(false_alarms))
        component2 = 1.0 - fpr_image_level
    else:
        component2 = 1.0  # Если нет Negative-примеров, считаем компонент идеальным
        
    # ==============================================================================
    # Компонент 3: Итоговый AIC Score (гармоническое среднее)
    # ==============================================================================
    # Гармоническое среднее штрафует решение, если провалена хотя бы одна из составляющих
    if mean_dice + component2 > 0:
        aic_score = 2.0 * (mean_dice * component2) / (mean_dice + component2)
    else:
        aic_score = 0.0
        
    return float(aic_score)

# Блок проверки (запустится только если мы напрямую вызываем этот файл)
if __name__ == "__main__":
    print("Запуск теста метрики на синтетических данных...")
    
    # Имитируем предсказания модели (вероятности от 0 до 1)
    preds = np.random.rand(4, 256, 256)
    
    # Имитируем реальные маски (целые числа 0 или 1)
    targets = np.random.randint(0, 2, (4, 256, 256))
    
    # ВАЖНО: передаем порог 0.5, так как у нас вероятности от 0 до 1
    score = calculate_aic(preds, targets, threshold=0.5)
    
    print(f"Тестовый AIC Score: {score}")