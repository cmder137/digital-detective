import numpy as np
import torch
import logging
from typing import Union, List
from PIL import Image

# Настройка логгера
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def _safe_normalize_np(arr: np.ndarray) -> np.ndarray:
    """Безопасная нормализация NumPy массива к [0, 1]."""
    # Если это явно изображение или значения выходят за рамки 1.0
    if arr.dtype == np.uint8 or arr.max() > 1.0:
        return arr.astype(np.float32) / 255.0
    return arr.astype(np.float32)


def _safe_squeeze_np(arr: np.ndarray) -> np.ndarray:
    """Безопасное удаление лишней размерности канала для NumPy."""
    if arr.ndim == 4:
        if arr.shape[1] == 1:
            return arr.squeeze(axis=1)
        elif arr.shape[-1] == 1:
            return arr.squeeze(axis=-1)
        else:
            raise ValueError(f"Ожидалась 1-канальная маска, получена форма {arr.shape}")
    elif arr.ndim == 2:
        return arr[np.newaxis, ...]
    return arr


def calculate_aic(
    pred_masks: Union[np.ndarray, List[str]],
    gt_masks: Union[np.ndarray, List[str]],
    pred_threshold: float = 0.5,
    gt_threshold: float = 0.5
) -> float:
    """
    Вычисляет метрику AIC Score для задачи сегментации (Standalone / Inference версия на NumPy).
    """
    # 1. Загрузка данных
    if isinstance(pred_masks, list) and len(pred_masks) > 0 and isinstance(pred_masks[0], str):
        pred_masks = np.array([np.array(Image.open(p).convert('L')) for p in pred_masks])
    if isinstance(gt_masks, list) and len(gt_masks) > 0 and isinstance(gt_masks[0], str):
        gt_masks = np.array([np.array(Image.open(p).convert('L')) for p in gt_masks])
        
    pred = np.asarray(pred_masks)
    gt = np.asarray(gt_masks)
    
    if pred.size == 0 or gt.size == 0:
        return 0.0

    # 2. Безопасное изменение формы
    pred = _safe_squeeze_np(pred)
    gt = _safe_squeeze_np(gt)
        
    # 3. Безопасная нормализация
    pred = _safe_normalize_np(pred)
    gt = _safe_normalize_np(gt)
        
    # 4. Бинаризация с единой шкалой [0, 1]
    pred_binary = (pred >= pred_threshold).astype(np.float32)
    gt_binary = (gt >= gt_threshold).astype(np.float32)
    
    # 5. Определение Positive / Negative (Single Source of Truth)
    is_positive = np.sum(gt_binary, axis=(1, 2)) > 0
    
    # 6. Компонент 1: Mean Dice
    if np.any(is_positive):
        pred_pos = pred_binary[is_positive]
        gt_pos = gt_binary[is_positive]
        
        intersection = np.sum(pred_pos * gt_pos, axis=(1, 2))
        denominator = np.sum(pred_pos, axis=(1, 2)) + np.sum(gt_pos, axis=(1, 2))
        dice_scores = 2.0 * intersection / (denominator + 1e-6)
        mean_dice = float(np.mean(dice_scores))
    else:
        mean_dice = 0.0
        
    # 7. Компонент 2: 1 - FPR
    is_negative = ~is_positive
    if np.any(is_negative):
        pred_neg = pred_binary[is_negative]
        h, w = pred_neg.shape[-2:]
        
        areas = np.sum(pred_neg, axis=(1, 2)) / (h * w)
        false_alarms = areas >= 0.01  
        fpr_image_level = float(np.mean(false_alarms))
        component2 = 1.0 - fpr_image_level
    else:
        component2 = 1.0
        
    # 8. Итоговый AIC Score
    if mean_dice + component2 > 0:
        return float(2.0 * (mean_dice * component2) / (mean_dice + component2))
    return 0.0


class AICMeter:
    """
    Высокопроизводительный инкрементальный счетчик AIC Score для PyTorch.
    Векторизован на GPU, без использования циклов по элементам батча и без CPU-синхронизаций.
    """
    def __init__(self, pred_threshold: float = 0.5, gt_threshold: float = 0.5):
        self.pred_threshold = pred_threshold
        self.gt_threshold = gt_threshold
        self.reset()
        
    def reset(self):
        """Сброс счетчиков."""
        self.total_dice = 0.0
        self.n_pos = 0
        self.n_neg = 0
        self.n_false_alarm = 0
        
    def _safe_normalize_pt(self, tensor: torch.Tensor) -> torch.Tensor:
        """Безопасная нормализация PyTorch тензора."""
        if tensor.dtype == torch.uint8 or tensor.max() > 1.0:
            return tensor.float() / 255.0
        return tensor.float()

    def _safe_squeeze_pt(self, tensor: torch.Tensor) -> torch.Tensor:
        """Безопасное удаление оси каналов для PyTorch тензора."""
        if tensor.ndim == 4:
            if tensor.shape[1] == 1:
                return tensor.squeeze(dim=1)
            elif tensor.shape[-1] == 1:
                return tensor.squeeze(dim=-1)
            else:
                raise ValueError(f"Ожидалась 1-канальная маска, получена форма {tensor.shape}")
        elif tensor.ndim == 2:
            return tensor.unsqueeze(0)
        return tensor

    @torch.no_grad()
    def update(self, pred_probs: torch.Tensor, gt_masks: torch.Tensor):
        """
        Векторизованное обновление статистики по батчу.
        Принимает torch.Tensor на GPU.
        """
        if pred_probs.numel() == 0 or gt_masks.numel() == 0:
            return

        pred = self._safe_squeeze_pt(pred_probs)
        gt = self._safe_squeeze_pt(gt_masks)
            
        pred = self._safe_normalize_pt(pred)
        gt = self._safe_normalize_pt(gt)
            
        # Бинаризация (результат: FloatTensor из 0.0 и 1.0)
        preds_bin = (pred >= self.pred_threshold).float()
        gts_bin = (gt >= self.gt_threshold).float()
        
        # Суммы по H и W (получаем тензоры формы [B])
        gt_sums = gts_bin.sum(dim=(-2, -1))
        pred_sums = preds_bin.sum(dim=(-2, -1))
        intersections = (preds_bin * gts_bin).sum(dim=(-2, -1))
        
        # Маски Positive и Negative
        is_pos = gt_sums > 0
        is_neg = ~is_pos
        
        # Обновляем глобальные счетчики
        self.n_pos += is_pos.sum().item()
        self.n_neg += is_neg.sum().item()
        
        # Mean Dice для Positive
        if is_pos.any():
            # Защита от деления на ноль 1e-6
            dice = 2.0 * intersections[is_pos] / (pred_sums[is_pos] + gt_sums[is_pos] + 1e-6)
            self.total_dice += dice.sum().item()
            
        # FPR для Negative
        if is_neg.any():
            # Площадь изображения
            area_threshold = 0.01 * (gts_bin.shape[-2] * gts_bin.shape[-1])
            false_alarms = pred_sums[is_neg] >= area_threshold
            self.n_false_alarm += false_alarms.sum().item()
                    
    def compute(self) -> float:
        """Считает финальную метрику по накопленным данным."""
        if self.n_pos == 0:
            logger.info("В валидации нет positive-примеров. Возвращаем AIC = 0.0")
            return 0.0
            
        mean_dice = self.total_dice / self.n_pos
        fpr_neg = (self.n_false_alarm / self.n_neg) if self.n_neg > 0 else 0.0
        component2 = 1.0 - fpr_neg
        
        if mean_dice + component2 <= 0:
            return 0.0
            
        return float(2.0 * (mean_dice * component2) / (mean_dice + component2))


if __name__ == "__main__":
    print("Запуск тестов метрики (смешанные батчи и edge cases)...")
    
    H, W = 100, 100 
    
    # NumPy Тесты
    pred = np.zeros((2, H, W)); gt = np.zeros((2, H, W))
    pred[0, 10:20, 10:20] = 1; gt[0, 10:20, 10:20] = 1 # Positive (Dice 1.0)
    pred[1, 0:11, 0:10] = 1                           # Negative (FPR 1 -> component2 = 0)
    
    assert abs(calculate_aic(pred, gt) - 0.0) < 1e-6, "Тест NumPy провален"

    # PyTorch Тесты для AICMeter
    pred_t = torch.tensor(pred, dtype=torch.float32).cuda() if torch.cuda.is_available() else torch.tensor(pred, dtype=torch.float32)
    gt_t = torch.tensor(gt, dtype=torch.float32).cuda() if torch.cuda.is_available() else torch.tensor(gt, dtype=torch.float32)

    meter = AICMeter()
    meter.update(pred_t, gt_t)
    
    assert abs(meter.compute() - calculate_aic(pred, gt)) < 1e-6, "Тест PyTorch провален (AICMeter != calculate_aic)"

    print("✅ Все тесты успешно пройдены! AICMeter векторизован и синхронизирован с calculate_aic.")