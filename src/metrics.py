import numpy as np
import torch
import logging
from typing import Union, List
from PIL import Image

# Настройка логгера
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# NUMPY-ВЕРСИЯ (для инференса и unit-тестов, вызывается на CPU)
# ============================================================

def _safe_normalize_np(arr: np.ndarray) -> np.ndarray:
    """Безопасная нормализация NumPy массива к [0, 1].
    
    ПРЕДУПРЕЖДЕНИЕ: uint8-маски со значениями {0, 1} будут некорректно
    масштабированы в [0, 0.0039]. Поддерживаемый контракт:
      - float в [0, 1]
      - uint8 {0, 255} (бинарные PNG маски)
    """
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
    if isinstance(pred_masks, list) and len(pred_masks) > 0 and isinstance(pred_masks[0], str):
        pred_masks = np.array([np.array(Image.open(p).convert('L')) for p in pred_masks])
    if isinstance(gt_masks, list) and len(gt_masks) > 0 and isinstance(gt_masks[0], str):
        gt_masks = np.array([np.array(Image.open(p).convert('L')) for p in gt_masks])

    pred = np.asarray(pred_masks)
    gt = np.asarray(gt_masks)

    if pred.size == 0 or gt.size == 0:
        return 0.0

    pred = _safe_squeeze_np(pred)
    gt = _safe_squeeze_np(gt)

    pred = _safe_normalize_np(pred)
    gt = _safe_normalize_np(gt)

    pred_binary = (pred >= pred_threshold).astype(np.float32)
    gt_binary = (gt >= gt_threshold).astype(np.float32)

    is_positive = np.sum(gt_binary, axis=(1, 2)) > 0

    if np.any(is_positive):
        pred_pos = pred_binary[is_positive]
        gt_pos = gt_binary[is_positive]
        intersection = np.sum(pred_pos * gt_pos, axis=(1, 2))
        denominator = np.sum(pred_pos, axis=(1, 2)) + np.sum(gt_pos, axis=(1, 2))
        dice_scores = 2.0 * intersection / (denominator + 1e-6)
        mean_dice = float(np.mean(dice_scores))
    else:
        mean_dice = 0.0

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

    if mean_dice + component2 > 0:
        return float(2.0 * (mean_dice * component2) / (mean_dice + component2))
    return 0.0


# ============================================================
# PYTORCH-ВЕРСИЯ (для цикла обучения)
# ============================================================

class AICMeter:
    """
    Высокопроизводительный инкрементальный счётчик AIC Score для PyTorch.
    
    Полностью zero-sync в update(): 
      - нет CUDA-синхронизаций (нет .max(), .item(), if tensor.any());
      - аккумуляция в GPU-скаляры;
      - безветочные маскированные суммы.
    Единственные .item() — в compute(), один раз за эпоху.
    
    КОНТРАКТ ВХОДА:
      - pred_probs: вероятности после sigmoid, значения в [0, 1].
      - gt_masks: float в [0, 1] ИЛИ uint8 {0, 255}.
                    (uint8 {0, 1} НЕ поддерживается — см. _safe_normalize_pt)
    """
    def __init__(self, pred_threshold: float = 0.5, gt_threshold: float = 0.5,
                 device: torch.device = None):
        self.pred_threshold = pred_threshold
        self.gt_threshold = gt_threshold
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reset()

    def reset(self):
        """Сброс счётчиков в GPU-скаляры."""
        self.total_dice = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.n_pos = torch.tensor(0, dtype=torch.long, device=self.device)
        self.n_neg = torch.tensor(0, dtype=torch.long, device=self.device)
        self.n_false_alarm = torch.tensor(0, dtype=torch.long, device=self.device)

    @staticmethod
    def _safe_squeeze_pt(tensor: torch.Tensor) -> torch.Tensor:
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

    @staticmethod
    def _safe_normalize_pt(tensor: torch.Tensor) -> torch.Tensor:
        """Нормализация без .max() (без CUDA-синхронизации).
        
        Работает по dtype:
          - uint8 → делится на 255 (контракт: значения {0, 255})
          - иначе → просто приводится к float32
        """
        if tensor.dtype == torch.uint8:
            return tensor.float() / 255.0
        if tensor.dtype != torch.float32:
            return tensor.float()
        return tensor

    @torch.no_grad()
    def update(self, pred_probs: torch.Tensor, gt_masks: torch.Tensor):
        """Векторизованное обновление статистики по батчу. Zero-sync."""
        if pred_probs.numel() == 0 or gt_masks.numel() == 0:
            return

        pred = self._safe_squeeze_pt(pred_probs)
        gt = self._safe_squeeze_pt(gt_masks)

        pred = self._safe_normalize_pt(pred)
        gt = self._safe_normalize_pt(gt)

        preds_bin = (pred >= self.pred_threshold).float()
        gts_bin = (gt >= self.gt_threshold).float()

        gt_sums = gts_bin.sum(dim=(-2, -1))
        pred_sums = preds_bin.sum(dim=(-2, -1))
        intersections = (preds_bin * gts_bin).sum(dim=(-2, -1))

        is_pos = gt_sums > 0
        is_neg = ~is_pos

        # --- Всё на GPU, без веток и без .item() ---
        self.n_pos = self.n_pos + is_pos.sum()
        self.n_neg = self.n_neg + is_neg.sum()

        dice_per_img = 2.0 * intersections / (pred_sums + gt_sums + 1e-6)
        # для negative картинок is_pos=False (0) → вклад обнуляется
        self.total_dice = self.total_dice + (dice_per_img * is_pos.float()).sum()

        area_threshold = 0.01 * (gts_bin.shape[-2] * gts_bin.shape[-1])
        false_alarms = (pred_sums >= area_threshold) & is_neg
        self.n_false_alarm = self.n_false_alarm + false_alarms.sum()

    def compute(self) -> float:
        """Финальная метрика. Единственные синхронизации — здесь, раз за эпоху."""
        n_pos = int(self.n_pos.item())
        if n_pos == 0:
            logger.info("В валидации нет positive-примеров. Возвращаем AIC = 0.0")
            return 0.0

        mean_dice = float(self.total_dice.item()) / n_pos
        n_neg = int(self.n_neg.item())
        fpr_neg = float(self.n_false_alarm.item()) / n_neg if n_neg > 0 else 0.0
        component2 = 1.0 - fpr_neg

        if mean_dice + component2 <= 0:
            return 0.0

        return float(2.0 * (mean_dice * component2) / (mean_dice + component2))


if __name__ == "__main__":
    print("Запуск тестов метрики...")
    H, W = 100, 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Тест 1: вырожденный случай (Dice=1, FPR=1 → AIC=0)
    pred = np.zeros((2, H, W))
    gt = np.zeros((2, H, W))
    pred[0, 10:20, 10:20] = 1
    gt[0, 10:20, 10:20] = 1
    pred[1, 0:11, 0:10] = 1

    aic_np = calculate_aic(pred, gt)
    assert abs(aic_np - 0.0) < 1e-6, f"Тест 1 NumPy провален: {aic_np}"

    pred_t = torch.tensor(pred, dtype=torch.float32, device=device)
    gt_t = torch.tensor(gt, dtype=torch.float32, device=device)
    meter = AICMeter(device=device)
    meter.update(pred_t, gt_t)
    aic_pt = meter.compute()
    assert abs(aic_pt - aic_np) < 1e-6, f"Тест 1 PyTorch провален: {aic_pt} != {aic_np}"

    # Тест 2: средний случай (проверяет формулу, не только ветки с нулями)
    #   Positive 1: идеальный Dice=1.0
    #   Positive 2: идеальный Dice=1.0
    #   Negative 1: false alarm (15x15 = 2.25% ≥ 1%)
    #   Negative 2: без false alarm (5x5 = 0.25% < 1%)
    #   Ожидание: mean_dice=1.0, fpr=0.5, component2=0.5
    #   AIC = 2*1.0*0.5 / (1.0+0.5) = 1.0/1.5 = 0.666...
    pred2 = np.zeros((4, H, W))
    gt2 = np.zeros((4, H, W))
    pred2[0, 10:40, 10:40] = 1; gt2[0, 10:40, 10:40] = 1
    pred2[1, 20:50, 20:50] = 1; gt2[1, 20:50, 20:50] = 1
    pred2[2, 10:25, 10:25] = 1  # 15x15 = 225/10000 = 2.25% → false alarm
    pred2[3, 10:15, 10:15] = 1  # 5x5 = 25/10000 = 0.25% → ok

    aic_np2 = calculate_aic(pred2, gt2)
    assert abs(aic_np2 - 2.0/3.0) < 1e-5, f"Тест 2 NumPy провален: {aic_np2}"

    pred2_t = torch.tensor(pred2, dtype=torch.float32, device=device)
    gt2_t = torch.tensor(gt2, dtype=torch.float32, device=device)
    meter2 = AICMeter(device=device)
    meter2.update(pred2_t, gt2_t)
    aic_pt2 = meter2.compute()
    assert abs(aic_pt2 - aic_np2) < 1e-5, f"Тест 2 PyTorch провален: {aic_pt2} != {aic_np2}"

    print(f"✅ Все тесты пройдены. AIC edge-case = {aic_pt:.4f}, AIC middle-case = {aic_pt2:.4f}")