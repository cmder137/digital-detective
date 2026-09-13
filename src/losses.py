import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class DiceBCELoss(nn.Module):
    def __init__(self, weight_dice: float = 0.5, weight_bce: float = 0.5):
        """
        Комбинированный лосс для задачи бинарной сегментации.
        Берет взвешенную сумму BCEWithLogitsLoss и DiceLoss.
        """
        super().__init__()
        # BCEWithLogitsLoss уже включает внутри себя Sigmoid,
        # поэтому на вход подаем "сырые" выходы модели (логиты)
        self.bce = nn.BCEWithLogitsLoss()
        
        # from_logits=True значит, что DiceLoss тоже сам применит Sigmoid
        self.dice = smp.losses.DiceLoss(
            mode=smp.losses.BINARY_MODE, 
            from_logits=True
        )
        
        self.w_dice = weight_dice
        self.w_bce = weight_bce

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        y_pred: тензор предсказаний модели (N, 1, H, W) - логиты (без сигмоиды)
        y_true: тензор реальных масок (N, 1, H, W) - значения 0 или 1
        """
        # Считаем оба лосса
        bce_loss = self.bce(y_pred, y_true)
        dice_loss = self.dice(y_pred, y_true)
        
        # Возвращаем их сумму с заданными весами
        return self.w_bce * bce_loss + self.w_dice * dice_loss

# Простенький тест
if __name__ == "__main__":
    print("Проверка комбинированного лосса...")
    
    criterion = DiceBCELoss()
    
    # Имитация: батч из 4 картинок 1x256x256
    # logits (сырые предсказания модели, могут быть отрицательными)
    preds = torch.randn(4, 1, 256, 256)
    
    # targets (0 или 1), обязательно тип float для BCE
    targets = torch.randint(0, 2, (4, 1, 256, 256)).float()
    
    loss_value = criterion(preds, targets)
    print(f"Значение лосса: {loss_value.item():.4f}")