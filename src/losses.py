import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class DiceBCELoss(nn.Module):
    def __init__(self, weight_dice: float = 0.5, weight_bce: float = 0.5):
        """
        Комбинированный лосс для задачи бинарной сегментации.
        weight_dice и weight_bce позволяют балансировать штрафы:
        - BCE помогает штрафовать каждый пиксель по отдельности.
        - Dice помогает собирать правильную форму маски.
        """
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        
        # Используем mode='binary' по рекомендации тимлида
        self.dice = smp.losses.DiceLoss(
            mode='binary', 
            from_logits=True
        )
        
        self.w_dice = weight_dice
        self.w_bce = weight_bce

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        y_pred: тензор предсказаний (N, 1, H, W) - сырые логиты
        y_true: тензор масок (N, 1, H, W) - значения 0 или 1
        """
        # Защита от неправильных размерностей (как просил лид)
        assert y_pred.ndim == 4 and y_pred.shape[1] == 1, f"y_pred должен иметь форму (N, 1, H, W), а получено {y_pred.shape}"
        assert y_true.ndim == 4 and y_true.shape[1] == 1, f"y_true должен иметь форму (N, 1, H, W), а получено {y_true.shape}"
        
        bce_loss = self.bce(y_pred, y_true)
        dice_loss = self.dice(y_pred, y_true)
        
        return self.w_bce * bce_loss + self.w_dice * dice_loss

if __name__ == "__main__":
    print("Проверка DiceBCELoss...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Тестируем на устройстве: {device}")
    
    criterion = DiceBCELoss().to(device)
    targets = torch.randint(0, 2, (4, 1, 256, 256), device=device).float()

    # 1. Случайные логиты
    preds_random = torch.randn(4, 1, 256, 256, device=device)
    loss_random = criterion(preds_random, targets)
    print(f"Random logits:  {loss_random.item():.4f}")

    # 2. Идеальные логиты (+15 для 1, -15 для 0)
    preds_perfect = (targets * 2 - 1) * 15
    loss_perfect = criterion(preds_perfect, targets)
    print(f"Perfect logits: {loss_perfect.item():.4f}  (должно быть ~0)")
    assert loss_perfect.item() < 0.01, "Лосс на идеальных предсказаниях слишком большой"

    # 3. Инвертированные логиты
    loss_wrong = criterion(-preds_perfect, targets)
    print(f"Wrong logits:   {loss_wrong.item():.4f}  (должно быть >> 0)")
    assert loss_wrong.item() > 1.0, "Лосс на неправильных предсказаниях слишком маленький"

    print("✅ Все тесты пройдены")