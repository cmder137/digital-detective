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
    print("Запуск тестов лосса (проверка крайних случаев)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Тестируем на устройстве: {device}")
    
    criterion = DiceBCELoss().to(device)
    
    # Создаем dummy-target (N, 1, H, W)
    targets = torch.zeros((2, 1, 64, 64), device=device).float()
    targets[:, :, 10:20, 10:20] = 1.0 # Рисуем квадратик
    
    # 1. Идеальные предсказания (очень большие логиты там где 1, и очень маленькие там где 0)
    perfect_logits = torch.where(targets == 1.0, torch.tensor(15.0), torch.tensor(-15.0)).to(device)
    loss_perfect = criterion(perfect_logits, targets)
    print(f"Лосс при идеальном предсказании (должен быть ~0): {loss_perfect.item():.6f}")
    
    # 2. Полностью неверные (инвертированные) предсказания
    inverted_logits = torch.where(targets == 1.0, torch.tensor(-15.0), torch.tensor(15.0)).to(device)
    loss_inverted = criterion(inverted_logits, targets)
    print(f"Лосс при инвертированном предсказании (должен быть большим): {loss_inverted.item():.6f}")