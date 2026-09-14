"""
Цикл обучения для AI Challenge 2026 — Digital Detective.

Что делает:
  1. Прогоняет модель через изображения с использованием AMP (Mixed Precision).
  2. Векторизованно (на GPU) считает метрику AIC на валидации без переноса на CPU.
  3. Безопасно сохраняет лучшие веса (best.pth), если AIC улучшился.
  4. Сохраняет чекпоинт (last.pth) с защитой от поврежденных файлов.
  5. Логирует метрики в results/metrics.csv.

Запуск:
    python -m src.train
"""
import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from src.losses import DiceBCELoss

# ============================================================
# ВОСПРОИЗВОДИМОСТЬ
# ============================================================
def set_seed(seed: int = 42):
    """Фиксируем seed для полной воспроизводимости."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, 'cuda'):
        torch.backends.cuda.matmul.allow_tf32 = False


def seed_worker(worker_id):
    """Фиксируем сиды в каждом воркере DataLoader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# ЛОГИРОВАНИЕ
# ============================================================
def log_metrics(epoch, train_loss, val_loss, val_aic, lr, log_path='results/metrics.csv'):
    """Безопасное логирование метрик в CSV."""
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    file_exists = os.path.exists(log_path)
    with open(log_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss', 'val_aic', 'lr'])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'epoch': epoch,
            'train_loss': f"{train_loss:.6f}",
            'val_loss': f"{val_loss:.6f}",
            'val_aic': f"{val_aic:.6f}",
            'lr': f"{lr:.2e}"
        })


# ============================================================
# ОБУЧЕНИЕ: ОДНА ЭПОХА (С AMP)
# ============================================================
def train_epoch(model, loader, optimizer, criterion, device, scaler):
    """Одна эпоха обучения с использованием смешанной точности (AMP)."""
    if len(loader) == 0:
        raise ValueError("train_loader пуст. Проверьте данные и фильтр файлов.")

    model.train()
    epoch_loss = 0.0
    n_valid_batches = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device).float()

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
            logits = model(images)
            loss = criterion(logits, masks)

        if not torch.isfinite(loss):
            print(f"⚠️ WARNING: loss = {loss.item():.6f} (non-finite), пропускаем батч")
            continue

        scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()

        loss_val = loss.item()
        epoch_loss += loss_val
        n_valid_batches += 1
        pbar.set_postfix({"loss": f"{loss_val:.4f}"})

    return epoch_loss / max(n_valid_batches, 1)


# ============================================================
# ВАЛИДАЦИЯ: ОДНА ЭПОХА (Векторизовано на GPU)
# ============================================================
@torch.inference_mode()
def eval_epoch(model, loader, criterion, device, threshold: float = 0.5):
    """Валидация. Метрика считается тензорно прямо на GPU, без for-циклов."""
    if len(loader) == 0:
        return 0.0, 0.0

    model.eval()
    epoch_loss = 0.0
    total_dice, n_pos, n_neg, n_false_alarm = 0.0, 0, 0, 0

    pbar = tqdm(loader, desc="Validating", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device).float()

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
            logits = model(images)
            loss = criterion(logits, masks)

        epoch_loss += loss.item()

        preds = (torch.sigmoid(logits) >= threshold).float()

        gt_sums = masks.sum(dim=(1, 2, 3))
        pred_sums = preds.sum(dim=(1, 2, 3))
        intersections = (preds * masks).sum(dim=(1, 2, 3))

        is_pos = gt_sums > 0
        is_neg = ~is_pos

        n_pos += is_pos.sum().item()
        n_neg += is_neg.sum().item()

        if is_pos.any():
            dice = 2.0 * intersections[is_pos] / (pred_sums[is_pos] + gt_sums[is_pos] + 1e-6)
            total_dice += dice.sum().item()

        if is_neg.any():
            area_threshold = 0.01 * (masks.shape[-2] * masks.shape[-1])
            false_alarms = pred_sums[is_neg] >= area_threshold
            n_false_alarm += false_alarms.sum().item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    if n_pos == 0:
        print("⚠️ WARNING: В валидации нет positive-примеров.")
        return epoch_loss / len(loader), 0.0

    mean_dice = total_dice / n_pos
    fpr_neg = (n_false_alarm / n_neg) if n_neg > 0 else 0.0
    component2 = 1.0 - fpr_neg

    aic_score = (
        2.0 * (mean_dice * component2) / (mean_dice + component2)
        if (mean_dice + component2) > 0
        else 0.0
    )
    return epoch_loss / len(loader), float(aic_score)


# ============================================================
# ВОЗОБНОВЛЕНИЕ ОБУЧЕНИЯ
# ============================================================
def resume_training(model, optimizer, scheduler, save_dir='checkpoints'):
    """Надежная загрузка чекпоинта last.pth."""
    last_path = os.path.join(save_dir, 'last.pth')
    if not os.path.exists(last_path):
        return 1, -1.0

    try:
        checkpoint = torch.load(last_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_aic = checkpoint.get('best_aic', -1.0)
        print(f"📂 Возобновление с эпохи {start_epoch}, best_aic = {best_aic:.4f}")
        return start_epoch, best_aic

    except Exception as e:
        print(f"⚠️ WARNING: Не удалось загрузить {last_path} ({e}). Начинаем с нуля.")
        return 1, -1.0


# ============================================================
# ГЛАВНЫЙ ЦИКЛ ОБУЧЕНИЯ
# ============================================================
def train(model, train_loader, val_loader, epochs=10, lr=1e-3,
          device='cuda', save_dir='checkpoints', seed=42, threshold=0.5):
    """Главный пайплайн обучения."""
    device = torch.device(device) if isinstance(device, str) else device
    set_seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    
    results_dir = os.path.join(os.path.dirname(os.path.abspath(save_dir)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, 'metrics.csv')

    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    start_epoch, best_aic = resume_training(model, optimizer, scheduler, save_dir)

    for epoch in range(start_epoch, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        current_lr = optimizer.param_groups[0]['lr']

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_aic = eval_epoch(model, val_loader, criterion, device, threshold=threshold)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AIC: {val_aic:.4f} | LR: {current_lr:.2e}")
        log_metrics(epoch, train_loss, val_loss, val_aic, current_lr, log_path=metrics_path)

        if np.isnan(val_aic) or np.isinf(val_aic):
            print(f"⚠️ WARNING: val_aic = {val_aic}, пропускаем обновление скедулера и чекпоинтов!")
            continue

        scheduler.step(val_aic)

        if val_aic > best_aic:
            best_aic = val_aic
            best_path = os.path.join(save_dir, 'best.pth')
            torch.save(model.state_dict(), best_path)
            print(f"🌟 Новый лучший AIC: {best_aic:.4f}! Модель сохранена.")

        last_path = os.path.join(save_dir, 'last.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_aic': best_aic,
        }, last_path)


# ============================================================
# ТЕСТОВЫЙ БЛОК (Запуск без реальных данных)
# ============================================================
if __name__ == "__main__":
    print("Запуск тестового прогона цикла обучения...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")

    class DummyModel(nn.Module):
        def forward(self, x):
            return torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)

    model = DummyModel().to(device)
    dummy_images = torch.randn(4, 3, 64, 64)
    dummy_masks = torch.randint(0, 2, (4, 1, 64, 64))

    dummy_dataset = TensorDataset(dummy_images, dummy_masks)
    
    g = torch.Generator()
    g.manual_seed(42)
    
    dummy_loader = DataLoader(
        dummy_dataset,
        batch_size=2,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    train(model, dummy_loader, dummy_loader, epochs=2, device=device)