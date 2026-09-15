"""
Цикл обучения для AI Challenge 2026 — Digital Detective.
"""
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from src.losses import DiceBCELoss
from src.metrics import AICMeter
from src.utils import set_seed, seed_worker, log_metrics

# Подхватываем настроенный логгер из run_train.py
logger = logging.getLogger('digital_detective')

# ============================================================
# ОБУЧЕНИЕ: ОДНА ЭПОХА (С AMP)
# ============================================================
def train_epoch(model, loader, optimizer, criterion, device, scaler):
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
            logger.warning(f"loss = {loss.item():.6f} (non-finite), пропускаем батч")
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
# ВАЛИДАЦИЯ: ОДНА ЭПОХА
# ============================================================
@torch.inference_mode()
def eval_epoch(model, loader, criterion, device, threshold: float = 0.5):
    if len(loader) == 0:
        return 0.0, 0.0

    model.eval()
    epoch_loss = 0.0
    meter = AICMeter(pred_threshold=threshold, gt_threshold=0.5)

    pbar = tqdm(loader, desc="Validating", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device).float()

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
            logits = model(images)
            loss = criterion(logits, masks)

        epoch_loss += loss.item()
        
        probs = torch.sigmoid(logits)
        meter.update(probs, masks)

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    val_loss = epoch_loss / len(loader)
    val_aic = meter.compute()
    
    return val_loss, val_aic

# ============================================================
# ВОЗОБНОВЛЕНИЕ ОБУЧЕНИЯ
# ============================================================
def resume_training(model, optimizer, scheduler, save_dir='checkpoints'):
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
        logger.info(f"📂 Возобновление с эпохи {start_epoch}, best_aic = {best_aic:.4f}")
        return start_epoch, best_aic
    except Exception as e:
        logger.warning(f"Не удалось загрузить {last_path} ({e}). Начинаем с нуля.")
        return 1, -1.0

# ============================================================
# ГЛАВНЫЙ ЦИКЛ ОБУЧЕНИЯ
# ============================================================
def train(model, train_loader, val_loader, epochs=10, lr=1e-3,
          device='cuda', save_dir='checkpoints', seed=42, threshold=0.5):
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
        logger.info(f"--- Epoch {epoch}/{epochs} ---")
        current_lr = optimizer.param_groups[0]['lr']

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_aic = eval_epoch(model, val_loader, criterion, device, threshold=threshold)

        logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AIC: {val_aic:.4f} | LR: {current_lr:.2e}")
        log_metrics(epoch, train_loss, val_loss, val_aic, current_lr, log_path=metrics_path)

        if np.isnan(val_aic) or np.isinf(val_aic):
            logger.warning(f"val_aic = {val_aic}, пропускаем обновление скедулера и чекпоинтов!")
            continue

        scheduler.step(val_aic)

        if val_aic > best_aic:
            best_aic = val_aic
            best_path = os.path.join(save_dir, 'best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_aic': best_aic,
            }, best_path)
            logger.info(f"🌟 Новый лучший AIC: {best_aic:.4f}! Модель сохранена.")

        last_path = os.path.join(save_dir, 'last.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_aic': best_aic,
        }, last_path)

if __name__ == "__main__":
    # Настраиваем базовый логгер для тестов
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    logger.info("Запуск тестового прогона цикла обучения...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        def forward(self, x):
            return self.conv(x)

    model = DummyModel().to(device)
    dummy_images = torch.randn(4, 3, 64, 64)
    dummy_masks = torch.randint(0, 2, (4, 1, 64, 64)).float()
    dummy_dataset = TensorDataset(dummy_images, dummy_masks)
    
    g = torch.Generator()
    g.manual_seed(42)
    
    dummy_loader = DataLoader(
        dummy_dataset, batch_size=2, shuffle=True,
        worker_init_fn=seed_worker, generator=g
    )

    train(model, dummy_loader, dummy_loader, epochs=2, device=device)