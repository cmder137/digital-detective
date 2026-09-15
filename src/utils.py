"""
Утилиты общего назначения для проекта Digital Detective.

Содержит:
  - Воспроизводимость: set_seed(), seed_worker()
  - Логирование: setup_logging(), log_metrics()
  - Железо: get_device()
  - I/O: ensure_dir(), save_json(), load_json()

Зона ответственности: Человек 3 (MLOps), но используется всеми.
"""
import csv
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch

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