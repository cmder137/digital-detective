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
    """
    Фиксирует seed для полной воспроизводимости.
    
    Args:
        seed: Целое число для инициализации генераторов.
    """
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
    """
    Фиксирует сиды в каждом воркере DataLoader.
    
    Используется как worker_init_fn в DataLoader для воспроизводимости
    аугментаций при num_workers > 0.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# ЛОГИРОВАНИЕ
# ============================================================
def setup_logging(log_dir: str = 'results/logs', log_name: str = 'train.log') -> logging.Logger:
    """
    Настраивает логирование в консоль и файл.
    
    Args:
        log_dir: Путь к директории для логов.
        log_name: Имя лог-файла (например, 'train.log' или 'predict.log').
    
    Returns:
        Настроенный Logger.
    """
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('digital_detective')
    logger.setLevel(logging.INFO)
    
    # Очищаем старые хендлеры (защита от дублирования)
    if logger.handlers:
        logger.handlers.clear()
    
    # Формат сообщений
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    
    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, log_name))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Stream handler
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    return logger


def log_metrics(epoch, train_loss, val_loss, val_aic, lr, log_path='results/metrics.csv'):
    """
    Безопасное логирование метрик в CSV.
    
    Args:
        epoch: Номер эпохи.
        train_loss: Loss на обучении.
        val_loss: Loss на валидации.
        val_aic: AIC Score на валидации.
        lr: Текущий learning rate.
        log_path: Путь к CSV-файлу.
    """
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
# ЖЕЛЕЗО
# ============================================================
def get_device() -> torch.device:
    """
    Определяет лучшее доступное устройство для вычислений.
    
    Returns:
        torch.device: CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


# ============================================================
# I/O УТИЛИТЫ
# ============================================================
def ensure_dir(path: str):
    """Создаёт директорию, если её не существует."""
    os.makedirs(path, exist_ok=True)


def save_json(data: dict, path: str):
    """Сохраняет словарь в JSON-файл."""
    dir_name = os.path.dirname(path)
    if dir_name:  # Создаём директорию только если она указана
        ensure_dir(dir_name)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_json(path: str) -> dict:
    """Загружает словарь из JSON-файла."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON-файл не найден: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)