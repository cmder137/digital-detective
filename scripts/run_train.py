import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Импорты наших модулей
from src.dataset import read_train_csv, filter_existing_rows, TrainDataset
from src.model import build_model, ARCH_REGISTRY
from src.train import train
from src.utils import set_seed
from src.utils import seed_worker


# Пытаемся импортировать стратифицированный сплит от Человека 1
try:
    from src.dataset import stratified_split
except ImportError:
    # Фолбек, если функция еще не влита в main (сигнатура синхронизирована)
    def stratified_split(rows, val_ratio=0.1, seed=42):
        random.seed(seed)
        pos_rows = [r for r in rows if r.get('gt') is not None]
        neg_rows = [r for r in rows if r.get('gt') is None]
        
        random.shuffle(pos_rows)
        random.shuffle(neg_rows)
        
        val_pos = int(len(pos_rows) * val_ratio)
        val_neg = int(len(neg_rows) * val_ratio)
        
        val_rows = pos_rows[:val_pos] + neg_rows[:val_neg]
        train_rows = pos_rows[val_pos:] + neg_rows[val_neg:]
        
        random.shuffle(train_rows)
        random.shuffle(val_rows)
        return train_rows, val_rows


# --- Утилиты для пайплайна ---
def setup_logging() -> logging.Logger:
    """Изолированная настройка логгера без сайд-эффектов при импорте модуля"""
    os.makedirs('results/logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler('results/logs/train.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def dict_to_tuple_collate(batch):
    """Преобразует список словарей (из TrainDataset) в кортеж тензоров (для train.py)"""
    images = torch.stack([b['image'] for b in batch])
    masks = torch.stack([b['mask'] for b in batch])
    return images, masks


def check_flops(model, device, img_size, logger):
    """Безопасная проверка лимита FLOPs перед началом обучения"""
    logger.info("Запуск предварительной проверки FLOPs...")
    
    # Изолированный импорт для поддержки старых версий PyTorch
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        logger.warning("FlopCounterMode недоступен (требуется PyTorch >= 2.1). Пропуск проверки FLOPs.")
        return

    # Перевод модели в eval для корректного замера (без порчи BN статистики)
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)
    
    try:
        flop_ctx = FlopCounterMode(model, display=False)
    except TypeError:
        flop_ctx = FlopCounterMode(display=False)
        
    with flop_ctx as flop_counter, torch.inference_mode():
        model(dummy_input)
        
    gflops = flop_counter.get_total_flops() / 1e9
    logger.info(f"Расчетные GFLOPs: {gflops:.2f} (Лимит: 100.00)")
    
    if gflops > 100.0:
        raise RuntimeError(f"БЛОКИРОВКА: Модель потребляет {gflops:.2f} GFLOPs! Лимит превышен.")


def main():
    parser = argparse.ArgumentParser(description="Запуск обучения Digital Detective")
    parser.add_argument('--data-dir', type=str, default='data', help='Путь к папке data')
    parser.add_argument('--architecture', type=str, default='unet', choices=list(ARCH_REGISTRY.keys()), help='Архитектура модели')
    parser.add_argument('--encoder', type=str, default='resnet34', help='Энкодер для SMP')
    parser.add_argument('--epochs', type=int, default=15, help='Количество эпох')
    parser.add_argument('--batch-size', type=int, default=16, help='Размер батча')
    parser.add_argument('--img-size', type=int, default=256, help='Размер картинки')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.1, help='Доля валидации')
    parser.add_argument('--threshold', type=float, default=0.5, help='Порог бинаризации')
    parser.add_argument('--workers', type=int, default=4, help='num_workers для DataLoader')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--fresh', action='store_true', help='Начать с нуля (удалить старые чекпоинты)')
    args = parser.parse_args()

    # Валидация аргументов
    if not (0.0 < args.val_split < 1.0):
        parser.error("--val-split должен быть в диапазоне от 0.0 до 1.0")

    logger = setup_logging()

    # Очистка чекпоинтов при запуске с нуля
    if args.fresh:
        logger.info("Флаг --fresh: удаление старых чекпоинтов...")
        for p in ['checkpoints/last.pth', 'checkpoints/best.pth']:
            if os.path.exists(p):
                os.remove(p)

    # 1. Сохранение конфига и инициализация сидов
    os.makedirs('checkpoints', exist_ok=True)
    with open('checkpoints/config_train.json', 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)

    set_seed(args.seed)
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    # Поддержка MPS для разработки на Mac + CUDA для боевого обучения
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    logger.info(f"Запуск на {device} | Модель: {args.architecture} | Энкодер: {args.encoder} | Размер: {args.img_size}")

    # 2. Подготовка данных
    data_dir = Path(args.data_dir)
    csv_path = data_dir / 'stage1' / 'train.csv'
    
    rows = read_train_csv(csv_path)
    rows = filter_existing_rows(rows, data_dir, show_progress=True)
    
    if not rows:
        logger.error(f"Ни одного файла не найдено. Проверьте --data-dir: {args.data_dir}")
        sys.exit(1)
    
    # Исправлено имя параметра на val_ratio согласно контракту
    train_rows, val_rows = stratified_split(rows, val_ratio=args.val_split, seed=args.seed)
    logger.info(f"Данные разбиты (Stratified): Train={len(train_rows)} | Val={len(val_rows)}")

    train_ds = TrainDataset(train_rows, img_size=args.img_size, train=True, data_dir=data_dir)
    val_ds = TrainDataset(val_rows, img_size=args.img_size, train=False, data_dir=data_dir)

    # 3. Инициализация Dataloader
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.workers, pin_memory=True, drop_last=True,
        collate_fn=dict_to_tuple_collate,
        worker_init_fn=seed_worker, generator=generator
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.workers, pin_memory=True,
        collate_fn=dict_to_tuple_collate,
        worker_init_fn=seed_worker, generator=generator
    )

    # 4. Сборка модели и проверка FLOPs
    model = build_model(
        architecture=args.architecture,
        encoder_name=args.encoder,
        encoder_weights='imagenet',
        in_channels=3,
        classes=1
    ).to(device)

    check_flops(model, device, args.img_size, logger)

    # 5. Запуск цикла
    logger.info("Старт обучения...")
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_dir='checkpoints',
        seed=args.seed,
        threshold=args.threshold
    )
    logger.info("Обучение завершено.")

if __name__ == "__main__":
    main()