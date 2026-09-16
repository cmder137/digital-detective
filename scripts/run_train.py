"""
Главная точка входа для запуска обучения (AI Challenge 2026 — Digital Detective).

Зона ответственности скрипта (MLOps / Pipeline):
1. Парсинг CLI-аргументов и сохранение конфига (config_train.json) для воспроизводимости.
2. Фиксация всех генераторов случайных чисел (сидов) в основном потоке и воркерах.
3. Загрузка данных, удаление битых путей и стратифицированное разбиение (Train/Val),
   чтобы сохранить баланс positive/negative классов в выборках.
4. Инициализация DataLoader с кастомным collate_fn (перевод словаря в кортеж).
5. Сборка модели сегментации и предварительный контроль аппаратных лимитов
   (автоматическая блокировка, если модель превышает 100 GFLOPs).
6. Запуск полного цикла обучения с записью логов в файл и консоль.

Пример запуска:
    python scripts/run_train.py --architecture unet --encoder resnet34 --img-size 256 --batch-size 16 --epochs 15
"""

import argparse
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.dataset import read_train_csv, filter_existing_rows, TrainDataset
from src.model import build_model, ARCH_REGISTRY
from src.train import train
from src.utils import set_seed, seed_worker, setup_logging, get_device, save_json

try:
    from src.dataset import stratified_split
except ImportError:
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

def dict_to_tuple_collate(batch):
    images = torch.stack([b['image'] for b in batch])
    masks = torch.stack([b['mask'] for b in batch])
    return images, masks

def check_flops(model, device, img_size, logger):
    logger.info("Запуск предварительной проверки FLOPs...")
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        logger.warning("FlopCounterMode недоступен. Пропуск проверки FLOPs.")
        return

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
    parser.add_argument('--fresh', action='store_true', help='Начать с нуля')
    args = parser.parse_args()

    if not (0.0 < args.val_split < 1.0):
        parser.error("--val-split должен быть в диапазоне от 0.0 до 1.0")

    logger = setup_logging(log_name='train.log')

    if args.fresh:
        logger.info("Флаг --fresh: удаление старых чекпоинтов...")
        for p in ['checkpoints/last.pth', 'checkpoints/best.pth']:
            if os.path.exists(p):
                os.remove(p)

    save_json(vars(args), 'checkpoints/config_train.json')

    set_seed(args.seed)
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    device = get_device()
        
    logger.info(f"Запуск на {device} | Модель: {args.architecture} | Энкодер: {args.encoder} | Размер: {args.img_size}")

    data_dir = Path(args.data_dir)
    csv_path = data_dir / 'stage1' / 'train.csv'
    
    rows = read_train_csv(csv_path)
    rows = filter_existing_rows(rows, data_dir, show_progress=True)
    rows = rows[:100]
    
    if not rows:
        logger.error(f"Ни одного файла не найдено. Проверьте --data-dir: {args.data_dir}")
        sys.exit(1)
    
    train_rows, val_rows = stratified_split(rows, val_ratio=args.val_split, seed=args.seed)
    logger.info(f"Данные разбиты (Stratified): Train={len(train_rows)} | Val={len(val_rows)}")

    train_ds = TrainDataset(train_rows, img_size=args.img_size, train=True, data_dir=data_dir)
    val_ds = TrainDataset(val_rows, img_size=args.img_size, train=False, data_dir=data_dir)

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

    model = build_model(
        architecture=args.architecture,
        encoder_name=args.encoder,
        encoder_weights='imagenet',
        in_channels=3,
        classes=1
    ).to(device)

    check_flops(model, device, args.img_size, logger)

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