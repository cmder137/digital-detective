"""
AI Challenge 2026 — Digital Detective
=====================================

Единая точка входа для воспроизведения решения.

Пайплайн
--------
    1. eda      — анализ данных (статистика, графики).
    2. label    — разметка строк positive/negative → labels.csv.
    3. train    — обучение модели → best.pth.
    4. tune     — подбор оптимального порога → optimal_threshold.json.
    5. predict  — инференс на тесте → submission.csv + predictions/.
    6. all      — полный пайплайн (1→5) одной командой.

Требования
----------
    pip install -r requirements.txt
    (см. requirements.txt в корне проекта)

Правила конкурса
----------------
    - test.csv используется ТОЛЬКО для инференса.
    - Не используются внешние датасеты.
    - Воспроизводимость: фиксированный seed, все гиперпараметры в configs/.

Запуск
------
    # Проверить окружение
    python solution.py check

    # Отдельные этапы
    python solution.py eda
    python solution.py label
    python solution.py train
    python solution.py tune
    python solution.py predict

    # Полный пайплайн (займёт часы)
    python solution.py all

    # С кастомным конфигом
    python solution.py train --config configs/final.json
"""
import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

# ============================================================
# КОРЕНЬ ПРОЕКТА
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# ЛОГГЕР
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('solution')


# ============================================================
# ВОСПРОИЗВОДИМОСТЬ
# ============================================================
def set_seed(seed: int = 42):
    """Фиксирует все сиды для полной воспроизводимости."""
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


# ============================================================
# ЗАГРУЗКА КОНФИГА
# ============================================================
DEFAULT_CONFIG = {
    'data_dir': 'data',
    'train_csv': 'stage1/train.csv',
    'test_csv': 'stage1/test.csv',
    'architecture': 'unet',
    'encoder': 'resnet34',
    'img_size': 384,
    'batch_size': 16,
    'num_workers': 4,
    'epochs': 25,
    'lr': 0.0001,
    'val_ratio': 0.15,
    'threshold': 0.5,
    'min_area_ratio': 0.001,
    'seed': 42,
    'checkpoint_dir': 'checkpoints',
    'pred_dir': 'predictions',
    'submission_csv': 'submission.csv',
    'results_dir': 'results',
}


def load_config(path: Path = None) -> SimpleNamespace:
    """Загружает конфиг из JSON и мержит с DEFAULTS."""
    cfg = dict(DEFAULT_CONFIG)
    if path is not None:
        path = Path(path)
        if path.exists():
            with open(path, encoding='utf-8') as f:
                cfg.update(json.load(f))
            logger.info(f'Конфиг загружен: {path}')
        else:
            logger.warning(f'Конфиг не найден: {path}. Используются дефолты.')
    return SimpleNamespace(**cfg)


# ============================================================
# ЭТАП 0: ПРОВЕРКА ОКРУЖЕНИЯ
# ============================================================
def cmd_check(args):
    """Проверяет, что все зависимости установлены."""
    logger.info('=' * 60)
    logger.info('ПРОВЕРКА ОКРУЖЕНИЯ')
    logger.info('=' * 60)

    # --- 1. Python ---
    logger.info(f'Python: {sys.version.split()[0]}')
    logger.info(f'Executable: {sys.executable}')

    # --- 2. Устройство ---
    if torch.cuda.is_available():
        logger.info(f'CUDA: ✅ {torch.cuda.get_device_name(0)}')
        logger.info(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    else:
        logger.warning('CUDA: ❌ (обучение будет медленным)')

    # --- 3. Библиотеки ---
    libs = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('segmentation_models_pytorch', 'smp'),
        ('albumentations', 'albumentations'),
        ('cv2', 'opencv'),
        ('PIL', 'Pillow'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('tqdm', 'tqdm'),
    ]
    for module_name, display_name in libs:
        try:
            mod = __import__(module_name)
            version = getattr(mod, '__version__', 'n/a')
            logger.info(f'{display_name:25s} ✅ {version}')
        except ImportError:
            logger.error(f'{display_name:25s} ❌ не установлен')

    # --- 4. Данные ---
    data_dir = Path(args.data_dir)
    logger.info('')
    logger.info(f'data_dir: {data_dir} — {"✅" if data_dir.exists() else "❌"}')

    train_csv = data_dir / 'stage1' / 'train.csv'
    logger.info(f'train.csv: {"✅" if train_csv.exists() else "❌"}')

    test_csv = data_dir / 'stage1' / 'test.csv'
    logger.info(f'test.csv:  {"✅" if test_csv.exists() else "❌"}')

    # --- 5. Репозиторий ---
    git_dir = PROJECT_ROOT / '.git'
    logger.info(f'.git: {"✅" if git_dir.exists() else "❌"}')

    logger.info('=' * 60)


# ============================================================
# ЭТАП 1: EDA
# ============================================================
def cmd_eda(args):
    """Анализ данных."""
    logger.info('=' * 60)
    logger.info('ЭТАП 1: EDA')
    logger.info('=' * 60)

    from src.dataset import read_train_csv, filter_existing_rows

    cfg = load_config(args.config)
    data_dir = Path(cfg.data_dir)

    train_csv = data_dir / cfg.train_csv if not Path(cfg.train_csv).is_absolute() \
        else Path(cfg.train_csv)

    rows = read_train_csv(train_csv)
    logger.info(f'Всего строк в CSV: {len(rows)}')

    rows = filter_existing_rows(rows, data_dir, show_progress=True)
    logger.info(f'Доступно на диске: {len(rows)}')

    if len(rows) == 0:
        logger.error('Нет доступных картинок. Проверь data_dir.')
        return

    # --- Статистика по маскам ---
    from src.dataset import resolve_path, load_mask_binary

    n_pos, n_neg = 0, 0
    n_missing_gt = 0

    for row in rows[:min(5000, len(rows))]:
        gt = row.get('gt')
        if gt is None:
            n_neg += 1
            continue
        gt_path = resolve_path(gt, data_dir)
        if not gt_path.exists():
            n_missing_gt += 1
            continue
        mask = load_mask_binary(gt_path)
        if mask.sum() == 0:
            n_neg += 1
        else:
            n_pos += 1

    logger.info('')
    logger.info(f'📊 Статистика (на {min(5000, len(rows))} строках):')
    logger.info(f'  Positive: {n_pos}')
    logger.info(f'  Negative: {n_neg}')
    logger.info(f'  Пропущено (нет gt): {n_missing_gt}')

    # --- Сохраняем ---
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / 'eda_summary.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({
            'total_csv': len(rows),
            'available': len(rows),
            'positive': n_pos,
            'negative': n_neg,
        }, f, indent=2)
    logger.info(f'✅ Сохранено: {out}')


# ============================================================
# ЭТАП 2: РАЗМЕТКА СТРОК
# ============================================================
def cmd_label(args):
    """Размечает строки train.csv флагом is_negative."""
    logger.info('=' * 60)
    logger.info('ЭТАП 2: РАЗМЕТКА СТРОК')
    logger.info('=' * 60)

    from scripts.label_rows import label_rows

    cfg = load_config(args.config)
    data_dir = Path(cfg.data_dir)

    out_path = data_dir / 'stage1' / 'labels.csv'

    label_rows(data_dir, out_path)

    logger.info(f'✅ Разметка сохранена: {out_path}')


# ============================================================
# ЭТАП 3: ОБУЧЕНИЕ
# ============================================================
def cmd_train(args):
    """Обучает модель."""
    logger.info('=' * 60)
    logger.info('ЭТАП 3: ОБУЧЕНИЕ')
    logger.info('=' * 60)

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    from torch.utils.data import DataLoader
    from src.dataset import (
        read_train_csv, filter_existing_rows, stratified_split,
        TrainDataset,
    )
    from src.model import build_model
    from src.train import train as train_fn

    data_dir = Path(cfg.data_dir)
    train_csv = data_dir / cfg.train_csv if not Path(cfg.train_csv).is_absolute() \
        else Path(cfg.train_csv)
    labels_path = data_dir / 'stage1' / 'labels.csv'

    # --- Данные ---
    logger.info('Загружаю данные...')
    rows = read_train_csv(train_csv)
    rows = filter_existing_rows(rows, data_dir, show_progress=True)
    logger.info(f'Доступно строк: {len(rows)}')

    #убрала labels_path
    train_rows, val_rows = stratified_split(
        rows, val_ratio=cfg.val_ratio, seed=cfg.seed
    )
    logger.info(f'Train: {len(train_rows)}, Val: {len(val_rows)}')

    train_ds = TrainDataset(
        train_rows, img_size=cfg.img_size, train=True, data_dir=data_dir,
    )
    val_ds = TrainDataset(
        val_rows, img_size=cfg.img_size, train=False, data_dir=data_dir,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    # --- Модель ---
    logger.info(f'Собираю модель: {cfg.architecture} + {cfg.encoder}')
    model = build_model(
        architecture=cfg.architecture,
        encoder_name=cfg.encoder,
        encoder_weights='imagenet',
    )

    # --- Устройство ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Устройство: {device}')

    # --- Обучение ---
    train_fn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.epochs,
        lr=cfg.lr,
        device=device,
        save_dir=cfg.checkpoint_dir,
        seed=cfg.seed,
        threshold=cfg.threshold,
    )

    logger.info('✅ Обучение завершено')
    logger.info(f'   best.pth: {cfg.checkpoint_dir}/best.pth')


# ============================================================
# ЭТАП 4: THRESHOLD TUNING
# ============================================================
def cmd_tune(args):
    """Подбор оптимального порога на валидации."""
    logger.info('=' * 60)
    logger.info('ЭТАП 4: THRESHOLD TUNING')
    logger.info('=' * 60)

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    from torch.utils.data import DataLoader
    from src.dataset import (
        read_train_csv, filter_existing_rows, stratified_split,
        TrainDataset,
    )
    from src.model import build_model

    data_dir = Path(cfg.data_dir)
    train_csv = data_dir / cfg.train_csv if not Path(cfg.train_csv).is_absolute() \
        else Path(cfg.train_csv)
    labels_path = data_dir / 'stage1' / 'labels.csv'
    checkpoint = Path(cfg.checkpoint_dir) / 'best.pth'

    if not checkpoint.exists():
        logger.error(f'Чекпоинт не найден: {checkpoint}')
        return

    # --- Данные ---
    #убрала labels_path
    rows = read_train_csv(train_csv)
    rows = filter_existing_rows(rows, data_dir, show_progress=True)
    _, val_rows = stratified_split(
        rows, val_ratio=cfg.val_ratio, seed=cfg.seed
    )

    val_ds = TrainDataset(
        val_rows, img_size=cfg.img_size, train=False, data_dir=data_dir,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    # --- Модель ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(
        architecture=cfg.architecture,
        encoder_name=cfg.encoder,
        encoder_weights=None,
    )
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    model = model.to(device).eval()

    # --- Собираем вероятности ---
    logger.info('Прогон модели на валидации...')
    all_probs, all_masks = [], []

    with torch.inference_mode():
        for batch in val_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16,
                enabled=(device.type == 'cuda'),
            ):
                logits = model(images)
            probs = torch.sigmoid(logits.float()).cpu().numpy()[:, 0]
            all_probs.append(probs)
            all_masks.append(masks.cpu().numpy()[:, 0])

    all_probs = np.concatenate(all_probs, axis=0)
    all_masks = (np.concatenate(all_masks, axis=0) > 0.5).astype(np.float32)

    # --- Перебор порогов ---
    thresholds = np.arange(0.30, 0.71, 0.02)
    logger.info(f'Перебираю {len(thresholds)} порогов...')

    results = []
    for th in thresholds:
        pred_bin = (all_probs >= th).astype(np.float32)
        gt_sums = all_masks.sum(axis=(1, 2))
        is_pos = gt_sums > 0
        is_neg = ~is_pos

        if is_pos.sum() > 0:
            inter = (pred_bin[is_pos] * all_masks[is_pos]).sum(axis=(1, 2))
            denom = pred_bin[is_pos].sum(axis=(1, 2)) + all_masks[is_pos].sum(axis=(1, 2))
            dice = np.where(denom > 0, 2.0 * inter / (denom + 1e-6), 1.0).mean()
        else:
            dice = 0.0

        if is_neg.sum() > 0:
            h, w = all_masks.shape[1], all_masks.shape[2]
            areas = pred_bin[is_neg].sum(axis=(1, 2)) / (h * w)
            fpr = (areas >= 0.01).mean()
            comp2 = 1.0 - fpr
        else:
            comp2 = 1.0

        aic = 2.0 * dice * comp2 / (dice + comp2 + 1e-6) if (dice + comp2) > 0 else 0.0
        results.append({'threshold': float(th), 'aic': float(aic),
                        'dice': float(dice), '1_minus_fpr': float(comp2)})
        logger.info(f'  th={th:.2f} | AIC={aic:.4f} | Dice={dice:.4f} | 1-FPR={comp2:.4f}')

    best = max(results, key=lambda r: r['aic'])
    logger.info('')
    logger.info(f'🏆 Оптимальный порог: {best["threshold"]:.2f}')
    logger.info(f'   AIC: {best["aic"]:.4f}')

    # --- Сохраняем ---
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / 'optimal_threshold.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({**best, 'all_results': results}, f, indent=2)
    logger.info(f'✅ Сохранено: {out}')


# ============================================================
# ЭТАП 5: ИНФЕРЕНС
# ============================================================
def cmd_predict(args):
    """Инференс на тесте."""
    logger.info('=' * 60)
    logger.info('ЭТАП 5: ИНФЕРЕНС')
    logger.info('=' * 60)

    cfg = load_config(args.config)

    # Подхватить оптимальный порог, если есть
    opt_path = Path(cfg.results_dir) / 'optimal_threshold.json'
    if opt_path.exists():
        with open(opt_path, encoding='utf-8') as f:
            opt = json.load(f)
        cfg.threshold = opt.get('threshold', cfg.threshold)
        logger.info(f'Порог из {opt_path}: {cfg.threshold:.2f}')

    from src.predict import run_predict

    # Собираем args для run_predict
    pred_args = SimpleNamespace(
        data_dir=cfg.data_dir,
        test_csv=cfg.test_csv,
        checkpoint=str(Path(cfg.checkpoint_dir) / 'best.pth'),
        architecture=cfg.architecture,
        encoder=cfg.encoder,
        img_size=cfg.img_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        threshold=cfg.threshold,
        min_area_ratio=cfg.min_area_ratio,
        pred_dir=cfg.pred_dir,
        submission_csv=cfg.submission_csv,
    )

    run_predict(pred_args)
    logger.info('✅ Инференс завершён')


# ============================================================
# ЭТАП 6: ПОЛНЫЙ ПАЙПЛАЙН
# ============================================================
def cmd_all(args):
    """Полный пайплайн 1→5."""
    logger.info('=' * 60)
    logger.info('ПОЛНЫЙ ПАЙПЛАЙН')
    logger.info('=' * 60)

    cfg = load_config(args.config)   # ← ДОБАВИТЬ ЭТУ СТРОКУ

    t0 = time.perf_counter()

    logger.info('')
    logger.info('>>> ЭТАП 1/5: EDA')
    cmd_eda(args)

    logger.info('')
    logger.info('>>> ЭТАП 2/5: Разметка строк')
    cmd_label(args)

    logger.info('')
    logger.info('>>> ЭТАП 3/5: Обучение')
    cmd_train(args)

    logger.info('')
    logger.info('>>> ЭТАП 4/5: Threshold tuning')
    cmd_tune(args)

    logger.info('')
    logger.info('>>> ЭТАП 5/5: Инференс')
    cmd_predict(args)

    elapsed = (time.perf_counter() - t0) / 3600
    logger.info('')
    logger.info('=' * 60)
    logger.info(f'✅ ПАЙПЛАЙН ЗАВЕРШЁН ЗА {elapsed:.2f} ч')
    logger.info('=' * 60)
    logger.info(f'  Модель:     {cfg.checkpoint_dir}/best.pth')
    logger.info(f'  Submission: {cfg.submission_csv}')


# ============================================================
# ТОЧКА ВХОДА
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='AI Challenge 2026 — Digital Detective (единая точка входа)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        '--config', type=str, default=None,
        help='Путь к JSON-конфигу',
    )
    parent.add_argument(
        '--data-dir', type=str, default='data',
        help='Корень датасета',
    )

    sub = parser.add_subparsers(dest='cmd', required=True)

    sub.add_parser('check',   parents=[parent], help='Проверка окружения')
    sub.add_parser('eda',     parents=[parent], help='EDA: статистика и графики')
    sub.add_parser('label',   parents=[parent], help='Разметка строк → labels.csv')
    sub.add_parser('train',   parents=[parent], help='Обучение модели → best.pth')
    sub.add_parser('tune',    parents=[parent], help='Threshold tuning → optimal_threshold.json')
    sub.add_parser('predict', parents=[parent], help='Инференс → submission.csv')
    sub.add_parser('all',     parents=[parent], help='Полный пайплайн 1→5')

    args = parser.parse_args()

    cmd_map = {
        'check': cmd_check,
        'eda': cmd_eda,
        'label': cmd_label,
        'train': cmd_train,
        'tune': cmd_tune,
        'predict': cmd_predict,
        'all': cmd_all,
    }
    cmd_map[args.cmd](args)


if __name__ == '__main__':
    main()