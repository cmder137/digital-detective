"""
AI Challenge 2026 — Digital Detective (Stage 1)
Бейзлайн: UNet + ResNet34 + DiceBCE + AIC-метрика + постобработка.

Автор: команда AIC2026 "Generation GPT"
Дата: 2026
"""
import argparse
import csv
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import segmentation_models_pytorch as smp


# ============================================================
# КОНФИГУРАЦИЯ (пути)
# ============================================================
DATA_DIR = Path('')  # TODO: указать путь к данным
SCRIPT_DIR = Path(__file__).resolve().parent

TRAIN_CSV = 'stage1/train.csv'
TEST_CSV = 'stage1/test.csv'

# ImageNet статистика — обязательна для предобученных энкодеров
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ============================================================
# ВОСПРОИЗВОДИМОСТЬ
# ============================================================
def set_seed(seed: int = 42):
    """Фиксирует все сиды для полной воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# РАБОТА С ПУТЯМИ
# ============================================================
def resolve_path(path: str) -> Path:
    """Пути к данным — относительно DATA_DIR."""
    p = Path(path.strip().replace('\\', '/'))
    return p if p.is_absolute() else DATA_DIR / p


def resolve_output_path(path: str) -> Path:
    """Артефакты (checkpoints, predictions, submission) — относительно папки скрипта."""
    p = Path(path.strip().replace('\\', '/'))
    return p if p.is_absolute() else SCRIPT_DIR / p


def save_train_config(args, ckpt_dir: Path):
    """Сохраняет аргументы обучения для воспроизводимости."""
    config = {k: getattr(args, k) for k in vars(args)}
    config['data_dir'] = str(DATA_DIR)
    config['script_dir'] = str(SCRIPT_DIR)
    path = ckpt_dir / 'config_train.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f'config saved -> {path}')


# ============================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================
def load_rgb(path: Path) -> np.ndarray:
    """Загружает RGB-изображение как numpy uint8 (H, W, 3)."""
    img = np.asarray(Image.open(path).convert('RGB'))
    return img


def load_mask_binary(path: Path) -> np.ndarray:
    """Загружает маску как бинарный float32 (H, W) со значениями 0.0/1.0."""
    m = np.asarray(Image.open(path))
    if m.ndim == 3:
        m = m[:, :, 0]
    return (m > 128).astype(np.float32)


def preprocess(img: np.ndarray, mask, size: int, train: bool):
    """
    Ресайз + (опционально) горизонтальный флип + нормализация ImageNet.
    Возвращает: image (3, H, W) float32, mask (H, W) float32 или None, orig_hw.
    """
    h, w = img.shape[:2]
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    if train and random.random() < 0.5:
        img = np.ascontiguousarray(img[:, ::-1])
        if mask is not None:
            mask = np.ascontiguousarray(mask[:, ::-1])
    if mask is not None:
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)

    # Нормализация под ImageNet
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose(2, 0, 1)  # (H, W, 3) -> (3, H, W)
    return img, mask, (h, w)


# ============================================================
# DATASET
# ============================================================
class TrainDataset(Dataset):
    """Датасет для обучения. Возвращает image (3,H,W) и mask (1,H,W)."""
    def __init__(self, rows, img_size: int, train: bool):
        self.rows = rows
        self.img_size = img_size
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = load_rgb(resolve_path(row['chng']))
        mask = load_mask_binary(resolve_path(row['gt']))
        img, mask, _ = preprocess(img, mask, self.img_size, self.train)
        return {
            'image': torch.from_numpy(img),
            'mask': torch.from_numpy(mask).unsqueeze(0),
        }


class TestDataset(Dataset):
    """Датасет для инференса. Возвращает image + путь + оригинальные размеры."""
    def __init__(self, rows, img_size: int):
        self.rows = rows
        self.img_size = img_size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        chng_rel = row['chng']
        img = load_rgb(resolve_path(chng_rel))
        img, _, orig_hw = preprocess(img, None, self.img_size, train=False)
        return {
            'image': torch.from_numpy(img),
            'chng_path': chng_rel,
            'orig_h': orig_hw[0],
            'orig_w': orig_hw[1],
        }


# ============================================================
# ЧТЕНИЕ CSV
# ============================================================
def read_train_csv(csv_path: Path):
    """Читает train.csv с колонками orgl_img_path, chng_img_path, gt_path."""
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append({
                'chng': row['chng_img_path'].strip(),
                'gt': row['gt_path'].strip(),
            })
    return rows


def read_test_csv(csv_path: Path):
    """Читает test.csv с колонкой img_path."""
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            img_path = row.get('img_path') or row.get('chng_img_path')
            if img_path is None:
                raise ValueError(f'test csv must contain "img_path", got: {row.keys()}')
            rows.append({'chng': img_path.strip()})
    return rows


def split_train_val(rows, val_ratio: float, seed: int):
    """Стратифицированное разбиение train/val (простое, по индексам)."""
    idx = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_val = max(1, int(len(rows) * val_ratio))
    val_set = set(idx[:n_val])
    train_rows = [rows[i] for i in idx if i not in val_set]
    val_rows = [rows[i] for i in idx if i in val_set]
    return train_rows, val_rows


# ============================================================
# МОДЕЛЬ И ЛОСС
# ============================================================
def build_model(encoder: str = 'resnet34'):
    """UNet + предобученный энкодер."""
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
        activation=None,  # sigmoid применяем отдельно
    )


def build_loss():
    """
    DiceLoss + BCEWithLogitsLoss — стандарт для сегментации с дисбалансом классов.
    Dice отвечает за качество на маленьких объектах, BCE — за стабильность.
    """
    dice = smp.losses.DiceLoss(mode='binary')
    bce = nn.BCEWithLogitsLoss()
    def loss_fn(logits, targets):
        return dice(logits, targets) + bce(logits, targets)
    return loss_fn


# ============================================================
# МЕТРИКА AIC SCORE
# ============================================================
def calculate_aic(preds: np.ndarray, targets: np.ndarray, threshold: float = 0.5):
    """
    Считает AIC Score по формуле соревнования.
    preds:   (N, H, W) вероятности от 0 до 1
    targets: (N, H, W) бинарные 0/1
    Возвращает: (aic, dice_pos, fpr_neg)
    """
    preds_bin = (preds > threshold).astype(np.uint8)

    # Positive — где есть GT-маска
    pos_mask = targets.sum(axis=(1, 2)) > 0
    if pos_mask.sum() > 0:
        inter = (preds_bin[pos_mask] * targets[pos_mask]).sum()
        union = preds_bin[pos_mask].sum() + targets[pos_mask].sum()
        dice_pos = (2.0 * inter) / (union + 1e-6)
    else:
        dice_pos = 0.0

    # Negative — где GT пустой
    neg_mask = ~pos_mask
    if neg_mask.sum() > 0:
        # Площадь предсказания как доля от всего изображения
        areas = preds_bin[neg_mask].sum(axis=(1, 2)) / (preds_bin.shape[1] * preds_bin.shape[2])
        # Ложная тревога, если площадь >= 1%
        fpr_neg = float(np.mean(areas >= 0.01))
    else:
        fpr_neg = 0.0

    # Гармоническое среднее
    aic = (2 * dice_pos * (1 - fpr_neg)) / (dice_pos + (1 - fpr_neg) + 1e-6)
    return float(aic), float(dice_pos), float(fpr_neg)


# ============================================================
# ПОСТ-ОБРАБОТКА МАСОК
# ============================================================
def clean_mask(mask: np.ndarray, min_area_ratio: float = 0.001) -> np.ndarray:
    """
    Удаляет мелкие связные компоненты из бинарной маски.
    Это критично для FPR: если на чистой картинке модель закрасит 1% пикселей,
    кадр считается ложной тревогой.
    mask: (H, W) uint8, значения 0/1
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    clean = np.zeros_like(mask, dtype=np.uint8)
    min_area = mask.shape[0] * mask.shape[1] * min_area_ratio
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 1
    return clean


# ============================================================
# ОБУЧЕНИЕ
# ============================================================
def train_epoch(model, loader, criterion, optimizer, device):
    """Одна эпоха обучения. Возвращает средний loss."""
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc='train', leave=False):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    """
    Валидация: считает loss И AIC Score.
    Возвращает (val_loss, aic, dice_pos, fpr_neg).
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_masks = []

    for batch in tqdm(loader, desc='val', leave=False):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        logits = model(images)
        total_loss += criterion(logits, masks).item()

        # sigmoid -> вероятности, копим для метрики
        probs = torch.sigmoid(logits).cpu().numpy()[:, 0]  # (B, H, W)
        all_preds.append(probs)
        all_masks.append(masks.cpu().numpy()[:, 0])        # (B, H, W)

    all_preds = np.concatenate(all_preds, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    aic, dice_pos, fpr_neg = calculate_aic(all_preds, all_masks, threshold=0.5)
    val_loss = total_loss / max(len(loader), 1)
    return val_loss, aic, dice_pos, fpr_neg


def run_train(args):
    """Полный цикл обучения с логированием AIC и сохранением лучшей модели."""
    set_seed(args.seed)

    rows = read_train_csv(resolve_path(args.train_csv))
    train_rows, val_rows = split_train_val(rows, args.val_ratio, args.seed)
    print(f'train={len(train_rows)} val={len(val_rows)}')

    train_ds = TrainDataset(train_rows, args.img_size, train=True)
    val_ds = TrainDataset(val_rows, args.img_size, train=False)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args.encoder).to(device)
    criterion = build_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3,
    )

    ckpt_dir = resolve_output_path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / args.checkpoint_name
    save_train_config(args, ckpt_dir)

    # Лог метрик по эпохам
    metrics_log_path = ckpt_dir / 'metrics.csv'
    with open(metrics_log_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'aic', 'dice_pos', 'fpr_neg', 'lr'])

    best_aic = -1.0
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, aic, dice_pos, fpr_neg = eval_epoch(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f'epoch {epoch}/{args.epochs} '
            f'train_loss={tr_loss:.4f} val_loss={va_loss:.4f} '
            f'AIC={aic:.4f} (dice={dice_pos:.4f}, fpr={fpr_neg:.4f}) '
            f'lr={current_lr:.2e}'
        )

        # Логируем в CSV
        with open(metrics_log_path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([epoch, tr_loss, va_loss, aic, dice_pos, fpr_neg, current_lr])

        scheduler.step(aic)

        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'encoder': args.encoder,
            'img_size': args.img_size,
            'aic': aic,
            'dice_pos': dice_pos,
            'fpr_neg': fpr_neg,
        }
        torch.save(state, ckpt_path.with_suffix('.last.pth'))

        if aic > best_aic:
            best_aic = aic
            torch.save(state, ckpt_path.with_suffix('.best.pth'))
            print(f'  🔥 new best AIC={best_aic:.4f} saved -> {ckpt_path.with_suffix(".best.pth")}')


# ============================================================
# ИНФЕРЕНС
# ============================================================
@torch.no_grad()
def run_predict(args):
    """Инференс: чистим маски, ресайзим обратно, сохраняем PNG и submission.csv."""
    set_seed(args.seed)

    rows = read_test_csv(resolve_path(args.test_csv))
    test_ds = TestDataset(rows, args.img_size)
    loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args.encoder).to(device)
    ckpt = torch.load(resolve_output_path(args.checkpoint), map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"loaded checkpoint: AIC={ckpt.get('aic', 'n/a')}, epoch={ckpt.get('epoch', 'n/a')}")

    pred_dir_name = Path(args.pred_dir).as_posix().rstrip('/')
    pred_dir = resolve_output_path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    sub_rows = []
    for batch in tqdm(loader, desc='predict'):
        images = batch['image'].to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()[:, 0]

        for i in range(probs.shape[0]):
            chng_rel = batch['chng_path'][i]
            oh = int(batch['orig_h'][i])
            ow = int(batch['orig_w'][i])

            # Бинаризация с порогом
            pred = (probs[i] > args.threshold).astype(np.uint8)
            # Постобработка — удаляем мелкий шум
            pred = clean_mask(pred, min_area_ratio=args.min_area_ratio)
            # Ресайз обратно к оригинальному разрешению
            pred = cv2.resize(pred, (ow, oh), interpolation=cv2.INTER_NEAREST)
            pred = pred * 255  # 0 / 255 для PNG

            out_name = Path(chng_rel).stem + '_pred.png'
            out_abs = pred_dir / out_name
            cv2.imwrite(str(out_abs), pred)

            sub_rows.append({
                'img_path': chng_rel,
                'prediction_path': f'{pred_dir_name}/{out_name}',
            })

    sub_path = resolve_output_path(args.submission_csv)
    with open(sub_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['img_path', 'prediction_path'])
        writer.writeheader()
        writer.writerows(sub_rows)
    print(f'submission: {sub_path} ({len(sub_rows)} rows)')


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)

    # --- train ---
    p_train = sub.add_parser('train')
    p_train.add_argument('--data-dir', type=str, default=None)
    p_train.add_argument('--train-csv', type=str, default=TRAIN_CSV)
    p_train.add_argument('--val-ratio', type=float, default=0.1)
    p_train.add_argument('--seed', type=int, default=42)
    p_train.add_argument('--img-size', type=int, default=256)
    p_train.add_argument('--batch-size', type=int, default=32)
    p_train.add_argument('--epochs', type=int, default=20)
    p_train.add_argument('--lr', type=float, default=1e-4)
    p_train.add_argument('--encoder', type=str, default='resnet34')
    p_train.add_argument('--num-workers', type=int, default=4)
    p_train.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    p_train.add_argument('--checkpoint-name', type=str, default='baseline_unet_resnet34')

    # --- predict ---
    p_pred = sub.add_parser('predict')
    p_pred.add_argument('--data-dir', type=str, default=None)
    p_pred.add_argument('--test-csv', type=str, default=TEST_CSV)
    p_pred.add_argument('--checkpoint', type=str,
                        default='checkpoints/baseline_unet_resnet34.best.pth')
    p_pred.add_argument('--img-size', type=int, default=256)
    p_pred.add_argument('--batch-size', type=int, default=32)
    p_pred.add_argument('--encoder', type=str, default='resnet34')
    p_pred.add_argument('--num-workers', type=int, default=4)
    p_pred.add_argument('--threshold', type=float, default=0.5,
                        help='Порог бинаризации (подбирается на валидации)')
    p_pred.add_argument('--min-area-ratio', type=float, default=0.001,
                        help='Минимальная площадь компоненты (доля от картинки)')
    p_pred.add_argument('--seed', type=int, default=42)
    p_pred.add_argument('--pred-dir', type=str, default='predictions')
    p_pred.add_argument('--submission-csv', type=str, default='submission.csv')

    args = parser.parse_args()

    global DATA_DIR
    if getattr(args, 'data_dir', None):
        DATA_DIR = Path(args.data_dir)

    if args.cmd == 'train':
        run_train(args)
    elif args.cmd == 'predict':
        run_predict(args)


if __name__ == '__main__':
    main()