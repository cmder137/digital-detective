"""
Датасет для AI Challenge 2026 — Digital Detective.

Содержит:
  - Утилиты: load_rgb, load_mask_binary, resolve_path.
  - Чтение CSV: read_train_csv, read_test_csv.
  - Фильтрацию строк без файлов: filter_existing_rows.
  - Dataset-классы: TrainDataset, TestDataset.

Особенности:
  - Поддержка negative-примеров: если gt_path пустой — возвращаем нулевую маску.
  - Умное разрешение путей: работает, если CSV и диск расходятся на 'stage1/'.
  - ImageNet-нормализация — в src/transforms.py.

Запуск теста:
    python -m src.dataset
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.transforms import get_train_transforms, get_val_transforms


# ============================================================
# УТИЛИТЫ
# ============================================================
def resolve_path(path: str, data_dir: Path) -> Path:
    """
    Умное разрешение пути к данным.

    Пробует несколько вариантов, пока не найдёт существующий файл:
      1. data_dir / path
      2. data_dir.parent / path
      3. cwd / path
      4. data_dir / path без 'stage1/' в начале
      5. data_dir.parent / path без 'stage1/'

    Если ничего не найдено — возвращает первый вариант (для отладки).
    """
    if not path or not str(path).strip():
        return Path('')  # пустой путь (negative-пример)

    p = Path(str(path).strip().replace('\\', '/'))

    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(data_dir / p)
        candidates.append(data_dir.parent / p)
        candidates.append(Path.cwd() / p)

        # Если путь начинается со 'stage1/', пробуем без него
        if p.parts and p.parts[0] == 'stage1':
            without = Path(*p.parts[1:])
            candidates.append(data_dir / without)
            candidates.append(data_dir.parent / without)

    for c in candidates:
        if c.exists():
            return c

    # Ничего не нашли — возвращаем первый для отладки
    return candidates[0]


def load_rgb(path: Path) -> np.ndarray:
    """Загружает RGB-изображение как numpy uint8 (H, W, 3)."""
    return np.asarray(Image.open(path).convert('RGB'))


def load_mask_binary(path: Path) -> np.ndarray:
    """
    Загружает маску как бинарный float32 (H, W) со значениями 0.0 / 1.0.
    GT-маски в датасете — PNG со значениями 0/255.
    """
    m = np.asarray(Image.open(path))
    if m.ndim == 3:
        m = m[:, :, 0]
    return (m > 127).astype(np.float32)


# ============================================================
# ЧТЕНИЕ CSV
# ============================================================
def read_train_csv(csv_path: Path) -> List[Dict]:
    """
    Читает train.csv.

    Возвращает список словарей: {'chng': str, 'gt': Optional[str]}.
    Если gt_path пустой — 'gt' = None (negative-пример).
    """
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            chng = (row.get('chng_img_path') or '').strip()
            gt = (row.get('gt_path') or '').strip()
            if not chng:
                continue
            rows.append({
                'chng': chng,
                'gt': gt if gt else None,
            })
    return rows


def read_test_csv(csv_path: Path) -> List[Dict]:
    """
    Читает test.csv.
    Возвращает список словарей: {'chng': str}.
    """
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = (row.get('img_path') or row.get('chng_img_path') or '').strip()
            if not img:
                continue
            rows.append({'chng': img})
    return rows


def filter_existing_rows(rows: List[Dict], data_dir: Path,
                         show_progress: bool = True) -> List[Dict]:
    """
    Оставляет только те строки, у которых есть файл изображения.
    Если gt указан, но файла нет — строка считается negative (gt=None).

    Возвращает отфильтрованный список.
    """
    try:
        from tqdm import tqdm
        iterator = tqdm(rows, desc='Проверяю файлы') if show_progress else rows
    except ImportError:
        iterator = rows

    result = []
    for row in iterator:
        img_path = resolve_path(row['chng'], data_dir)
        if not img_path.exists():
            continue

        gt = row.get('gt')
        if gt is not None:
            gt_path = resolve_path(gt, data_dir)
            if not gt_path.exists():
                # Файла маски нет — трактуем как negative
                row = {'chng': row['chng'], 'gt': None}

        result.append(row)
    return result


# ============================================================
# DATASET
# ============================================================
class TrainDataset(Dataset):
    """
    Датасет для обучения.

    Возвращает словарь:
      'image': torch.Tensor (3, H, W) — нормализованное RGB-изображение
      'mask':  torch.Tensor (1, H, W) — бинарная маска {0.0, 1.0}

    Если строка — negative (gt=None), маска будет полностью нулевой.
    """
    def __init__(self, rows: List[Dict], img_size: int,
                 train: bool = True, data_dir: Path = Path('data')):
        self.rows = rows
        self.img_size = img_size
        self.train = train
        self.data_dir = Path(data_dir)
        self.transforms = (
            get_train_transforms(img_size) if train
            else get_val_transforms(img_size)
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]

        # --- Загрузка изображения ---
        img_path = resolve_path(row['chng'], self.data_dir)
        img = load_rgb(img_path)
        h, w = img.shape[:2]

        # --- Загрузка маски (или нулевая для negative) ---
        if row.get('gt') is None:
            mask = np.zeros((h, w), dtype=np.float32)
        else:
            gt_path = resolve_path(row['gt'], self.data_dir)
            mask = load_mask_binary(gt_path)

        # --- Аугментации (синхронно для img и mask) ---
        augmented = self.transforms(image=img, mask=mask)
        img_t = augmented['image']     # (3, H, W) float32
        mask_t = augmented['mask']     # (H, W) float32

        # Приводим маску к форме (1, H, W)
        if isinstance(mask_t, np.ndarray):
            mask_t = torch.from_numpy(mask_t)
        if mask_t.ndim == 2:
            mask_t = mask_t.unsqueeze(0)

        return {
            'image': img_t,
            'mask': mask_t.float(),
        }


class TestDataset(Dataset):
    """
    Датасет для инференса.

    Возвращает словарь:
      'image':     torch.Tensor (3, H, W)
      'chng_path': str  — путь к исходному изображению (как в CSV)
      'orig_h':    int  — оригинальная высота
      'orig_w':    int  — оригинальная ширина
    """
    def __init__(self, rows: List[Dict], img_size: int,
                 data_dir: Path = Path('data')):
        self.rows = rows
        self.img_size = img_size
        self.data_dir = Path(data_dir)
        self.transforms = get_val_transforms(img_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        row = self.rows[idx]
        img_path = resolve_path(row['chng'], self.data_dir)
        img = load_rgb(img_path)
        orig_h, orig_w = img.shape[:2]

        augmented = self.transforms(image=img)
        img_t = augmented['image']

        return {
            'image': img_t,
            'chng_path': row['chng'],
            'orig_h': orig_h,
            'orig_w': orig_w,
        }


# ============================================================
# ТЕСТ
# ============================================================
if __name__ == '__main__':
    from torch.utils.data import DataLoader

    DATA_DIR = Path('data')
    TRAIN_CSV = DATA_DIR / 'stage1' / 'train.csv'

    print(f'📂 Читаю {TRAIN_CSV}')
    rows = read_train_csv(TRAIN_CSV)
    print(f'   Всего строк в CSV: {len(rows)}')

    rows = filter_existing_rows(rows, DATA_DIR, show_progress=True)
    print(f'   Доступно на диске: {len(rows)}')

    if len(rows) == 0:
        print('❌ Нет доступных картинок. Проверь --data-dir.')
        raise SystemExit(1)

    # --- Тест TrainDataset ---
    subset = rows[:16]
    ds = TrainDataset(subset, img_size=256, train=True, data_dir=DATA_DIR)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

    batch = next(iter(loader))
    print()
    print(f'image:       {batch["image"].shape} {batch["image"].dtype}')
    print(f'mask:        {batch["mask"].shape} {batch["mask"].dtype}')
    print(f'mask unique: {torch.unique(batch["mask"])}')
    print(f'mask range:  [{batch["mask"].min():.2f}, {batch["mask"].max():.2f}]')
    print('✅ TrainDataset работает')

    # --- Тест TestDataset ---
    test_ds = TestDataset(subset[:4], img_size=256, data_dir=DATA_DIR)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=0)
    batch_t = next(iter(test_loader))
    print()
    print(f'test image: {batch_t["image"].shape}')
    print(f'test orig:  {batch_t["orig_h"].tolist()} x {batch_t["orig_w"].tolist()}')
    print(f'test paths: {batch_t["chng_path"]}')
    print('✅ TestDataset работает')