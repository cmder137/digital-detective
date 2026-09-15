"""
EDA для AI Challenge 2026 — Digital Detective (Stage 1).

Что делает скрипт:
  1. Читает train.csv.
  2. Пропускает строки, у которых нет файла img или mask на диске.
  3. Выводит общую статистику по датасету:
       - сколько всего картинок в CSV;
       - сколько реально доступно на диске;
       - сколько пропущено (нет файла);
       - сколько positive / negative;
       - распределение размеров изображений;
       - распределение площади масок (% от картинки).
  4. Визуализирует первые N доступных картинок с наложенной маской.
  5. Сохраняет графики в results/plots/.

Запуск:
    python scripts/run_eda.py
    python scripts/run_eda.py --data-dir /path/to/data --n 100
"""
import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm


# ============================================================
# НАСТРОЙКИ ПО УМОЛЧАНИЮ
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent.parent  # корень проекта
DEFAULT_DATA_DIR = SCRIPT_DIR / 'data'
DEFAULT_TRAIN_CSV = 'stage1/train.csv'
DEFAULT_OUT_DIR = SCRIPT_DIR / 'results' / 'plots'


# ============================================================
# УТИЛИТЫ
# ============================================================
def resolve_path(path: str, data_dir: Path) -> Path:
    """Путь к данным — относительно data_dir."""
    p = Path(path.strip().replace('\\', '/'))
    return p if p.is_absolute() else data_dir / p


def load_rgb(path: Path) -> np.ndarray:
    """Загружает RGB-изображение как numpy uint8 (H, W, 3)."""
    return np.asarray(Image.open(path).convert('RGB'))


def load_mask_binary(path: Path) -> np.ndarray:
    """Загружает маску как бинарный uint8 (H, W) со значениями 0/1."""
    m = np.asarray(Image.open(path))
    if m.ndim == 3:
        m = m[:, :, 0]
    return (m > 128).astype(np.uint8)


def read_train_csv(csv_path: Path):
    """Читает train.csv и возвращает список словарей."""
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'chng': row['chng_img_path'].strip(),
                'gt': row['gt_path'].strip(),
            })
    return rows


def check_files_exist(row, data_dir: Path) -> bool:
    """
    Проверяет, что оба файла (img и mask) реально лежат на диске.
    Возвращает True, если оба файла есть, иначе False.
    """
    img_path = resolve_path(row['chng'], data_dir)
    mask_path = resolve_path(row['gt'], data_dir)
    return img_path.exists() and mask_path.exists()


def filter_existing_rows(rows, data_dir: Path):
    """
    Фильтрует rows, оставляя только те, у которых есть и img, и mask.
    Возвращает (existing_rows, n_missing).
    """
    existing = []
    missing = 0
    for row in tqdm(rows, desc='Проверяю наличие файлов'):
        if check_files_exist(row, data_dir):
            existing.append(row)
        else:
            missing += 1
    return existing, missing


# ============================================================
# СТАТИСТИКА
# ============================================================
def compute_stats(rows, data_dir: Path, max_check: int = None):
    """
    Считает статистику по датасету. Предполагается, что все строки в rows
    уже проверены на существование файлов.

    Возвращает dict с полями:
      - total, n_pos, n_neg
      - heights, widths
      - mask_areas (доля от картинки), mask_pixels (абсолютные пиксели)
    """
    total = len(rows)
    n_pos = 0
    n_neg = 0
    heights = []
    widths = []
    mask_areas = []
    mask_pixels = []

    iterator = tqdm(rows, desc='Считаю статистику')
    if max_check is not None:
        iterator = tqdm(rows[:max_check], desc=f'Считаю статистику ({max_check})')

    for row in iterator:
        img = load_rgb(resolve_path(row['chng'], data_dir))
        h, w = img.shape[:2]
        heights.append(h)
        widths.append(w)

        mask = load_mask_binary(resolve_path(row['gt'], data_dir))
        pixels = int(mask.sum())
        mask_pixels.append(pixels)
        mask_areas.append(pixels / (h * w))

        if pixels > 0:
            n_pos += 1
        else:
            n_neg += 1

    return {
        'total': total,
        'n_pos': n_pos,
        'n_neg': n_neg,
        'heights': np.array(heights),
        'widths': np.array(widths),
        'mask_areas': np.array(mask_areas),
        'mask_pixels': np.array(mask_pixels),
    }


def print_stats(stats, n_total_csv: int, n_missing: int):
    """Красиво печатает статистику в консоль."""
    print('\n' + '=' * 60)
    print('📊 ОБЩАЯ ХАРАКТЕРИСТИКА ДАТАСЕТА')
    print('=' * 60)

    print(f'\n📁 Файлы:')
    print(f'   Всего строк в CSV:              {n_total_csv}')
    print(f'   Пропущено (нет img или mask):   {n_missing} '
          f'({100 * n_missing / max(n_total_csv, 1):.1f}%)')
    print(f'   Доступно для анализа:           {stats["total"]}')

    print(f'\n🔢 Классы (по доступным):')
    print(f'   ✅ Positive (есть подделка): {stats["n_pos"]} '
          f'({100 * stats["n_pos"] / max(stats["total"], 1):.1f}%)')
    print(f'   ⬜ Negative (чистые):        {stats["n_neg"]} '
          f'({100 * stats["n_neg"] / max(stats["total"], 1):.1f}%)')

    if stats['total'] == 0:
        print('\n⚠️  Нет доступных картинок для анализа. Проверь --data-dir.')
        return

    print(f'\n📐 Размеры изображений (H×W):')
    print(f'   Height: min={stats["heights"].min()}, '
          f'median={int(np.median(stats["heights"]))}, '
          f'max={stats["heights"].max()}, '
          f'mean={stats["heights"].mean():.1f}')
    print(f'   Width:  min={stats["widths"].min()}, '
          f'median={int(np.median(stats["widths"]))}, '
          f'max={stats["widths"].max()}, '
          f'mean={stats["widths"].mean():.1f}')

    pos_areas = stats['mask_areas'][stats['mask_areas'] > 0]
    pos_pixels = stats['mask_pixels'][stats['mask_pixels'] > 0]

    if len(pos_areas) > 0:
        print(f'\n🎭 Площадь масок (только positive, n={len(pos_areas)}):')
        print(f'   Доля от картинки (%):')
        print(f'     min={pos_areas.min() * 100:.3f}%, '
              f'median={np.median(pos_areas) * 100:.2f}%, '
              f'mean={pos_areas.mean() * 100:.2f}%, '
              f'max={pos_areas.max() * 100:.2f}%')
        print(f'   Абсолютная площадь (пиксели):')
        print(f'     min={pos_pixels.min()}, '
              f'median={int(np.median(pos_pixels))}, '
              f'max={pos_pixels.max()}')

        tiny = (pos_areas < 0.01).sum()
        small = ((pos_areas >= 0.01) & (pos_areas < 0.05)).sum()
        medium = ((pos_areas >= 0.05) & (pos_areas < 0.20)).sum()
        large = (pos_areas >= 0.20).sum()
        print(f'\n📏 Распределение по площади:')
        print(f'   < 1%   (tiny):   {tiny} ({100 * tiny / len(pos_areas):.1f}%)')
        print(f'   1–5%   (small):  {small} ({100 * small / len(pos_areas):.1f}%)')
        print(f'   5–20%  (medium): {medium} ({100 * medium / len(pos_areas):.1f}%)')
        print(f'   > 20%  (large):  {large} ({100 * large / len(pos_areas):.1f}%)')
    else:
        print('\n⚠️  Нет positive-картинок с непустой маской.')

    print('\n' + '=' * 60 + '\n')


# ============================================================
# ВИЗУАЛИЗАЦИЯ
# ============================================================
def visualize_samples(rows, data_dir: Path, out_path: Path,
                      n: int = 100, cols: int = 5):
    """
    Визуализирует N картинок с наложенной маской в сетке cols × rows.
    Предполагается, что все rows уже проверены на существование файлов.
    """
    n = min(n, len(rows))
    if n == 0:
        print('⚠️  Нет картинок для визуализации.')
        return

    rows_count = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows_count, cols, figsize=(cols * 3, rows_count * 3))
    axes = np.array(axes).reshape(-1)

    for i in tqdm(range(n), desc='Рисую примеры'):
        row = rows[i]
        img = load_rgb(resolve_path(row['chng'], data_dir))
        mask = load_mask_binary(resolve_path(row['gt'], data_dir))

        overlay = img.copy()
        if mask.sum() > 0:
            overlay[mask > 0] = [255, 0, 0]

        blended = (0.7 * img + 0.3 * overlay).astype(np.uint8)

        ax = axes[i]
        ax.imshow(blended)
        ax.set_title(f'#{i} | {img.shape[1]}×{img.shape[0]} | '
                     f'area={100 * mask.sum() / mask.size:.2f}%', fontsize=8)
        ax.axis('off')

    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'✅ Визуализация сохранена: {out_path}')


def plot_distributions(stats, out_dir: Path):
    """Строит 3 графика: размеры, распределение площади масок, соотношение классов."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if stats['total'] == 0:
        print('⚠️  Нет данных для графиков.')
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    axes[0].hist(stats['heights'], bins=30, alpha=0.7, label='Height')
    axes[0].hist(stats['widths'], bins=30, alpha=0.7, label='Width')
    axes[0].set_title('Размеры изображений')
    axes[0].set_xlabel('Пиксели')
    axes[0].set_ylabel('Количество')
    axes[0].legend()

    pos_areas = stats['mask_areas'][stats['mask_areas'] > 0] * 100
    axes[1].hist(pos_areas, bins=50, alpha=0.8, color='red')
    axes[1].set_title('Площадь масок (positive)')
    axes[1].set_xlabel('Доля от картинки (%)')
    axes[1].set_ylabel('Количество')
    axes[1].axvline(1, color='black', linestyle='--', label='1% (порог FPR)')
    axes[1].legend()

    axes[2].bar(['Positive', 'Negative'],
                [stats['n_pos'], stats['n_neg']],
                color=['green', 'gray'])
    axes[2].set_title('Соотношение классов')
    axes[2].set_ylabel('Количество')

    plt.tight_layout()
    path = out_dir / 'eda_distributions.png'
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'✅ Графики распределений: {path}')


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='EDA для Digital Detective')
    parser.add_argument('--data-dir', type=str, default=str(DEFAULT_DATA_DIR),
                        help='Корень с данными (где лежит stage1/)')
    parser.add_argument('--train-csv', type=str, default=DEFAULT_TRAIN_CSV,
                        help='Путь к train.csv относительно data_dir')
    parser.add_argument('--out-dir', type=str, default=str(DEFAULT_OUT_DIR),
                        help='Куда сохранять графики')
    parser.add_argument('--n', type=int, default=100,
                        help='Сколько картинок визуализировать')
    parser.add_argument('--cols', type=int, default=5,
                        help='Сколько колонок в сетке')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--max-stats', type=int, default=None,
                        help='Считать статистику только на первых N (для скорости)')
    parser.add_argument('--no-filter', action='store_true',
                        help='Не проверять наличие файлов (для отладки)')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    # 1. Читаем CSV
    csv_path = resolve_path(args.train_csv, data_dir)
    print(f'📂 Читаю {csv_path}')
    rows = read_train_csv(csv_path)
    n_total_csv = len(rows)
    print(f'   Всего строк в CSV: {n_total_csv}')

    # 2. Фильтруем строки, у которых нет файлов на диске
    if args.no_filter:
        existing_rows = rows
        n_missing = 0
        print('⚠️  Фильтрация отключена (--no-filter)')
    else:
        existing_rows, n_missing = filter_existing_rows(rows, data_dir)
        print(f'   Доступно на диске: {len(existing_rows)}')
        print(f'   Пропущено (нет img/mask): {n_missing}')

    if len(existing_rows) == 0:
        print('\n❌ Нет доступных картинок. Проверь путь --data-dir.')
        return

    # 3. Статистика — только по доступным
    stats = compute_stats(existing_rows, data_dir, max_check=args.max_stats)
    print_stats(stats, n_total_csv, n_missing)

    # 4. Графики распределений
    plot_distributions(stats, out_dir)

    # 5. Визуализация первых N доступных
    samples_path = out_dir / f'first_{args.n}_samples.png'
    visualize_samples(existing_rows, data_dir, samples_path,
                      n=args.n, cols=args.cols)

    print('\n🎉 EDA завершён!')


if __name__ == '__main__':
    main()