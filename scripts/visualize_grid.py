"""
Создаёт визуализации датасета в формате сеток 5×10 (50 картинок на сетку).

Для каждой картинки:
  - Загружает img из data/stage1/train/img/.
  - Загружает mask из data/stage1/train/mask/ (если нет — считает маску пустой).
  - Накладывает маску красным полупрозрачным слоем.
  - Подписывает: индекс, размер (W×H), площадь маски (в % от картинки).

Результат: N файлов grid_XXX.png в указанной папке.

Запуск:
    python scripts/visualize_grid.py --n 5000 --per-grid 50 --out results/plots/grids

Пример:
    python scripts/visualize_grid.py --n 100 --per-grid 50 --cols 5
        → создаст 2 файла по 50 картинок
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Добавляем корень проекта в sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import (
    read_train_csv, resolve_path, load_rgb, load_mask_binary,
    filter_existing_rows,
)


# ============================================================
# СОЗДАНИЕ ОДНОЙ СЕТКИ
# ============================================================
def make_grid(rows, data_dir, out_path, cols=5, overlay_alpha=0.4, title_prefix='#'):
    """
    Создаёт одну сетку cols × N/cols с картинками и их масками.

    Args:
        rows: список строк датасета (словари с 'chng' и 'gt').
        data_dir: корень датасета.
        out_path: путь для сохранения PNG.
        cols: число колонок в сетке (по умолчанию 5).
        overlay_alpha: прозрачность красной маски (0.4 = 40% красного).
        title_prefix: префикс для заголовков.
    """
    n = len(rows)
    rows_count = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows_count, cols, figsize=(cols * 3, rows_count * 3))
    axes = np.array(axes).reshape(-1)

    for i, row in enumerate(tqdm(rows, desc=f'  {out_path.name}', leave=False)):
        # --- Загрузка изображения ---
        img_path = resolve_path(row['chng'], data_dir)
        img = load_rgb(img_path)
        h, w = img.shape[:2]

        # --- Загрузка маски (или нулевая) ---
        if row.get('gt') is not None:
            gt_path = resolve_path(row['gt'], data_dir)
            if gt_path.exists():
                mask = load_mask_binary(gt_path)
                # Ресайз, если размеры не совпадают
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                mask = np.zeros((h, w), dtype=np.float32)
        else:
            mask = np.zeros((h, w), dtype=np.float32)

        area_pct = 100 * float(mask.sum()) / mask.size

        # --- Наложение маски красным ---
        if mask.sum() > 0:
            overlay = img.copy()
            overlay[mask > 0] = [255, 0, 0]
            blended = (
                (1 - overlay_alpha) * img + overlay_alpha * overlay
            ).astype(np.uint8)
        else:
            blended = img

        # --- Отрисовка ---
        ax = axes[i]
        ax.imshow(blended)
        ax.set_title(
            f'{title_prefix}{i} | {w}×{h} | area={area_pct:.2f}%',
            fontsize=8,
        )
        ax.axis('off')

    # Спрятать пустые оси (если n не кратно cols)
    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Создаёт grid-визуализации датасета (5×10 с масками)'
    )
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Корень датасета (где лежит stage1/)')
    parser.add_argument('--n', type=int, default=5000,
                        help='Сколько всего картинок обработать')
    parser.add_argument('--per-grid', type=int, default=50,
                        help='Сколько картинок в одной сетке (5×10=50)')
    parser.add_argument('--cols', type=int, default=5,
                        help='Колонок в сетке (по умолчанию 5)')
    parser.add_argument('--out', type=str, default='results/plots/grids',
                        help='Куда сохранять PNG-файлы')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed для воспроизводимости выборки')
    parser.add_argument('--alpha', type=float, default=0.4,
                        help='Прозрачность красной маски (0..1)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Чтение CSV ---
    train_csv = data_dir / 'stage1' / 'train.csv'
    if not train_csv.exists():
        raise FileNotFoundError(f'Не найден {train_csv}. Проверь --data-dir.')
    print(f'📂 Читаю {train_csv}')
    rows = read_train_csv(train_csv)
    print(f'   Всего строк в CSV: {len(rows)}')

    # --- 2. Фильтрация по существующим файлам ---
    rows = filter_existing_rows(rows, data_dir, show_progress=True)
    print(f'   Доступно строк: {len(rows)}')

    if len(rows) == 0:
        raise RuntimeError('Нет доступных картинок. Проверь путь --data-dir.')

    # --- 3. Выборка N картинок ---
    if len(rows) > args.n:
        random.seed(args.seed)
        rows = random.sample(rows, args.n)
    print(f'   Выбрано для визуализации: {len(rows)}')

    # --- 4. Разбивка на grid'ы ---
    per_grid = args.per_grid
    n_grids = (len(rows) + per_grid - 1) // per_grid
    print(f'   Создам {n_grids} grid-файлов по {per_grid} картинок')

    # --- 5. Создание каждой сетки ---
    for g in range(n_grids):
        chunk = rows[g * per_grid:(g + 1) * per_grid]
        out_path = out_dir / f'grid_{g + 1:03d}.png'
        print(f'\n[{g + 1}/{n_grids}] {out_path.name} ({len(chunk)} картинок)')
        make_grid(
            chunk, data_dir, out_path,
            cols=args.cols,
            overlay_alpha=args.alpha,
            title_prefix=f'#{g * per_grid}',  # глобальные номера: #0, #50, #100, ...
        )

    print(f'\n✅ Готово: {n_grids} файлов в {out_dir}')


if __name__ == '__main__':
    main()