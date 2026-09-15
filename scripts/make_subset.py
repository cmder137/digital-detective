"""
Создание подвыборки датасета для быстрой отладки.

Назначение
----------
Полный обучающий датасет содержит ~34 000 доступных пар (img + mask).
Обучение на нём занимает часы даже на GPU. Для отладки пайплайна
(train.py, losses.py, metrics.py) нужен маленький датасет, на котором
1 эпоха проходит за ~2 минуты.

Что делает скрипт
-----------------
1. Читает data/stage1/train.csv.
2. Оставляет только строки, у которых есть файл изображения на диске.
3. Разделяет их на positive (есть маска) и negative (маски нет).
4. Берёт случайные N пар с заданной пропорцией positive/negative.
5. Копирует img (и mask, если есть) в data_subset/stage1/train/.
6. Пишет новый train.csv для подвыборки.

Правила конкурса
----------------
- Используются ТОЛЬКО train-данные (train.csv).
- Тестовые данные (test.csv) НЕ используются.
- Никакие внешние датасеты не привлекаются.
- Скрипт только читает исходные данные, не изменяет их.

Запуск
------
    # Простой запуск: 500 пар, 50/50 positive/negative
    python scripts/make_subset.py --n 500 --out data_subset

    # Только positive (например, для отладки Dice loss)
    python scripts/make_subset.py --n 300 --positive-ratio 1.0

    # Только negative (например, для отладки FPR)
    python scripts/make_subset.py --n 300 --positive-ratio 0.0
"""
import argparse
import csv
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

# Добавляем корень проекта в sys.path, чтобы импортировать src.*
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import read_train_csv, resolve_path, filter_existing_rows


# ============================================================
# РАЗДЕЛЕНИЕ НА POSITIVE / NEGATIVE
# ============================================================
def split_positive_negative(rows: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Разделяет строки датасета на две группы:

    - positive: у строки есть gt-маска (подделка присутствует).
    - negative: у строки нет маски (чистое изображение).

    Предполагается, что rows уже прошёл через filter_existing_rows,
    то есть у всех строк есть файл изображения на диске,
    а gt либо None, либо существует.

    Args:
        rows: список словарей {'chng': str, 'gt': Optional[str]}.

    Returns:
        (positives, negatives) — два списка строк.
    """
    positives = [r for r in rows if r.get('gt') is not None]
    negatives = [r for r in rows if r.get('gt') is None]
    return positives, negatives


# ============================================================
# ВЫБОРКА N ПАР С ЗАДАННОЙ ПРОПОРЦИЕЙ
# ============================================================
def sample_subset(
    positives: List[Dict],
    negatives: List[Dict],
    n: int,
    positive_ratio: float,
    seed: int,
) -> Tuple[List[Dict], int, int]:
    """
    Берёт случайные N строк с заданной пропорцией positive/negative.

    Если одного класса не хватает (например, positive_ratio=0.9, но
    positive всего 10), добираем недостающее из другого класса,
    чтобы общее количество было ровно N (или максимум возможного).

    Args:
        positives: строки с маской.
        negatives: строки без маски.
        n: сколько всего строк взять.
        positive_ratio: доля positive (0.0..1.0).
        seed: random seed для воспроизводимости.

    Returns:
        (subset, n_pos, n_neg) — выбранные строки и их реальное количество.
    """
    random.seed(seed)

    # Целевое количество каждого класса
    n_pos_target = int(n * positive_ratio)
    n_neg_target = n - n_pos_target

    # Ограничиваем доступным количеством
    n_pos = min(n_pos_target, len(positives))
    n_neg = min(n_neg_target, len(negatives))

    # Если одного класса не хватило — добираем из другого
    deficit = n - (n_pos + n_neg)
    if deficit > 0 and n_pos < len(positives):
        add = min(deficit, len(positives) - n_pos)
        n_pos += add
        deficit -= add
    if deficit > 0 and n_neg < len(negatives):
        add = min(deficit, len(negatives) - n_neg)
        n_neg += add

    # Перемешиваем и берём срез
    random.shuffle(positives)
    random.shuffle(negatives)
    subset = positives[:n_pos] + negatives[:n_neg]

    # Перемешиваем финальную выборку, чтобы positive/negative были вперемешку
    random.shuffle(subset)

    return subset, n_pos, n_neg


# ============================================================
# КОПИРОВАНИЕ ФАЙЛОВ И СОЗДАНИЕ CSV
# ============================================================
def copy_subset(
    subset: List[Dict],
    data_dir: Path,
    out_dir: Path,
) -> Tuple[Path, int, int]:
    """
    Копирует img и mask в out_dir и создаёт новый train.csv.

    Структура выхода:
        out_dir/
        └── stage1/
            ├── train.csv
            └── train/
                ├── img/    ← копии изображений
                └── mask/   ← копии масок (только для positive)

    Пути в новом CSV — относительные от out_dir:
        stage1/train/img/xxx.jpg
        stage1/train/mask/xxx.png
    Это значит, что TrainDataset(out_dir=out_dir) загрузит их корректно.

    Args:
        subset: строки для копирования.
        data_dir: корень исходного датасета.
        out_dir: куда копировать.

    Returns:
        (out_csv, n_pos_copied, n_neg_copied)
    """
    out_img_dir = out_dir / 'stage1' / 'train' / 'img'
    out_mask_dir = out_dir / 'stage1' / 'train' / 'mask'
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / 'stage1' / 'train.csv'
    n_pos = 0
    n_neg = 0

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['orgl_img_path', 'chng_img_path', 'gt_path'])

        for row in tqdm(subset, desc='Копирую файлы'):
            # --- Копируем изображение ---
            img_src = resolve_path(row['chng'], data_dir)
            img_dst = out_img_dir / img_src.name

            # Защита от дубликатов (на случай, если одинаковые имена в разных подпапках)
            if not img_dst.exists():
                shutil.copy2(img_src, img_dst)

            # --- Копируем маску (если есть) ---
            gt = row.get('gt')
            if gt is not None:
                gt_src = resolve_path(gt, data_dir)
                gt_dst = out_mask_dir / gt_src.name

                if not gt_dst.exists():
                    shutil.copy2(gt_src, gt_dst)

                writer.writerow([
                    '',  # orgl_img_path — не используем
                    f'stage1/train/img/{img_src.name}',
                    f'stage1/train/mask/{gt_src.name}',
                ])
                n_pos += 1
            else:
                # Negative: маски нет — пишем пустой gt_path
                writer.writerow([
                    '',
                    f'stage1/train/img/{img_src.name}',
                    '',
                ])
                n_neg += 1

    return out_csv, n_pos, n_neg


# ============================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================
def make_subset(
    data_dir: Path,
    out_dir: Path,
    n: int = 500,
    positive_ratio: float = 0.5,
    seed: int = 42,
) -> None:
    """
    Полный пайплайн создания подвыборки.

    Шаги:
      1. Проверка train.csv.
      2. Чтение CSV.
      3. Фильтрация строк без файлов на диске.
      4. Разделение на positive/negative.
      5. Случайная выборка N пар.
      6. Копирование файлов и создание CSV.
    """
    # --- 1. Проверка исходных данных ---
    train_csv = data_dir / 'stage1' / 'train.csv'
    if not train_csv.exists():
        raise FileNotFoundError(
            f'Не найден {train_csv}. Проверь путь --data-dir.'
        )

    print(f'📂 Читаю {train_csv}')
    rows = read_train_csv(train_csv)
    print(f'   Всего строк в CSV: {len(rows)}')

    # --- 2. Фильтрация строк без файлов на диске ---
    print('Проверяю наличие файлов на диске...')
    rows = filter_existing_rows(rows, data_dir, show_progress=True)
    print(f'   Доступно строк: {len(rows)}')

    if len(rows) == 0:
        raise RuntimeError(
            'Нет доступных картинок. Проверь путь --data-dir и структуру папок.'
        )

    # --- 3. Разделение на positive / negative ---
    positives, negatives = split_positive_negative(rows)
    print(f'   Positive: {len(positives)}')
    print(f'   Negative: {len(negatives)}')

    # --- 4. Случайная выборка N пар ---
    subset, n_pos, n_neg = sample_subset(
        positives, negatives, n, positive_ratio, seed
    )
    print(f'   Беру: {n_pos} positive + {n_neg} negative = {len(subset)}')

    # --- 5. Копирование файлов и создание CSV ---
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv, actual_pos, actual_neg = copy_subset(subset, data_dir, out_dir)

    # --- 6. Итоговый отчёт ---
    total_size = sum(
        p.stat().st_size for p in out_dir.rglob('*') if p.is_file()
    )
    size_mb = total_size / 1024 / 1024

    print()
    print('=' * 60)
    print(f'✅ Подвыборка сохранена: {out_dir}')
    print(f'   CSV:      {out_csv}')
    print(f'   Positive: {actual_pos}')
    print(f'   Negative: {actual_neg}')
    print(f'   Всего:    {actual_pos + actual_neg}')
    print(f'   Размер:   {size_mb:.1f} МБ')
    print('=' * 60)


# ============================================================
# ТОЧКА ВХОДА
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description='Создаёт подвыборку датасета для быстрой отладки'
    )
    parser.add_argument(
        '--data-dir', type=str, default='data',
        help='Корень исходного датасета (где лежит stage1/)',
    )
    parser.add_argument(
        '--out', type=str, default='data_subset',
        help='Куда копировать подвыборку',
    )
    parser.add_argument(
        '--n', type=int, default=500,
        help='Сколько всего пар взять (default 500)',
    )
    parser.add_argument(
        '--positive-ratio', type=float, default=0.5,
        help='Доля positive в выборке (0.0..1.0). Default 0.5 = 50/50',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed для воспроизводимости',
    )
    args = parser.parse_args()

    # Валидация аргументов
    if args.n <= 0:
        raise ValueError(f'--n должен быть > 0, получено {args.n}')
    if not 0.0 <= args.positive_ratio <= 1.0:
        raise ValueError(
            f'--positive-ratio должен быть 0.0..1.0, '
            f'получено {args.positive_ratio}'
        )

    make_subset(
        data_dir=Path(args.data_dir),
        out_dir=Path(args.out),
        n=args.n,
        positive_ratio=args.positive_ratio,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()