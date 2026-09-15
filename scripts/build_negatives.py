"""
Сбор negative-примеров (чистых картинок без подделок) из train.csv.

Назначение
----------
В датасете есть как positive (с маской подделки), так и negative
(картинки без подделок). Для оценки метрики FPR и для отладки модели
полезно иметь отдельный список negative-примеров.

Скрипт собирает все строки, у которых в CSV НЕТ gt_path (или он пустой),
проверяет, что файл img существует на диске, и записывает результат
в negatives.csv.

Что это даёт
------------
1. ML Engineer может отдельно посчитать FPR на negative-выборке.
2. Удобно проверять баланс positive/negative в датасете.
3. Можно обучать модель отдельно на negative для отладки.

Правила конкурса
----------------
- Используются ТОЛЬКО train-данные (train.csv).
- test.csv НЕ трогается.
- Никакие внешние источники не привлекаются.

Запуск
------
    python scripts/build_negatives.py --data-dir data --out data/stage1/negatives.csv
"""
import argparse
import csv
import sys
from pathlib import Path

from tqdm import tqdm

# Добавляем корень проекта в sys.path, чтобы импортировать src.*
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import read_train_csv, resolve_path


# ============================================================
# ОСНОВНАЯ ЛОГИКА
# ============================================================
def build_negatives(data_dir: Path, out_path: Path) -> int:
    """
    Собирает все negative-примеры из train.csv.

    Negative — это строка, у которой нет gt_path (или он пустой).
    Дополнительно проверяется, что файл изображения существует на диске.

    Args:
        data_dir: корень датасета (где лежит stage1/).
        out_path: путь для сохранения negatives.csv.

    Returns:
        Количество найденных negative-примеров.
    """
    train_csv = data_dir / 'stage1' / 'train.csv'
    if not train_csv.exists():
        raise FileNotFoundError(
            f'Не найден {train_csv}. Проверь --data-dir.'
        )

    print(f'📂 Читаю {train_csv}')
    rows = read_train_csv(train_csv)
    print(f'   Всего строк в CSV: {len(rows)}')

    # --- Проходим по всем строкам ---
    negatives = []
    n_positive = 0
    n_missing_img = 0

    for row in tqdm(rows, desc='Ищу negative'):
        # Positive — пропускаем
        if row.get('gt') is not None:
            n_positive += 1
            continue

        # Negative — проверяем, что img существует
        img_path = resolve_path(row['chng'], data_dir)
        if not img_path.exists():
            n_missing_img += 1
            continue

        # Сохраняем относительный путь как в CSV
        negatives.append(row['chng'])

    # --- Отчёт ---
    print()
    print('=' * 60)
    print(f'📊 Итоги:')
    print(f'   Всего строк:         {len(rows)}')
    print(f'   Positive (с маской): {n_positive}')
    print(f'   Negative (без маски):{len(negatives)}')
    print(f'   Пропущено (нет img): {n_missing_img}')
    print('=' * 60)

    # --- Записываем CSV ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['img_path'])
        for path in negatives:
            writer.writerow([path])

    print(f'\n✅ Сохранено: {out_path} ({len(negatives)} строк)')
    return len(negatives)


# ============================================================
# ТОЧКА ВХОДА
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description='Собирает negative-примеры (без gt_path) из train.csv'
    )
    parser.add_argument(
        '--data-dir', type=str, default='data',
        help='Корень датасета (где лежит stage1/)',
    )
    parser.add_argument(
        '--out', type=str, default='data/stage1/negatives.csv',
        help='Куда сохранить negatives.csv',
    )
    args = parser.parse_args()

    build_negatives(Path(args.data_dir), Path(args.out))


if __name__ == '__main__':
    main()