"""
Обёртка для запуска инференса (src/predict.py).

Что делает
----------
1. Загружает конфиг из JSON (configs/debug.json или configs/final.json).
2. Позволяет переопределить любой параметр через CLI.
3. Валидирует пути, пороги, img_size, существование чекпоинта.
4. Вызывает src.predict.run_predict с подготовленными аргументами.

Зачем нужна обёртка, если есть src/predict.py?
----------------------------------------------
- Единый источник гиперпараметров (JSON-конфиг).
- Меньше опечаток при запуске.
- Проще воспроизводить: сохранил конфиг → запустил → получил результат.

Правила конкурса
----------------
- Используется ТОЛЬКО test.csv (не train, не val).
- Инференс не изменяет данные, только читает.
- Порог бинаризации задаётся отдельно (после threshold tuning).

Запуск
------
    # С конфигом (рекомендуется)
    python scripts/run_predict.py --config configs/final.json

    # С дефолтным конфигом
    python scripts/run_predict.py

    # Переопределить чекпоинт
    python scripts/run_predict.py --config configs/final.json --checkpoint checkpoints/best.pth

    # Переопределить порог (после threshold tuning)
    python scripts/run_predict.py --config configs/final.json --threshold 0.55

    # Быстрая проверка на 10 картинках
    python scripts/run_predict.py --config configs/debug.json --limit 10
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

# Корень проекта в sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import resolve_path


# ============================================================
# ЗНАЧЕНИЯ ПО УМОЛЧАНИЮ
# ============================================================
DEFAULTS = {
    'data_dir': 'data',
    'test_csv': 'stage1/test.csv',
    'checkpoint': 'checkpoints/best.pth',
    'architecture': 'unet',
    'encoder': 'resnet34',
    'img_size': 384,
    'batch_size': 8,
    'num_workers': 2,
    'threshold': 0.5,
    'min_area_ratio': 0.001,
    'pred_dir': 'predictions',
    'submission_csv': 'submission.csv',
    'limit': None,  # None = обработать все строки
}


# ============================================================
# ЗАГРУЗКА КОНФИГА
# ============================================================
def load_config(config_path: Path) -> dict:
    """
    Загружает JSON-конфиг.

    Args:
        config_path: путь к configs/*.json.

    Returns:
        Словарь параметров.

    Raises:
        FileNotFoundError: если файла нет.
        ValueError: если JSON некорректен.
    """
    if not config_path.exists():
        raise FileNotFoundError(f'Конфиг не найден: {config_path}')

    try:
        with open(config_path, encoding='utf-8') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f'Некорректный JSON в {config_path}: {e}'
        ) from e

    if not isinstance(config, dict):
        raise ValueError(
            f'Конфиг должен быть словарём, получено {type(config)}'
        )

    return config


# ============================================================
# МЕРЖ КОНФИГА И CLI
# ============================================================
def merge_args(config: dict, cli_args: argparse.Namespace) -> SimpleNamespace:
    """
    Объединяет конфиг из JSON с CLI-аргументами.
    CLI имеет приоритет над JSON.

    Логика:
      1. Начинаем с DEFAULTS.
      2. Накладываем config (перезаписывает дефолты).
      3. Накладываем CLI-аргументы, которые не None.
    """
    merged = dict(DEFAULTS)
    merged.update(config)

    for key, value in vars(cli_args).items():
        if key == 'config':
            continue
        if value is not None:
            merged[key] = value

    return SimpleNamespace(**merged)


# ============================================================
# ПРИМЕНЕНИЕ --limit
# ============================================================
def apply_limit(args: SimpleNamespace) -> None:
    """
    Если задан --limit, обрезает test.csv до первых N строк.
    Сохраняет временный файл в data/.tmp/.

    Это позволяет быстро протестировать predict.py на 5-10 картинках,
    не прогоняя все 2160.
    """
    if args.limit is None:
        return

    data_dir = Path(args.data_dir)
    test_csv = resolve_path(args.test_csv, data_dir)

    if not test_csv.exists():
        raise FileNotFoundError(f'Не найден test.csv: {test_csv}')

    # Читаем оригинал
    with open(test_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    rows_limited = rows[:args.limit]

    # Пишем временный файл
    tmp_dir = data_dir / '.tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_csv = tmp_dir / f'test_limited_{args.limit}.csv'

    with open(tmp_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_limited)

    print(f'⚠️  --limit={args.limit}: используется {tmp_csv}')
    print(f'   (из {len(rows)} строк взято {len(rows_limited)})')

    # Переопределяем test_csv на временный
    args.test_csv = str(tmp_csv.relative_to(data_dir))


# ============================================================
# ВАЛИДАЦИЯ
# ============================================================
def validate_args(args: SimpleNamespace) -> None:
    """
    Проверяет, что все параметры корректны.

    Raises:
        ValueError: если параметр вне допустимого диапазона.
        FileNotFoundError: если файл не найден.
    """
    # --- Порог ---
    if not 0.0 < args.threshold < 1.0:
        raise ValueError(
            f'threshold должен быть в (0, 1), получено {args.threshold}'
        )

    # --- min_area_ratio ---
    if args.min_area_ratio < 0:
        raise ValueError(
            f'min_area_ratio >= 0, получено {args.min_area_ratio}'
        )

    # --- img_size ---
    if args.img_size <= 0 or args.img_size % 32 != 0:
        raise ValueError(
            f'img_size > 0 и кратен 32, получено {args.img_size}'
        )

    # --- batch_size ---
    if args.batch_size <= 0:
        raise ValueError(
            f'batch_size > 0, получено {args.batch_size}'
        )

    # --- limit ---
    if args.limit is not None and args.limit <= 0:
        raise ValueError(
            f'limit > 0 или None, получено {args.limit}'
        )

    # --- Пути ---
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f'data_dir не найден: {data_dir}')

    test_csv = resolve_path(args.test_csv, data_dir)
    if not test_csv.exists():
        raise FileNotFoundError(f'test.csv не найден: {test_csv}')

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f'Чекпоинт не найден: {checkpoint}')

    # --- Архитектура ---
    from src.model import ARCH_REGISTRY
    if args.architecture not in ARCH_REGISTRY:
        raise ValueError(
            f'Неизвестная архитектура: {args.architecture}. '
            f'Доступные: {list(ARCH_REGISTRY.keys())}'
        )


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Запуск инференса через конфиг + CLI-переопределения',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Конфиг ---
    parser.add_argument('--config', type=str, default='configs/debug.json',
                        help='Путь к JSON-конфигу')

    # --- Переопределения (None = взять из конфига) ---
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--test-csv', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--architecture', type=str, default=None,
                        choices=['unet', 'unetplusplus',
                                 'deeplabv3plus', 'fpn', 'manet'])
    parser.add_argument('--encoder', type=str, default=None)
    parser.add_argument('--img-size', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--min-area-ratio', type=float, default=None)
    parser.add_argument('--pred-dir', type=str, default=None)
    parser.add_argument('--submission-csv', type=str, default=None)
    parser.add_argument('--limit', type=int, default=None,
                        help='Обработать только первые N строк (для теста)')

    cli_args = parser.parse_args()

    # --- 1. Загрузка конфига ---
    config_path = Path(cli_args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if config_path.exists():
        print(f'📂 Загружаю конфиг: {config_path}')
        config = load_config(config_path)
        print(f'   Параметров: {len(config)}')
    else:
        print(f'⚠️  Конфиг не найден: {config_path}')
        print(f'   Используются дефолты')
        config = {}

    # --- 2. Мерж с CLI ---
    args = merge_args(config, cli_args)

    # --- 3. Применяем --limit ---
    apply_limit(args)

    # --- 4. Валидация ---
    validate_args(args)

    # --- 5. Финальный отчёт ---
    print()
    print('=' * 60)
    print('📋 Параметры инференса:')
    print('=' * 60)
    print(f'   data_dir:        {args.data_dir}')
    print(f'   test_csv:        {args.test_csv}')
    print(f'   checkpoint:      {args.checkpoint}')
    print(f'   architecture:    {args.architecture}')
    print(f'   encoder:         {args.encoder}')
    print(f'   img_size:        {args.img_size}')
    print(f'   batch_size:      {args.batch_size}')
    print(f'   threshold:       {args.threshold}')
    print(f'   min_area_ratio:  {args.min_area_ratio}')
    print(f'   pred_dir:        {args.pred_dir}')
    print(f'   submission_csv:  {args.submission_csv}')
    print(f'   num_workers:     {args.num_workers}')
    print(f'   limit:           {args.limit}')
    print('=' * 60)
    print()

    # --- 6. Запуск ---
    from src.predict import run_predict
    run_predict(args)


if __name__ == '__main__':
    main()