"""
Инференс для AI Challenge 2026 — Digital Detective.
====================================================

ЧТО ДЕЛАЕТ ЭТОТ СКРИПТ
----------------------
Загружает обученную модель (best.pth), прогоняет её на тестовых данных
и создаёт файл submission.csv с ссылками на предсказанные маски.

Пайплайн:
    1. Загрузка чекпоинта (best.pth) в модель.
    2. Создание DataLoader на TestDataset.
    3. Forward pass (torch.inference_mode, AMP на GPU).
    4. Sigmoid: логиты -> вероятности [0, 1].
    5. Бинаризация: probs > threshold -> 0/1.
    6. clean_mask: удаление мелкого шума.
    7. Ресайз маски обратно к оригинальному размеру.
    8. Сохранение PNG (0/255).
    9. Формирование submission.csv.
    10. Замер latency (для проверки лимита 50 мс).

КАК ЗАПУСКАТЬ
-------------
Вариант A: напрямую (все параметры через CLI)

    python -m src.predict \\
        --data-dir data \\
        --test-csv stage1/test.csv \\
        --checkpoint checkpoints/best.pth \\
        --img-size 384 \\
        --batch-size 8 \\
        --threshold 0.5 \\
        --pred-dir predictions \\
        --submission-csv submission.csv

Вариант B: через обёртку (рекомендуется, параметры из конфига)

    python scripts/run_predict.py --config configs/final.json

Вариант C: через единую точку входа

    python solution.py predict

Быстрая проверка на 5 картинках (для отладки):

    python scripts/run_predict.py --config configs/debug.json --limit 5

ЧТО НУЖНО ПЕРЕД ЗАПУСКОМ
------------------------
1. Скачанные тестовые данные:
       data/stage1/test.csv
       data/stage1/test_stage1_img/   (или data/test_stage1_img/)

2. Обученная модель:
       checkpoints/best.pth

3. Правильный порог (после threshold tuning):
       results/optimal_threshold.json  (или задать через --threshold)

4. Активированное окружение:
       conda activate digital-detective

ЧТО СОЗДАЁТ
-----------
    predictions/*.png         — бинарные маски (0/255), размер = оригинал
    submission.csv            — таблица: img_path, prediction_path

ПРАВИЛА КОНКУРСА
----------------
    - Используется ТОЛЬКО test.csv (никакого обучения на тесте).
    - PNG: одноканальные, значения строго 0/255, размер = оригинал.
    - submission.csv: колонки img_path, prediction_path.
    - Лимит времени: 50 мс на 1 картинку (на NVIDIA H100).
    - Лимит модели: <= 100 GFLOPs.

ПРИМЕЧАНИЯ
----------
    - На GPU используется AMP (fp16) для ускорения.
    - На CPU работает в fp32.
    - Первые 2 батча не учитываются при замере latency (прогрев).
    - Если данных нет, скрипт корректно сообщит об ошибке.
"""
import argparse
import csv
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import read_test_csv, TestDataset, resolve_path
from src.model import build_model
from src.postprocess import clean_mask


# ============================================================
# КОНСТАНТЫ ПО УМОЛЧАНИЮ
# ============================================================
DEFAULT_IMG_SIZE = 384
DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_WORKERS = 2
DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_AREA_RATIO = 0.001
DEFAULT_PRED_DIR = 'predictions'
DEFAULT_SUBMISSION_CSV = 'submission.csv'
DEFAULT_ARCHITECTURE = 'unet'
DEFAULT_ENCODER = 'resnet34'
DEFAULT_CHECKPOINT = 'checkpoints/best.pth'
DEFAULT_DATA_DIR = 'data'
DEFAULT_TEST_CSV = 'stage1/test.csv'


# ============================================================
# ЗАГРУЗКА ЧЕКПОИНТА
# ============================================================
def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> torch.nn.Module:
    """
    Загружает чекпоинт в модель.

    Поддерживает три формата чекпоинтов:
        1. {'model_state_dict': ...} — как сохраняет src/train.py
        2. {'state_dict': ...}       — альтернативный формат
        3. Голый state_dict          — просто OrderedDict с весами

    Args:
        model: nn.Module, в которую загружаются веса.
        checkpoint_path: путь к .pth файлу.
        device: torch.device (cpu / cuda).

    Returns:
        model с загруженными весами.

    Raises:
        FileNotFoundError: если чекпоинт не найден.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f'Чекпоинт не найден: {ckpt_path}\n'
            f'Проверь путь --checkpoint или скачай best.pth.'
        )

    print(f'[1/10] Загрузка чекпоинта: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Определяем формат чекпоинта
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state = ckpt['state_dict']
        else:
            state = ckpt
    else:
        state = ckpt

    model.load_state_dict(state)
    print('       Веса загружены.')

    # Метаданные, если есть
    if isinstance(ckpt, dict):
        if 'epoch' in ckpt:
            print(f'       Эпоха: {ckpt["epoch"]}')
        if 'best_aic' in ckpt:
            print(f'       Best AIC: {ckpt["best_aic"]:.4f}')

    return model


# ============================================================
# ОСНОВНОЙ ПАЙПЛАЙН
# ============================================================
@torch.inference_mode()
def run_predict(args) -> None:
    """
    Основной пайплайн инференса.

    Args:
        args: объект с параметрами (data_dir, test_csv, checkpoint,
              img_size, batch_size, threshold, pred_dir, submission_csv,
              architecture, encoder, min_area_ratio, num_workers).
    """
    # --- Определяем устройство ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nУстройство: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    else:
        print('CUDA не доступна, работаем на CPU.')

    data_dir = Path(args.data_dir)
    out_dir = Path(args.pred_dir)

    # =========================================================
    # ШАГ 2: Загрузка test.csv и DataLoader
    # =========================================================
    test_csv = resolve_path(args.test_csv, data_dir)
    print(f'\n[2/10] Чтение {test_csv}')

    if not test_csv.exists():
        raise FileNotFoundError(
            f'test.csv не найден: {test_csv}\n'
            f'Распакуй test_stage1.zip в data/.'
        )

    rows = read_test_csv(test_csv)
    print(f'       Всего строк: {len(rows)}')

    if len(rows) == 0:
        raise RuntimeError('test.csv пустой — нечего предсказывать.')

    test_ds = TestDataset(rows, img_size=args.img_size, data_dir=data_dir)
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )
    print(f'       Батчей: {len(loader)}')

    # =========================================================
    # ШАГ 1 (продолжение): Сборка модели
    # =========================================================
    print(f'\n[3/10] Сборка модели: {args.architecture} + {args.encoder}')
    model = build_model(
        architecture=args.architecture,
        encoder_name=args.encoder,
        encoder_weights=None,  # веса загрузим из чекпоинта
        in_channels=3,
        classes=1,
    )
    model = load_checkpoint(model, args.checkpoint, device)
    model = model.to(device).eval()

    # Папка для PNG-масок
    out_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # ПРОГРЕВ (для корректного замера latency)
    # =========================================================
    warmup_batches = min(3, len(loader))
    print(f'\n[4/10] Прогрев модели ({warmup_batches} батчей)')
    for i, batch in enumerate(loader):
        if i >= warmup_batches:
            break
        images = batch['image'].to(device)
        _ = model(images)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # =========================================================
    # ОСНОВНОЙ ЦИКЛ ИНФЕРЕНСА
    # =========================================================
    print(f'\n[5/10] Инференс ({len(loader)} батчей)')
    sub_rows = []          # строки для submission.csv
    latencies_ms = []      # времена на 1 картинку (мс)

    pbar = tqdm(loader, desc='Predict')
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)

        # --- Замер времени (синхронизация GPU обязательна) ---
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        # =====================================================
        # ШАГ 3: Forward pass (AMP на GPU для ускорения)
        # =====================================================
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == 'cuda'),
        ):
            logits = model(images)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Latency: пропускаем первые 2 батча (прогрев)
        if batch_idx >= 2:
            per_image_ms = elapsed_ms / images.shape[0]
            latencies_ms.extend([per_image_ms] * images.shape[0])

        # =====================================================
        # ШАГ 4: Sigmoid (fp32 для точной бинаризации)
        # =====================================================
        probs = torch.sigmoid(logits.float()).cpu().numpy()  # (B, 1, H, W)

        # =====================================================
        # Обработка каждой картинки в батче
        # =====================================================
        for i in range(probs.shape[0]):
            chng_rel = batch['chng_path'][i]
            orig_h = int(batch['orig_h'][i])
            orig_w = int(batch['orig_w'][i])

            # Вытаскиваем (H, W) из (1, H, W)
            prob = probs[i, 0] if probs[i].ndim == 3 else probs[i]

            # =================================================
            # ШАГ 5: Бинаризация по порогу
            # =================================================
            pred_bin = (prob > args.threshold).astype(np.uint8)

            # =================================================
            # ШАГ 6: clean_mask — удаление мелкого шума
            # =================================================
            cleaned = clean_mask(
                pred_bin,
                min_area_ratio=args.min_area_ratio,
                bin_threshold=args.threshold,
            )

            # =================================================
            # ШАГ 7: Ресайз к оригинальному размеру
            # =================================================
            if cleaned.shape[:2] != (orig_h, orig_w):
                cleaned = cv2.resize(
                    cleaned, (orig_w, orig_h),
                    interpolation=cv2.INTER_NEAREST,
                )

            # =================================================
            # ШАГ 8: Сохранение PNG (0/255)
            # =================================================
            mask_uint8 = (cleaned * 255).astype(np.uint8)
            out_name = Path(chng_rel).stem + '_pred.png'
            out_abs = out_dir / out_name
            cv2.imwrite(str(out_abs), mask_uint8)

            # =================================================
            # ШАГ 9: Строка для submission.csv
            # =================================================
            sub_rows.append({
                'img_path': chng_rel,
                'prediction_path': str(out_abs),
            })

    # =========================================================
    # ШАГ 10: Отчёт по latency
    # =========================================================
    print('\n[6/10] Замер latency')
    if latencies_ms:
        lat_median = float(np.median(latencies_ms))
        lat_p95 = float(np.percentile(latencies_ms, 95))
        lat_max = float(np.max(latencies_ms))
        print(f'        Медиана:  {lat_median:.2f} мс')
        print(f'        p95:      {lat_p95:.2f} мс')
        print(f'        Max:      {lat_max:.2f} мс')
        if device.type == 'cuda':
            if lat_median > 50:
                print('        ПРЕВЫШЕН лимит 50 мс на GPU!')
            else:
                print('        Лимит 50 мс соблюдён.')
        else:
            print('        Замер на CPU. На H100 будет в 10-15 раз быстрее.')
    else:
        print('        Пропущено (меньше 3 батчей).')

    # =========================================================
    # ШАГ 9 (продолжение): Сохранение submission.csv
    # =========================================================
    print(f'\n[7/10] Сохранение submission.csv')
    sub_path = Path(args.submission_csv)
    sub_path.parent.mkdir(parents=True, exist_ok=True)

    # Делаем prediction_path относительным от submission.csv
    sub_parent = sub_path.parent.resolve()
    for row in sub_rows:
        mask_abs = Path(row['prediction_path']).resolve()
        try:
            rel = mask_abs.relative_to(sub_parent)
            row['prediction_path'] = rel.as_posix()
        except ValueError:
            # Если не получилось — оставляем как есть
            pass

    with open(sub_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['img_path', 'prediction_path'])
        writer.writeheader()
        writer.writerows(sub_rows)

    print(f'        Сохранено: {sub_path}')
    print(f'        Строк: {len(sub_rows)}')
    print(f'        Маски: {out_dir}/')

    # =========================================================
    # ИТОГОВЫЙ ОТЧЁТ
    # =========================================================
    print()
    print('=' * 60)
    print('ИНФЕРЕНС ЗАВЕРШЁН')
    print('=' * 60)
    print(f'  Устройство:     {device}')
    print(f'  Модель:         {args.architecture} + {args.encoder}')
    print(f'  Чекпоинт:       {args.checkpoint}')
    print(f'  Порог:          {args.threshold}')
    print(f'  Картинок:       {len(sub_rows)}')
    print(f'  Submission:     {sub_path}')
    print(f'  Маски:          {out_dir}/')
    print('=' * 60)
    print()
    print('Дальше:')
    print('  1. Проверь формат PNG (0/255, 1 канал):')
    print('       python -c "from PIL import Image; import numpy as np; '
          'from pathlib import Path; p = next(Path(\'predictions\').glob(\'*.png\')); '
          'm = np.asarray(Image.open(p)); print(m.shape, np.unique(m))"')
    print('  2. Собери submission.zip:')
    print('       python scripts/build_submission.py')
    print('  3. Отправь submission.zip на платформу конкурса.')
    print('=' * 60)


# ============================================================
# ТОЧКА ВХОДА (CLI)
# ============================================================
def parse_args() -> argparse.Namespace:
    """
    Парсит аргументы командной строки.

    Все параметры опциональны — по умолчанию используются значения
    из блока КОНСТАНТЫ ПО УМОЛЧАНИЮ выше.
    """
    parser = argparse.ArgumentParser(
        description='Инференс для AI Challenge 2026 — Digital Detective.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры запуска:

  # Базовый запуск с дефолтными параметрами
  python -m src.predict

  # Явное указание всех ключевых параметров
  python -m src.predict \\
      --data-dir data \\
      --test-csv stage1/test.csv \\
      --checkpoint checkpoints/best.pth \\
      --threshold 0.54

  # Через обёртку с конфигом (рекомендуется)
  python scripts/run_predict.py --config configs/final.json
        """,
    )

    # --- Пути ---
    parser.add_argument(
        '--data-dir', type=str, default=DEFAULT_DATA_DIR,
        help=f'Корень датасета (default: {DEFAULT_DATA_DIR})',
    )
    parser.add_argument(
        '--test-csv', type=str, default=DEFAULT_TEST_CSV,
        help=f'Путь к test.csv относительно data_dir (default: {DEFAULT_TEST_CSV})',
    )
    parser.add_argument(
        '--checkpoint', type=str, default=DEFAULT_CHECKPOINT,
        help=f'Путь к чекпоинту модели (default: {DEFAULT_CHECKPOINT})',
    )

    # --- Модель ---
    parser.add_argument(
        '--architecture', type=str, default=DEFAULT_ARCHITECTURE,
        choices=['unet', 'unetplusplus', 'deeplabv3plus', 'fpn', 'manet'],
        help=f'Архитектура (default: {DEFAULT_ARCHITECTURE})',
    )
    parser.add_argument(
        '--encoder', type=str, default=DEFAULT_ENCODER,
        help=f'Энкодер (default: {DEFAULT_ENCODER})',
    )

    # --- Инференс ---
    parser.add_argument(
        '--img-size', type=int, default=DEFAULT_IMG_SIZE,
        help=f'Размер входа модели, кратен 32 (default: {DEFAULT_IMG_SIZE})',
    )
    parser.add_argument(
        '--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
        help=f'Размер батча (default: {DEFAULT_BATCH_SIZE})',
    )
    parser.add_argument(
        '--num-workers', type=int, default=DEFAULT_NUM_WORKERS,
        help=f'Число воркеров DataLoader (default: {DEFAULT_NUM_WORKERS})',
    )
    parser.add_argument(
        '--threshold', type=float, default=DEFAULT_THRESHOLD,
        help=f'Порог бинаризации (0..1), (default: {DEFAULT_THRESHOLD})',
    )
    parser.add_argument(
        '--min-area-ratio', type=float, default=DEFAULT_MIN_AREA_RATIO,
        help=f'Мин. площадь компоненты для clean_mask '
             f'(default: {DEFAULT_MIN_AREA_RATIO})',
    )

    # --- Выход ---
    parser.add_argument(
        '--pred-dir', type=str, default=DEFAULT_PRED_DIR,
        help=f'Папка для PNG-масок (default: {DEFAULT_PRED_DIR})',
    )
    parser.add_argument(
        '--submission-csv', type=str, default=DEFAULT_SUBMISSION_CSV,
        help=f'Путь к submission.csv (default: {DEFAULT_SUBMISSION_CSV})',
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """
    Проверяет корректность аргументов.

    Raises:
        ValueError: если параметр вне допустимого диапазона.
    """
    if not 0.0 < args.threshold < 1.0:
        raise ValueError(
            f'--threshold должен быть в (0, 1), получено {args.threshold}'
        )
    if args.min_area_ratio < 0:
        raise ValueError(
            f'--min-area-ratio >= 0, получено {args.min_area_ratio}'
        )
    if args.img_size <= 0 or args.img_size % 32 != 0:
        raise ValueError(
            f'--img-size должен быть > 0 и кратен 32, получено {args.img_size}'
        )
    if args.batch_size <= 0:
        raise ValueError(
            f'--batch-size > 0, получено {args.batch_size}'
        )
    if args.num_workers < 0:
        raise ValueError(
            f'--num-workers >= 0, получено {args.num_workers}'
        )


def main() -> None:
    """Точка входа при запуске как `python -m src.predict`."""
    print('=' * 60)
    print('ИНФЕРЕНС: AI Challenge 2026 — Digital Detective')
    print('=' * 60)
    print('Скрипт: src/predict.py')
    print('Назначение: создание submission.csv на тестовых данных')
    print('=' * 60)

    args = parse_args()
    validate_args(args)

    print()
    print('Параметры запуска:')
    print(f'  data_dir:         {args.data_dir}')
    print(f'  test_csv:         {args.test_csv}')
    print(f'  checkpoint:       {args.checkpoint}')
    print(f'  architecture:     {args.architecture}')
    print(f'  encoder:          {args.encoder}')
    print(f'  img_size:         {args.img_size}')
    print(f'  batch_size:       {args.batch_size}')
    print(f'  num_workers:      {args.num_workers}')
    print(f'  threshold:        {args.threshold}')
    print(f'  min_area_ratio:   {args.min_area_ratio}')
    print(f'  pred_dir:         {args.pred_dir}')
    print(f'  submission_csv:   {args.submission_csv}')

    try:
        run_predict(args)
    except FileNotFoundError as e:
        print()
        print('=' * 60)
        print('ОШИБКА: файл или папка не найдены')
        print('=' * 60)
        print(f'  {e}')
        print()
        print('Что делать:')
        print('  1. Проверь путь --data-dir (должен вести к папке с stage1/)')
        print('  2. Убедись, что test_stage1.zip распакован в data/')
        print('  3. Проверь путь --checkpoint (best.pth должен существовать)')
        print('=' * 60)
        sys.exit(1)
    except RuntimeError as e:
        print()
        print('=' * 60)
        print('ОШИБКА: проблема во время инференса')
        print('=' * 60)
        print(f'  {e}')
        print('=' * 60)
        sys.exit(1)


if __name__ == '__main__':
    main()