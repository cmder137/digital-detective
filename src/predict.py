"""
Инференс для AI Challenge 2026 — Digital Detective.

Что делает (по шагам):
  1. Загрузка чекпоинта (best.pth) в модель.
  2. DataLoader на TestDataset.
  3. Forward через модель (torch.inference_mode()).
  4. Sigmoid: логиты → вероятности [0, 1].
  5. Бинаризация: probs > threshold → 0/1.
  6. clean_mask: удаление мелкого шума.
  7. Ресайз маски обратно к оригинальному размеру.
  8. Сохранение PNG (0/255).
  9. submission.csv (img_path, prediction_path).
  10. Замер latency (для проверки лимита 50 мс).

Правила конкурса
----------------
- Используются ТОЛЬКО test.csv (без обучения на тесте).
- PNG: одноканальные, 0/255, размер = оригинал.
- submission.csv: колонки img_path, prediction_path.
- Лимит: 50 мс на картинку (на H100).

Запуск
------
    python -m src.predict \\
      --data-dir data \\
      --test-csv stage1/test.csv \\
      --checkpoint checkpoints/best.pth \\
      --img-size 384 \\
      --threshold 0.5 \\
      --pred-dir predictions \\
      --submission-csv submission.csv
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import read_test_csv, TestDataset, resolve_path
from src.model import build_model
from src.postprocess import clean_mask


# ============================================================
# ШАГ 1: ЗАГРУЗКА ЧЕКПОИНТА
# ============================================================
def load_checkpoint(model, checkpoint_path: str, device):
    """
    Загружает чекпоинт в модель.

    Поддерживает несколько форматов:
      - {'model_state_dict': ...}
      - {'state_dict': ...}
      - голый state_dict

    Args:
        model: nn.Module.
        checkpoint_path: путь к .pth.
        device: torch.device.

    Returns:
        Model с загруженными весами.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Чекпоинт не найден: {ckpt_path}')

    print(f'📂 Загружаю чекпоинт: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Определяем формат
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
    print('   ✅ Веса загружены')

    if isinstance(ckpt, dict):
        if 'epoch' in ckpt:
            print(f'   Эпоха: {ckpt["epoch"]}')
        if 'best_aic' in ckpt:
            print(f'   Best AIC: {ckpt["best_aic"]:.4f}')

    return model


# ============================================================
# ОСНОВНОЙ ПАЙПЛАЙН
# ============================================================
@torch.inference_mode()
def run_predict(args) -> None:
    """Основной пайплайн инференса."""

    # --- Устройство ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'🖥️  Устройство: {device}')

    data_dir = Path(args.data_dir)
    out_dir = Path(args.pred_dir)

    # =========================================================
    # ШАГ 2: DataLoader
    # =========================================================
    test_csv = resolve_path(args.test_csv, data_dir)
    print(f'📂 Читаю {test_csv}')
    rows = read_test_csv(test_csv)
    print(f'   Всего строк: {len(rows)}')

    if len(rows) == 0:
        raise RuntimeError('Пустой test.csv — нечего предсказывать')

    test_ds = TestDataset(rows, img_size=args.img_size, data_dir=data_dir)
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )
    print(f'   Батчей: {len(loader)}')

    # =========================================================
    # ШАГ 1 (продолжение): Модель
    # =========================================================
    print(f'🏗️  Собираю модель: {args.architecture} + {args.encoder}')
    model = build_model(
        architecture=args.architecture,
        encoder_name=args.encoder,
        encoder_weights=None,  # веса из чекпоинта
        in_channels=3,
        classes=1,
    )
    model = load_checkpoint(model, args.checkpoint, device)
    model = model.to(device).eval()

    # Папка для предсказаний
    out_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # ШАГ 3 (прогрев): чтобы замер latency был корректным
    # =========================================================
    warmup_batches = min(3, len(loader))
    print(f'🔥 Прогрев модели ({warmup_batches} батчей)')
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
    print(f'\n🚀 Инференс ({len(loader)} батчей)')
    sub_rows = []
    latencies_ms = []

    pbar = tqdm(loader, desc='Predict')
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)

        # --- Замер времени ---
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        # =====================================================
        # ШАГ 3: Forward
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

        # Latency: пропускаем первые 2 батча (стабилизация)
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
            # ШАГ 5: Бинаризация
            # =================================================
            pred_bin = (prob > args.threshold).astype(np.uint8)

            # =================================================
            # ШАГ 6: clean_mask (удаление шума)
            # =================================================
            cleaned = clean_mask(
                pred_bin,
                min_area_ratio=args.min_area_ratio,
                bin_threshold=args.threshold,  # для 0/1-маски игнорируется
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
            # prediction_path — относительный от submission.csv
            sub_rows.append({
                'img_path': chng_rel,
                'prediction_path': out_abs.as_posix(),
            })

    # =========================================================
    # ШАГ 10: Отчёт по latency
    # =========================================================
    print()
    print('=' * 60)
    if latencies_ms:
        lat_median = float(np.median(latencies_ms))
        lat_p95 = float(np.percentile(latencies_ms, 95))
        lat_max = float(np.max(latencies_ms))
        print(f'⏱️  Latency ({device.type}, на 1 картинку):')
        print(f'   Медиана:  {lat_median:.2f} мс')
        print(f'   p95:      {lat_p95:.2f} мс')
        print(f'   Max:      {lat_max:.2f} мс')
        if device.type == 'cuda':
            if lat_median > 50:
                print('   ⚠️  Превышен лимит 50 мс на GPU!')
            else:
                print('   ✅ Лимит 50 мс соблюдён')
        else:
            print('   ℹ️  Замер на CPU. На H100 будет в 10–15 раз быстрее.')
    print('=' * 60)

    # =========================================================
    # ШАГ 9: Сохранение submission.csv
    # =========================================================
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
            # Если по какой-то причине не получилось — оставляем абсолютный
            pass

    with open(sub_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['img_path', 'prediction_path'])
        writer.writeheader()
        writer.writerows(sub_rows)

    print(f'\n✅ Сохранено: {sub_path}')
    print(f'   Строк: {len(sub_rows)}')
    print(f'   Маски: {out_dir}/')


# ============================================================
# ТОЧКА ВХОДА
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Инференс для AI Challenge 2026'
    )

    # Пути
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Корень датасета')
    parser.add_argument('--test-csv', type=str, default='stage1/test.csv',
                        help='Путь к test.csv относительно data_dir')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/best.pth',
                        help='Путь к чекпоинту')

    # Модель
    parser.add_argument('--architecture', type=str, default='unet',
                        choices=['unet', 'unetplusplus',
                                 'deeplabv3plus', 'fpn', 'manet'],
                        help='Архитектура')
    parser.add_argument('--encoder', type=str, default='resnet34',
                        help='Энкодер')

    # Инференс
    parser.add_argument('--img-size', type=int, default=384,
                        help='Размер входа модели')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Размер батча')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Воркеры DataLoader')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Порог бинаризации (0..1)')
    parser.add_argument('--min-area-ratio', type=float, default=0.001,
                        help='Мин. площадь компоненты для clean_mask')

    # Выход
    parser.add_argument('--pred-dir', type=str, default='predictions',
                        help='Папка для PNG-масок')
    parser.add_argument('--submission-csv', type=str,
                        default='submission.csv',
                        help='Путь к submission.csv')

    args = parser.parse_args()

    # Валидация
    if not 0.0 < args.threshold < 1.0:
        raise ValueError(
            f'--threshold должен быть в (0, 1), получено {args.threshold}'
        )
    if args.min_area_ratio < 0:
        raise ValueError(
            f'--min-area-ratio >= 0, получено {args.min_area_ratio}'
        )
    if args.img_size <= 0:
        raise ValueError(
            f'--img-size > 0, получено {args.img_size}'
        )

    run_predict(args)


if __name__ == '__main__':
    main()