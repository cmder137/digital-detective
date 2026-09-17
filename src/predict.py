"""
╨Ш╨╜╤Д╨╡╤А╨╡╨╜╤Б ╨┤╨╗╤П AI Challenge 2026 тАФ Digital Detective.

╨з╤В╨╛ ╨┤╨╡╨╗╨░╨╡╤В (╨┐╨╛ ╤И╨░╨│╨░╨╝):
  1. ╨Ч╨░╨│╤А╤Г╨╖╨║╨░ ╤З╨╡╨║╨┐╨╛╨╕╨╜╤В╨░ (best.pth) ╨▓ ╨╝╨╛╨┤╨╡╨╗╤М.
  2. DataLoader ╨╜╨░ TestDataset.
  3. Forward ╤З╨╡╤А╨╡╨╖ ╨╝╨╛╨┤╨╡╨╗╤М (torch.inference_mode()).
  4. Sigmoid: ╨╗╨╛╨│╨╕╤В╤Л тЖТ ╨▓╨╡╤А╨╛╤П╤В╨╜╨╛╤Б╤В╨╕ [0, 1].
  5. ╨С╨╕╨╜╨░╤А╨╕╨╖╨░╤Ж╨╕╤П: probs > threshold тЖТ 0/1.
  6. clean_mask: ╤Г╨┤╨░╨╗╨╡╨╜╨╕╨╡ ╨╝╨╡╨╗╨║╨╛╨│╨╛ ╤И╤Г╨╝╨░.
  7. ╨а╨╡╤Б╨░╨╣╨╖ ╨╝╨░╤Б╨║╨╕ ╨╛╨▒╤А╨░╤В╨╜╨╛ ╨║ ╨╛╤А╨╕╨│╨╕╨╜╨░╨╗╤М╨╜╨╛╨╝╤Г ╤А╨░╨╖╨╝╨╡╤А╤Г.
  8. ╨б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╕╨╡ PNG (0/255).
  9. submission.csv (img_path, prediction_path).
  10. ╨Ч╨░╨╝╨╡╤А latency (╨┤╨╗╤П ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╨╗╨╕╨╝╨╕╤В╨░ 50 ╨╝╤Б).

╨Я╤А╨░╨▓╨╕╨╗╨░ ╨║╨╛╨╜╨║╤Г╤А╤Б╨░
----------------
- ╨Ш╤Б╨┐╨╛╨╗╤М╨╖╤Г╤О╤В╤Б╤П ╨в╨Ю╨Ы╨м╨Ъ╨Ю test.csv (╨▒╨╡╨╖ ╨╛╨▒╤Г╤З╨╡╨╜╨╕╤П ╨╜╨░ ╤В╨╡╤Б╤В╨╡).
- PNG: ╨╛╨┤╨╜╨╛╨║╨░╨╜╨░╨╗╤М╨╜╤Л╨╡, 0/255, ╤А╨░╨╖╨╝╨╡╤А = ╨╛╤А╨╕╨│╨╕╨╜╨░╨╗.
- submission.csv: ╨║╨╛╨╗╨╛╨╜╨║╨╕ img_path, prediction_path.
- ╨Ы╨╕╨╝╨╕╤В: 50 ╨╝╤Б ╨╜╨░ ╨║╨░╤А╤В╨╕╨╜╨║╤Г (╨╜╨░ H100).

╨Ч╨░╨┐╤Г╤Б╨║
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
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import read_test_csv, TestDataset, resolve_path
from src.model import build_model
from src.postprocess import clean_mask


# ============================================================
# ╨и╨Р╨У 1: ╨Ч╨Р╨У╨а╨г╨Ч╨Ъ╨Р ╨з╨Х╨Ъ╨Я╨Ю╨Ш╨Э╨в╨Р
# ============================================================
def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device):
    """
    ╨Ч╨░╨│╤А╤Г╨╢╨░╨╡╤В ╤З╨╡╨║╨┐╨╛╨╕╨╜╤В ╨▓ ╨╝╨╛╨┤╨╡╨╗╤М.

    ╨Я╨╛╨┤╨┤╨╡╤А╨╢╨╕╨▓╨░╨╡╤В ╤Д╨╛╤А╨╝╨░╤В╤Л:
      - {'model_state_dict': ...}
      - {'state_dict': ...}
      - ╨│╨╛╨╗╤Л╨╣ state_dict

    Args:
        model: nn.Module.
        checkpoint_path: ╨┐╤Г╤В╤М ╨║ .pth.
        device: torch.device.

    Returns:
        Model ╤Б ╨╖╨░╨│╤А╤Г╨╢╨╡╨╜╨╜╤Л╨╝╨╕ ╨▓╨╡╤Б╨░╨╝╨╕.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'╨з╨╡╨║╨┐╨╛╨╕╨╜╤В ╨╜╨╡ ╨╜╨░╨╣╨┤╨╡╨╜: {ckpt_path}')

    print(f'ЁЯУВ ╨Ч╨░╨│╤А╤Г╨╢╨░╤О ╤З╨╡╨║╨┐╨╛╨╕╨╜╤В: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # ╨Ю╨┐╤А╨╡╨┤╨╡╨╗╤П╨╡╨╝ ╤Д╨╛╤А╨╝╨░╤В
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
    print('   тЬЕ ╨Т╨╡╤Б╨░ ╨╖╨░╨│╤А╤Г╨╢╨╡╨╜╤Л')

    if isinstance(ckpt, dict):
        if 'epoch' in ckpt:
            print(f'   ╨н╨┐╨╛╤Е╨░: {ckpt["epoch"]}')
        if 'best_aic' in ckpt:
            print(f'   Best AIC: {ckpt["best_aic"]:.4f}')

    return model


# ============================================================
# ╨Ю╨б╨Э╨Ю╨Т╨Э╨Ю╨Щ ╨Я╨Р╨Щ╨Я╨Ы╨Р╨Щ╨Э
# ============================================================
@torch.inference_mode()
def run_predict(args) -> None:
    """╨Ю╤Б╨╜╨╛╨▓╨╜╨╛╨╣ ╨┐╨░╨╣╨┐╨╗╨░╨╣╨╜ ╨╕╨╜╤Д╨╡╤А╨╡╨╜╤Б╨░."""

    # --- ╨г╤Б╤В╤А╨╛╨╣╤Б╤В╨▓╨╛ ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'ЁЯЦея╕П  ╨г╤Б╤В╤А╨╛╨╣╤Б╤В╨▓╨╛: {device}')

    data_dir = Path(args.data_dir)
    out_dir = Path(args.pred_dir)

    # =========================================================
    # ╨и╨Р╨У 2: DataLoader
    # =========================================================
    test_csv = resolve_path(args.test_csv, data_dir)
    print(f'ЁЯУВ ╨з╨╕╤В╨░╤О {test_csv}')
    rows = read_test_csv(test_csv)
    print(f'   ╨Т╤Б╨╡╨│╨╛ ╤Б╤В╤А╨╛╨║: {len(rows)}')

    if len(rows) == 0:
        raise RuntimeError('╨Я╤Г╤Б╤В╨╛╨╣ test.csv тАФ ╨╜╨╡╤З╨╡╨│╨╛ ╨┐╤А╨╡╨┤╤Б╨║╨░╨╖╤Л╨▓╨░╤В╤М')

    test_ds = TestDataset(rows, img_size=args.img_size, data_dir=data_dir)
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )
    print(f'   ╨С╨░╤В╤З╨╡╨╣: {len(loader)}')

    # =========================================================
    # ╨и╨Р╨У 1 (╨┐╤А╨╛╨┤╨╛╨╗╨╢╨╡╨╜╨╕╨╡): ╨Ь╨╛╨┤╨╡╨╗╤М
    # =========================================================
    print(f'ЁЯПЧя╕П  ╨б╨╛╨▒╨╕╤А╨░╤О ╨╝╨╛╨┤╨╡╨╗╤М: {args.architecture} + {args.encoder}')
    model = build_model(
        architecture=args.architecture,
        encoder_name=args.encoder,
        encoder_weights=None,  # ╨▓╨╡╤Б╨░ ╨╕╨╖ ╤З╨╡╨║╨┐╨╛╨╕╨╜╤В╨░
        in_channels=3,
        classes=1,
    )
    model = load_checkpoint(model, args.checkpoint, device)
    model = model.to(device).eval()

    # ╨Я╨░╨┐╨║╨░ ╨┤╨╗╤П ╨┐╤А╨╡╨┤╤Б╨║╨░╨╖╨░╨╜╨╕╨╣
    out_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # ╨Я╨а╨Ю╨У╨а╨Х╨Т (╨┤╨╗╤П ╨║╨╛╤А╤А╨╡╨║╤В╨╜╨╛╨│╨╛ ╨╖╨░╨╝╨╡╤А╨░ latency)
    # =========================================================
    warmup_batches = min(3, len(loader))
    print(f'ЁЯФе ╨Я╤А╨╛╨│╤А╨╡╨▓ ╨╝╨╛╨┤╨╡╨╗╨╕ ({warmup_batches} ╨▒╨░╤В╤З╨╡╨╣)')
    for i, batch in enumerate(loader):
        if i >= warmup_batches:
            break
        images = batch['image'].to(device)
        _ = model(images)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # =========================================================
    # ╨Ю╨б╨Э╨Ю╨Т╨Э╨Ю╨Щ ╨ж╨Ш╨Ъ╨Ы ╨Ш╨Э╨д╨Х╨а╨Х╨Э╨б╨Р
    # =========================================================
    print(f'\nЁЯЪА ╨Ш╨╜╤Д╨╡╤А╨╡╨╜╤Б ({len(loader)} ╨▒╨░╤В╤З╨╡╨╣)')
    sub_rows = []
    latencies_ms = []

    pbar = tqdm(loader, desc='Predict')
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)

        # --- ╨Ч╨░╨╝╨╡╤А ╨▓╤А╨╡╨╝╨╡╨╜╨╕ ---
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        # =====================================================
        # ╨и╨Р╨У 3: Forward
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

        # Latency: ╨┐╤А╨╛╨┐╤Г╤Б╨║╨░╨╡╨╝ ╨┐╨╡╤А╨▓╤Л╨╡ 2 ╨▒╨░╤В╤З╨░ (╤Б╤В╨░╨▒╨╕╨╗╨╕╨╖╨░╤Ж╨╕╤П)
        if batch_idx >= 2:
            per_image_ms = elapsed_ms / images.shape[0]
            latencies_ms.extend([per_image_ms] * images.shape[0])

        # =====================================================
        # ╨и╨Р╨У 4: Sigmoid (fp32 ╨┤╨╗╤П ╤В╨╛╤З╨╜╨╛╨╣ ╨▒╨╕╨╜╨░╤А╨╕╨╖╨░╤Ж╨╕╨╕)
        # =====================================================
        probs = torch.sigmoid(logits.float()).cpu().numpy()  # (B, 1, H, W)

        # =====================================================
        # ╨Ю╨▒╤А╨░╨▒╨╛╤В╨║╨░ ╨║╨░╨╢╨┤╨╛╨╣ ╨║╨░╤А╤В╨╕╨╜╨║╨╕ ╨▓ ╨▒╨░╤В╤З╨╡
        # =====================================================
        for i in range(probs.shape[0]):
            chng_rel = batch['chng_path'][i]
            orig_h = int(batch['orig_h'][i])
            orig_w = int(batch['orig_w'][i])

            # ╨Т╤Л╤В╨░╤Б╨║╨╕╨▓╨░╨╡╨╝ (H, W) ╨╕╨╖ (1, H, W)
            prob = probs[i, 0] if probs[i].ndim == 3 else probs[i]

            # =================================================
            # ╨и╨Р╨У 5: ╨С╨╕╨╜╨░╤А╨╕╨╖╨░╤Ж╨╕╤П
            # =================================================
            pred_bin = (prob > args.threshold).astype(np.uint8)

            # =================================================
            # ╨и╨Р╨У 6: clean_mask (╤Г╨┤╨░╨╗╨╡╨╜╨╕╨╡ ╤И╤Г╨╝╨░)
            # =================================================
            cleaned = clean_mask(
                pred_bin,
                min_area_ratio=args.min_area_ratio,
                bin_threshold=args.threshold,
            )

            # =================================================
            # ╨и╨Р╨У 7: ╨а╨╡╤Б╨░╨╣╨╖ ╨║ ╨╛╤А╨╕╨│╨╕╨╜╨░╨╗╤М╨╜╨╛╨╝╤Г ╤А╨░╨╖╨╝╨╡╤А╤Г
            # =================================================
            if cleaned.shape[:2] != (orig_h, orig_w):
                cleaned = cv2.resize(
                    cleaned, (orig_w, orig_h),
                    interpolation=cv2.INTER_NEAREST,
                )

            # =================================================
            # ╨и╨Р╨У 8: ╨б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╕╨╡ PNG (0/255)
            # =================================================
            mask_uint8 = (cleaned * 255).astype(np.uint8)
            out_name = Path(chng_rel).stem + '_pred.png'
            out_abs = out_dir / out_name
            cv2.imwrite(str(out_abs), mask_uint8)

            # =================================================
            # ╨и╨Р╨У 9: ╨б╤В╤А╨╛╨║╨░ ╨┤╨╗╤П submission.csv
            # =================================================
            sub_rows.append({
                'img_path': chng_rel,
                'prediction_path': str(out_abs),
            })

    # =========================================================
    # ╨и╨Р╨У 10: ╨Ю╤В╤З╤С╤В ╨┐╨╛ latency
    # =========================================================
    print()
    print('=' * 60)
    if latencies_ms:
        lat_median = float(np.median(latencies_ms))
        lat_p95 = float(np.percentile(latencies_ms, 95))
        lat_max = float(np.max(latencies_ms))
        print(f'тП▒я╕П  Latency ({device.type}, ╨╜╨░ 1 ╨║╨░╤А╤В╨╕╨╜╨║╤Г):')
        print(f'   ╨Ь╨╡╨┤╨╕╨░╨╜╨░:  {lat_median:.2f} ╨╝╤Б')
        print(f'   p95:      {lat_p95:.2f} ╨╝╤Б')
        print(f'   Max:      {lat_max:.2f} ╨╝╤Б')
        if device.type == 'cuda':
            if lat_median > 50:
                print('   тЪая╕П  ╨Я╤А╨╡╨▓╤Л╤И╨╡╨╜ ╨╗╨╕╨╝╨╕╤В 50 ╨╝╤Б ╨╜╨░ GPU!')
            else:
                print('   тЬЕ ╨Ы╨╕╨╝╨╕╤В 50 ╨╝╤Б ╤Б╨╛╨▒╨╗╤О╨┤╤С╨╜')
        else:
            print('   тД╣я╕П  ╨Ч╨░╨╝╨╡╤А ╨╜╨░ CPU. ╨Э╨░ H100 ╨▒╤Г╨┤╨╡╤В ╨▓ 10тАУ15 ╤А╨░╨╖ ╨▒╤Л╤Б╤В╤А╨╡╨╡.')
    print('=' * 60)

    # =========================================================
    # ╨и╨Р╨У 9 (╨┐╤А╨╛╨┤╨╛╨╗╨╢╨╡╨╜╨╕╨╡): ╨б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╕╨╡ submission.csv
    # =========================================================
    sub_path = Path(args.submission_csv)
    sub_path.parent.mkdir(parents=True, exist_ok=True)

    # ╨Ф╨╡╨╗╨░╨╡╨╝ prediction_path ╨╛╤В╨╜╨╛╤Б╨╕╤В╨╡╨╗╤М╨╜╤Л╨╝ ╨╛╤В submission.csv
    sub_parent = sub_path.parent.resolve()
    for row in sub_rows:
        mask_abs = Path(row['prediction_path']).resolve()
        try:
            rel = mask_abs.relative_to(sub_parent)
            row['prediction_path'] = rel.as_posix()
        except ValueError:
            # ╨Х╤Б╨╗╨╕ ╨╜╨╡ ╨┐╨╛╨╗╤Г╤З╨╕╨╗╨╛╤Б╤М тАФ ╨╛╤Б╤В╨░╨▓╨╗╤П╨╡╨╝ ╨║╨░╨║ ╨╡╤Б╤В╤М
            pass

    with open(sub_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['img_path', 'prediction_path'])
        writer.writeheader()
        writer.writerows(sub_rows)

    print(f'\nтЬЕ ╨б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╛: {sub_path}')
    print(f'   ╨б╤В╤А╨╛╨║: {len(sub_rows)}')
    print(f'   ╨Ь╨░╤Б╨║╨╕: {out_dir}/')


# ============================================================
# ╨в╨Ю╨з╨Ъ╨Р ╨Т╨е╨Ю╨Ф╨Р
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='╨Ш╨╜╤Д╨╡╤А╨╡╨╜╤Б ╨┤╨╗╤П AI Challenge 2026'
    )

    # ╨Я╤Г╤В╨╕
    parser.add_argument('--data-dir', type=str, default='data',
                        help='╨Ъ╨╛╤А╨╡╨╜╤М ╨┤╨░╤В╨░╤Б╨╡╤В╨░')
    parser.add_argument('--test-csv', type=str, default='stage1/test.csv',
                        help='╨Я╤Г╤В╤М ╨║ test.csv ╨╛╤В╨╜╨╛╤Б╨╕╤В╨╡╨╗╤М╨╜╨╛ data_dir')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/best.pth',
                        help='╨Я╤Г╤В╤М ╨║ ╤З╨╡╨║╨┐╨╛╨╕╨╜╤В╤Г')

    # ╨Ь╨╛╨┤╨╡╨╗╤М
    parser.add_argument('--architecture', type=str, default='unet',
                        choices=['unet', 'unetplusplus',
                                 'deeplabv3plus', 'fpn', 'manet'],
                        help='╨Р╤А╤Е╨╕╤В╨╡╨║╤В╤Г╤А╨░')
    parser.add_argument('--encoder', type=str, default='resnet34',
                        help='╨н╨╜╨║╨╛╨┤╨╡╤А')

    # ╨Ш╨╜╤Д╨╡╤А╨╡╨╜╤Б
    parser.add_argument('--img-size', type=int, default=384,
                        help='╨а╨░╨╖╨╝╨╡╤А ╨▓╤Е╨╛╨┤╨░ ╨╝╨╛╨┤╨╡╨╗╨╕')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='╨а╨░╨╖╨╝╨╡╤А ╨▒╨░╤В╤З╨░')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='╨Т╨╛╤А╨║╨╡╤А╤Л DataLoader')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='╨Я╨╛╤А╨╛╨│ ╨▒╨╕╨╜╨░╤А╨╕╨╖╨░╤Ж╨╕╨╕ (0..1)')
    parser.add_argument('--min-area-ratio', type=float, default=0.001,
                        help='╨Ь╨╕╨╜. ╨┐╨╗╨╛╤Й╨░╨┤╤М ╨║╨╛╨╝╨┐╨╛╨╜╨╡╨╜╤В╤Л ╨┤╨╗╤П clean_mask')

    # ╨Т╤Л╤Е╨╛╨┤
    parser.add_argument('--pred-dir', type=str, default='predictions',
                        help='╨Я╨░╨┐╨║╨░ ╨┤╨╗╤П PNG-╨╝╨░╤Б╨╛╨║')
    parser.add_argument('--submission-csv', type=str,
                        default='submission.csv',
                        help='╨Я╤Г╤В╤М ╨║ submission.csv')

    args = parser.parse_args()

    # ╨Т╨░╨╗╨╕╨┤╨░╤Ж╨╕╤П
    if not 0.0 < args.threshold < 1.0:
        raise ValueError(
            f'--threshold ╨┤╨╛╨╗╨╢╨╡╨╜ ╨▒╤Л╤В╤М ╨▓ (0, 1), ╨┐╨╛╨╗╤Г╤З╨╡╨╜╨╛ {args.threshold}'
        )
    if args.min_area_ratio < 0:
        raise ValueError(
            f'--min-area-ratio >= 0, ╨┐╨╛╨╗╤Г╤З╨╡╨╜╨╛ {args.min_area_ratio}'
        )
    if args.img_size <= 0:
        raise ValueError(
            f'--img-size > 0, ╨┐╨╛╨╗╤Г╤З╨╡╨╜╨╛ {args.img_size}'
        )

    run_predict(args)


if __name__ == '__main__':
    main()