"""
Создаёт случайный чекпоинт для тестирования predict.py.
"""
import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=str, default='unet')
    parser.add_argument('--encoder', type=str, default='resnet34')
    parser.add_argument('--out', type=str, default='checkpoints/random.pth')
    args = parser.parse_args()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'🏗️  Собираю модель: {args.architecture} + {args.encoder}')
    model = build_model(
        architecture=args.architecture,
        encoder_name=args.encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'   Параметров: {n_params:.2f} M')

    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': 0,
        'best_aic': 0.0,
        'architecture': args.architecture,
        'encoder': args.encoder,
    }, out_path)

    print(f'✅ Сохранено: {out_path}')


if __name__ == '__main__':
    main()