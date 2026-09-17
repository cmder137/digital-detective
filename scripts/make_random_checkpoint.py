"""
Создаёт случайный чекпоинт для тестирования predict.py.

Нужен, чтобы протестировать пайплайн инференса ДО того, как ML Engineer
обучит модель. Модель будет выдавать случайные маски, но формат submission,
размеры PNG и корректность путей проверятся полностью.

Запуск:
    python scripts/make_random_checkpoint.py
    python scripts/make_random_checkpoint.py --architecture unet --encoder resnet34
    python scripts/make_random_checkpoint.py --out checkpoints/random.pth
"""
import argparse
import sys
from pathlib import Path

import torch

# Добавляем корень проекта в sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import build_model


def main():
    parser = argparse.ArgumentParser(
        description='Создаёт случайный чекпоинт (без обучения)'
    )
    parser.add_argument('--architecture', type=str, default='unet',
                        help='unet / deeplabv3plus / fpn / ...')
    parser.add_argument('--encoder', type=str, default='resnet34',
                        help='resnet34 / efficientnet-b2 / ...')
    parser.add_argument('--out', type=str, default='checkpoints/random.pth',
                        help='Куда сохранить чекпоинт')
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'🏗️  Собираю модель: {args.architecture} + {args.encoder}')
    model = build_model(
        architecture=args.architecture,
        encoder_name=args.encoder,
        encoder_weights=None,  # случайные веса
        in_channels=3,
        classes=1,
    )

    # Считаем параметры для информации
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
    print()
    print('Теперь можно запустить:')
    print(f'  python -m src.predict --checkpoint {out_path}')


if __name__ == '__main__':
    main()