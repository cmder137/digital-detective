import logging
from typing import Literal, Optional

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# ============================================================
# РЕЕСТР ПОДДЕРЖИВАЕМЫХ АРХИТЕКТУР
# ============================================================
ARCH_REGISTRY = {
    'unet': smp.Unet,
    'unetplusplus': smp.UnetPlusPlus,
    'deeplabv3plus': smp.DeepLabV3Plus,
    'fpn': smp.FPN,
    'manet': smp.MAnet,
}

# Строгая типизация для IDE
ArchitectureType = Literal['unet', 'unetplusplus', 'deeplabv3plus', 'fpn', 'manet']


def build_model(
    architecture: ArchitectureType = 'unet',
    encoder_name: str = 'resnet34',
    encoder_weights: Optional[str] = 'imagenet',
    in_channels: int = 3,
    classes: int = 1,
) -> nn.Module:
    """
    Собирает модель сегментации с использованием библиотеки SMP.
    
    ВНИМАНИЕ: Модель возвращает сырые логиты (raw logits) без финальной 
    активации (Sigmoid/Softmax). Это необходимо для стабильной работы 
    BCEWithLogitsLoss в процессе обучения.

    Args:
        architecture: Архитектура из реестра (unet, deeplabv3plus и т.д.)
        encoder_name: Имя энкодера (напр., 'resnet34' - отлично вписывается в 100 GFLOPs)
        encoder_weights: Веса для претрейна ('imagenet' или None)
        in_channels: Количество входных каналов (3 для RGB)
        classes: Количество выходных классов (1 для бинарной маски)

    Returns:
        Модель PyTorch (nn.Module). Возвращается на CPU.
    """
    architecture = architecture.lower()
    
    if architecture not in ARCH_REGISTRY:
        raise ValueError(
            f"Архитектура '{architecture}' не поддерживается. "
            f"Доступные: {list(ARCH_REGISTRY.keys())}"
        )

    # Предупреждение о конфликте каналов и весов
    if in_channels != 3 and encoder_weights is not None:
        logging.warning(
            f"ВНИМАНИЕ: in_channels={in_channels}, но запрашиваются веса '{encoder_weights}'. "
            f"SMP попытается адаптировать веса, но это может повлиять на сходимость. "
            f"Для безопасной инициализации с нуля передайте encoder_weights=None."
        )

    # Динамическая инициализация из реестра
    ModelClass = ARCH_REGISTRY[architecture]
    
    model = ModelClass(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )
    
    return model


if __name__ == "__main__":
    # Sanity-check и профилирование (вызывается через python -m src.model)
    from torch.utils.flop_counter import FlopCounterMode
    
    print("Собираем базовую модель UNet (ResNet34)...")
    
    # Модель создается на CPU (как положено)
    model = build_model(
        architecture='unet', 
        encoder_name='resnet34',
        encoder_weights=None # Для тестов скачивать веса не нужно
    ).eval()
    
    # Эмулируем 1 картинку 256x256 (один инференс)
    dummy_input = torch.randn(1, 3, 256, 256)
    
    print(f"Размерность входа: {dummy_input.shape}")
    
    # Используем встроенный в PyTorch 2+ профилировщик
    with FlopCounterMode(model, display=False) as flop_counter:
        output = model(dummy_input)
        
    # Организаторы пишут: 100 GFLOPs = 50 GMACs. 
    # FlopCounterMode считает строгие FLOPs, но иногда его путают с MACs. 
    # В PyTorch 2 FlopCounterMode выдает именно FLOPs, так что мы просто берем это число.
    total_flops = flop_counter.get_total_flops()
    gflops = total_flops / 1e9
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    
    assert output.shape == (1, 1, 256, 256), f"Неверная размерность выхода: {output.shape}"
    
    print(f"✅ Модель успешно собрана!")
    print(f"Выходная размерность: {output.shape}")
    print(f"Параметры: {params_m:.2f} M")
    print(f"Строгие GFLOPs: {gflops:.2f} (Лимит: 100.00 GFLOPs)")
    
    if gflops > 100:
        print("❌ КРИТИЧЕСКИ: Превышен лимит GFLOPs для одного изображения!")
    else:
        print("✅ Лимит GFLOPs соблюден с запасом.")