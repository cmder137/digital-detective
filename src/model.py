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
        activation=None, 
    )
    
    return model


if __name__ == "__main__":
    print("Проверка модели и замер ресурсов...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")

    # Изолированный импорт профилировщика
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError as exc:
        raise RuntimeError("FlopCounterMode недоступен. Проверьте версию PyTorch.") from exc

    # Параметры бенчмарка
    test_size = 256
    test_in_channels = 3
    test_classes = 1

    dummy_input = torch.randn(1, test_in_channels, test_size, test_size, device=device)

    model = build_model(
        architecture='unet', 
        encoder_name='resnet34',
        encoder_weights=None,
        in_channels=test_in_channels,
        classes=test_classes
    ).to(device).eval()

    # --- 1. Замер FLOPs (с защитой сигнатуры) ---
    try:
        flop_ctx = FlopCounterMode(model, display=False)
    except TypeError:
        flop_ctx = FlopCounterMode(display=False)

    with flop_ctx as flop_counter, torch.no_grad():
        output = model(dummy_input)
            
    total_flops = flop_counter.get_total_flops()
    if total_flops <= 0:
        raise RuntimeError("FlopCounterMode вернул <= 0. Проверьте совместимость PyTorch.")
        
    gflops = total_flops / 1e9
    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    # --- 2. Замер задержки (Latency - per request) ---
    warmup_iters = 50 if device.type == 'cuda' else 20
    iters = 200
    latencies_ms = []

    with torch.inference_mode():
        # Прогрев
        for _ in range(warmup_iters):
            model(dummy_input)
            
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            
        # Основной цикл
        for _ in range(iters):
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            
            t0 = time.perf_counter()
            model(dummy_input)
            
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                
            latencies_ms.append((time.perf_counter() - t0) * 1000)

    latency_median = statistics.median(latencies_ms)
    latency_p95 = sorted(latencies_ms)[min(int(0.95 * len(latencies_ms)), len(latencies_ms) - 1)]
    latency_max = max(latencies_ms)

    # --- 3. Итоги и строгие проверки ---
    print(f"\nРазмер входа: {test_size}x{test_size}x{test_in_channels}")
    print(f"Параметры:    {params_m:.2f} M")
    print(f"GFLOPs:       {gflops:.2f} (Лимит: <= 100.00)")
    print("ℹ️  FLOPs посчитаны через FlopCounterMode (строгие FLOPs, 1 MAC = 2 FLOPs).")
    print("   Согласно правилам конкурса, этот инструмент является арбитром.\n")
    
    print(f"Latency Med:  {latency_median:.2f} мс (Лимит: <= 50.00 мс)")
    print(f"Latency p95:  {latency_p95:.2f} мс")
    print(f"Latency Max:  {latency_max:.2f} мс")
    print("⚠️  Локальный замер. Тестирование будет на H100 (быстрее в ~2-3 раза).\n")

    if output.shape != (1, test_classes, test_size, test_size):
        raise RuntimeError(f"Неверная форма выхода: {output.shape}")

    if gflops > 100.0:
        raise RuntimeError(f"Превышен лимит по GFLOPs: {gflops:.2f} > 100.00")
        
    if latency_median > 50.0:
        raise RuntimeError(f"Превышен лимит латентности по медиане: {latency_median:.2f} мс > 50.00")
        
    if latency_max > 50.0:
        print(f"⚠️ ВНИМАНИЕ: max ({latency_max:.2f} мс) выше 50 мс.")
        print("   На H100, вероятно, пройдёт, но рекомендуется оптимизация или замер на целевом железе.")
    else:
        print("\n✅ Все лимиты соревнования успешно соблюдены.")