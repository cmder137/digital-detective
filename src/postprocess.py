"""
Постобработка предсказанных масок.

Задача: удалить мелкие «шумовые» пятна из бинарной маски.
Зачем: метрика AIC штрафует, если на negative-картинке (без подделки)
       модель закрасит >= 1% пикселей. Мелкий шум легко превышает этот порог.

ВАЖНО: min_area_ratio должен быть МАЛЕНЬКИМ (0.001 = 0.1%).
       В датасете есть реальные маски 0.31% площади — их нельзя удалять.
       Для борьбы с FPR достаточно убрать шум < 0.1%.

Метрика FPR в соревновании:
    Изображение считается ложной тревогой, если площадь маски >= 1% от картинки.
"""
import cv2
import numpy as np


def clean_mask(
    mask: np.ndarray,
    min_area_ratio: float = 0.001,
    connectivity: int = 8,
    bin_threshold: float = 0.5,
) -> np.ndarray:
    """
    Удаляет мелкие связные компоненты из бинарной маски.

    Args:
        mask: np.ndarray (H, W) — маска.
              Поддерживаются форматы:
                - bool
                - uint8 0/1
                - uint8 0/255 (PNG)
                - float32/float64 [0, 1] (вероятности модели)
        min_area_ratio: минимальная площадь компоненты как доля от площади
                        картинки. Default 0.001 = 0.1%.
                        НЕ ставь 0.01 — удалит реальные мелкие подделки!
        connectivity: 4 или 8. 8 — диагональные пиксели считаются соседями.
        bin_threshold: порог бинаризации для float-масок в шкале [0, 1].
                       Должен совпадать с threshold из predict.py.

    Returns:
        np.ndarray (H, W) uint8 — очищенная маска со значениями 0 / 1.
    """
    if mask is None:
        return None

    # Приводим к uint8 0/1 независимо от входа
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[:, :, 0]  # если пришло (H, W, 1)

    if m.dtype == bool:
        m = m.astype(np.uint8)
    elif m.max() > 1:
        # Маска в шкале 0/255 (например, PNG)
        m = (m > 127).astype(np.uint8)
    else:
        # Float-маска в шкале [0, 1] (вероятности модели)
        m = (m > bin_threshold).astype(np.uint8)

    h, w = m.shape[:2]
    min_area = int(h * w * min_area_ratio)

    # Если нечего удалять — выходим сразу
    if m.sum() == 0:
        return m

    # Находим связные компоненты
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        m, connectivity=connectivity
    )

    # Компонента 0 — фон. Начинаем с 1.
    clean = np.zeros_like(m, dtype=np.uint8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean[labels == i] = 1

    return clean


def clean_mask_soft(mask: np.ndarray) -> np.ndarray:
    """
    Мягкая чистка: удаляет только «мусор» < 0.05% площади.
    Используй, если модель аккуратная и есть риск удалить мелкие подделки.
    """
    return clean_mask(mask, min_area_ratio=0.0005)


def clean_mask_default(mask: np.ndarray) -> np.ndarray:
    """
    Стандартная чистка: удаляет компоненты < 0.1% площади.
    Баланс между удалением шума и сохранением реальных мелких масок.
    """
    return clean_mask(mask, min_area_ratio=0.001)


# ============================================================
# ТЕСТЫ
# ============================================================
if __name__ == '__main__':
    import time

    print('🧪 Тесты clean_mask()')
    print('=' * 60)

    # --- Тест 1: Крупный объект + мелкий шум ---
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[50:150, 50:150] = 1        # 10000 px
    mask[200:201, 200:201] = 1      # 1 px
    mask[210:212, 210:212] = 1      # 4 px

    cleaned = clean_mask(mask, min_area_ratio=0.001)
    print(f'Тест 1 — крупный объект + шум:')
    print(f'  До: {mask.sum()}, После: {cleaned.sum()}')
    assert cleaned.sum() == 10_000
    print('  ✅ Прошло')

    # --- Тест 2: Только мелкое пятно ---
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[100:101, 100:101] = 1
    cleaned = clean_mask(mask, min_area_ratio=0.001)
    assert cleaned.sum() == 0
    print(f'Тест 2 — только мелкое пятно: ✅ Прошло')

    # --- Тест 3: Пустая маска ---
    mask = np.zeros((256, 256), dtype=np.uint8)
    cleaned = clean_mask(mask)
    assert cleaned.sum() == 0 and cleaned.dtype == np.uint8
    print(f'Тест 3 — пустая маска: ✅ Прошло')

    # --- Тест 4: 0/255 ---
    mask_255 = np.zeros((256, 256), dtype=np.uint8)
    mask_255[50:150, 50:150] = 255
    mask_255[200:201, 200:201] = 255
    cleaned = clean_mask(mask_255, min_area_ratio=0.001)
    assert cleaned.sum() == 10_000 and cleaned.max() == 1
    print(f'Тест 4 — 0/255: ✅ Прошло')

    # --- Тест 5: bool ---
    mask_bool = np.zeros((256, 256), dtype=bool)
    mask_bool[50:150, 50:150] = True
    cleaned = clean_mask(mask_bool)
    assert cleaned.sum() == 10_000
    print(f'Тест 5 — bool: ✅ Прошло')

    # --- Тест 6: граница FPR (1%) ---
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[0:10, 0:10] = 1  # 100 px = 1% от 10 000
    cleaned = clean_mask(mask, min_area_ratio=0.001)
    assert cleaned.sum() == 100
    print(f'Тест 6 — граница FPR: ✅ Прошло')

    # --- Тест 7: скорость ---
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    mask[100:500, 100:500] = 1
    t0 = time.perf_counter()
    for _ in range(10):
        _ = clean_mask(mask, min_area_ratio=0.001)
    elapsed = (time.perf_counter() - t0) / 10 * 1000
    print(f'Тест 7 — скорость: {elapsed:.1f} мс')
    assert elapsed < 50
    print('  ✅ Прошло (< 50 мс)')

    # --- Тест 8: float-маска [0, 1] ---
    mask_float = np.zeros((256, 256), dtype=np.float32)
    mask_float[50:150, 50:150] = 0.9
    mask_float[200:201, 200:201] = 0.8
    cleaned = clean_mask(mask_float, min_area_ratio=0.001)
    print(f'Тест 8 — float [0, 1]: После {cleaned.sum()} (ожидалось 10000)')
    assert cleaned.sum() == 10_000
    print('  ✅ Прошло (bin_threshold сработал)')

    # --- Тест 9: float-маска с bin_threshold=0.7 ---
    mask_float = np.zeros((256, 256), dtype=np.float32)
    mask_float[50:150, 50:150] = 0.9    # пройдёт
    mask_float[180:220, 180:220] = 0.6  # НЕ пройдёт при threshold=0.7
    cleaned = clean_mask(mask_float, min_area_ratio=0.001, bin_threshold=0.7)
    print(f'Тест 9 — bin_threshold=0.7: После {cleaned.sum()} (ожидалось 10000)')
    assert cleaned.sum() == 10_000
    print('  ✅ Прошло')

    print('\n' + '=' * 60)
    print('🎉 Все тесты пройдены!')