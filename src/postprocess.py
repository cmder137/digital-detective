"""
Постобработка предсказанных масок.

Задача: удалить мелкие «шумовые» пятна из бинарной маски.
Зачем: метрика AIC штрафует, если на negative-картинке (без подделки)
       модель закрасит >= 1% пикселей. Мелкий шум легко превышает этот порог,
       поэтому перед сохранением маски нужно «чистить» предсказание.

Метрика FPR в соревновании:
    Изображение считается ложной тревогой, если площадь маски >= 1% от картинки.
    Значит, всё, что меньше 1% площади, нужно удалять.

Мы используем cv2.connectedComponentsWithStats — он находит связные
компоненты (белые пятна) и их площади. Компоненты меньше порога удаляем.
"""
import cv2
import numpy as np


def clean_mask(
    mask: np.ndarray,
    min_area_ratio: float = 0.001,
    connectivity: int = 8,
) -> np.ndarray:
    """
    Удаляет мелкие связные компоненты из бинарной маски.

    Args:
        mask: np.ndarray (H, W) — бинарная маска.
              Значения: 0 / 1, 0 / 255, bool или float — всё приведётся к 0/1.
        min_area_ratio: минимальная площадь компоненты как доля от площади
                        картинки. Всё, что меньше — удаляется.
                        Default 0.001 = 0.1% площади.
                        Для борьбы с FPR в соревновании можно поставить 0.005–0.01 (0.5–1%).
        connectivity: 4 или 8. 8 — «диагональные» пиксели считаются соседями,
                      склеивает компоненты агрессивнее. Default 8.

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
        # если 0/255 или 0..1 float — нормализуем
        m = (m > 127).astype(np.uint8) if m.max() > 1 else (m > 0.5).astype(np.uint8)
    else:
        m = m.astype(np.uint8)

    h, w = m.shape[:2]
    min_area = int(h * w * min_area_ratio)

    # Если нечего удалять — выходим сразу
    if m.sum() == 0:
        return m

    # Находим связные компоненты
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        m, connectivity=connectivity
    )

    # Компонента 0 — это фон (чёрный). Начинаем с 1.
    clean = np.zeros_like(m, dtype=np.uint8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean[labels == i] = 1

    return clean


def clean_mask_strict(mask: np.ndarray) -> np.ndarray:
    """
    Более агрессивная чистка для соревнования: удаляет всё, что < 1%
    площади картинки. Соответствует порогу FPR из условий AIC.
    """
    return clean_mask(mask, min_area_ratio=0.01)


def clean_mask_soft(mask: np.ndarray) -> np.ndarray:
    """
    Мягкая чистка: удаляет только «мусор» < 0.1% площади.
    Полезна, если модель в целом аккуратная и нужно убрать одиночные пиксели.
    """
    return clean_mask(mask, min_area_ratio=0.001)


# ============================================================
# ТЕСТЫ
# ============================================================
if __name__ == '__main__':
    import time

    print('🧪 Тесты clean_mask()')
    print('=' * 60)

    # --- Тест 1: Маска с крупным объектом и мелким шумом ---
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[50:150, 50:150] = 1        # крупный объект: 100*100 = 10 000 px (15% площади)
    mask[200:201, 200:201] = 1      # шум: 1 px
    mask[210:212, 210:212] = 1      # шум: 4 px

    cleaned = clean_mask(mask, min_area_ratio=0.001)
    print(f'Тест 1 — крупный объект + 2 шумовых пятна:')
    print(f'  До:    {mask.sum():>6} пикселей')
    print(f'  После: {cleaned.sum():>6} пикселей')
    print(f'  Удалено шума: {mask.sum() - cleaned.sum()} пикселей')
    assert cleaned.sum() == 10_000, f'Ожидалось 10000, получили {cleaned.sum()}'
    print('  ✅ Прошло')

    # --- Тест 2: Маска из одного мелкого пятна (должна стать пустой) ---
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[100:101, 100:101] = 1      # 1 пиксель
    cleaned = clean_mask(mask, min_area_ratio=0.001)
    print(f'\nТест 2 — только одно мелкое пятно:')
    print(f'  До:    {mask.sum()} пикселей')
    print(f'  После: {cleaned.sum()} пикселей')
    assert cleaned.sum() == 0, 'Пятно должно быть удалено'
    print('  ✅ Прошло (пятно удалено)')

    # --- Тест 3: Пустая маска ---
    mask = np.zeros((256, 256), dtype=np.uint8)
    cleaned = clean_mask(mask)
    print(f'\nТест 3 — пустая маска:')
    assert cleaned.sum() == 0
    assert cleaned.shape == mask.shape
    assert cleaned.dtype == np.uint8
    print('  ✅ Прошло')

    # --- Тест 4: Маска из 0/255 (как PNG) ---
    mask_255 = np.zeros((256, 256), dtype=np.uint8)
    mask_255[50:150, 50:150] = 255
    mask_255[200:201, 200:201] = 255
    cleaned = clean_mask(mask_255, min_area_ratio=0.001)
    print(f'\nТест 4 — маска со значениями 0/255:')
    print(f'  После: {cleaned.sum()} пикселей (ожидалось 10000)')
    assert cleaned.sum() == 10_000
    assert cleaned.max() == 1, 'На выходе должны быть 0/1'
    print('  ✅ Прошло')

    # --- Тест 5: Boolean маска ---
    mask_bool = np.zeros((256, 256), dtype=bool)
    mask_bool[50:150, 50:150] = True
    mask_bool[200:201, 200:201] = True
    cleaned = clean_mask(mask_bool)
    print(f'\nТест 5 — boolean маска:')
    assert cleaned.sum() == 10_000
    print('  ✅ Прошло')

    # --- Тест 6: Площадь ровно 1% (граничный случай FPR) ---
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[0:10, 0:10] = 1            # 100 px = 1% от 10 000
    cleaned_strict = clean_mask(mask, min_area_ratio=0.01)
    cleaned_soft = clean_mask(mask, min_area_ratio=0.001)
    print(f'\nТест 6 — площадь ровно 1%:')
    print(f'  min_area_ratio=0.01 → после: {cleaned_strict.sum()} (граница, должен остаться)')
    print(f'  min_area_ratio=0.001 → после: {cleaned_soft.sum()}')
    # 100 >= 100*100*0.01 = 100 → остаётся
    assert cleaned_strict.sum() == 100
    print('  ✅ Прошло')

    # --- Тест 7: Скорость ---
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    mask[100:500, 100:500] = 1
    for _ in range(100):
        mask[np.random.randint(0, 1024), np.random.randint(0, 1024)] = 1

    t0 = time.perf_counter()
    for _ in range(10):
        _ = clean_mask(mask, min_area_ratio=0.001)
    elapsed = (time.perf_counter() - t0) / 10 * 1000
    print(f'\nТест 7 — скорость на 1024×1024:')
    print(f'  Время: {elapsed:.1f} мс на картинку')
    assert elapsed < 50, 'Слишком медленно (лимит 50 мс)'
    print('  ✅ Прошло (< 50 мс)')

    print('\n' + '=' * 60)
    print('🎉 Все тесты пройдены!')