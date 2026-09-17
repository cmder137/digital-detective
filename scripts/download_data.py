"""
Скачивает датасет AI Challenge 2026 с Яндекс.Диска.

Что делает
----------
1. Получает прямую ссылку на файл через публичный API Яндекс.Диска.
2. Скачивает архив в папку data/ БЕЗ распаковки.
3. Поддерживает докачку (если скачивание оборвалось).

Subsets
-------
- train       → train_stage1.zip (~39 ГБ)
- test        → test_stage1.zip (~592 МБ)
- nanobanana  → nanobanana.zip (~16.9 ГБ)
- all         → все три архива

Почему не распаковываем
-----------------------
Пользователь может распаковать архив сам:
  Windows: правой кнопкой → «Извлечь всё»
  Python:  python -c "import zipfile; zipfile.ZipFile('data/train_stage1.zip').extractall('data/')"

Это удобнее: сначала скачал, проверил размер, потом распаковал.

Запуск
------
    # Скачать train (39 ГБ)
    python scripts/download_data.py --data-dir data --subset train

    # Скачать test (592 МБ)
    python scripts/download_data.py --data-dir data --subset test

    # Всё
    python scripts/download_data.py --data-dir data --subset all
"""
import argparse
import sys
from pathlib import Path

import requests
from tqdm import tqdm

# Добавляем корень проекта в sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================
# Публичные ссылки Яндекс.Диска (от организаторов)
YANDEX_LINKS = {
    'train': {
        'url': 'https://disk.360.yandex.ru/d/VmpzG-QnNnwBAw',
        'filename': 'train_stage1.zip',
        'size_gb': 39.0,
    },
    'test': {
        'url': 'https://disk.360.yandex.ru/d/kSp6umzs1IMOBA',
        'filename': 'test_stage1.zip',
        'size_gb': 0.6,
    },
    'nanobanana': {
        'url': 'https://disk.360.yandex.ru/d/b7q_0Q5zhnOAqA',
        'filename': 'nanobanana.zip',
        'size_gb': 16.9,
    },
}

# API Яндекс.Диска для получения прямой ссылки
YANDEX_API = 'https://cloud-api.yandex.net/v1/disk/public/resources/download'


# ============================================================
# ПОЛУЧЕНИЕ ПРЯМОЙ ССЫЛКИ
# ============================================================
def get_download_href(public_key: str) -> str:
    """
    Получает прямую ссылку на скачивание файла с Яндекс.Диска.

    Args:
        public_key: публичная ссылка вида https://disk.yandex.ru/d/XXXXX

    Returns:
        Прямая ссылка для скачивания (временная, действует ~несколько часов).

    Raises:
        RuntimeError: если не удалось получить ссылку.
    """
    params = {'public_key': public_key}
    try:
        resp = requests.get(YANDEX_API, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            f'Не удалось обратиться к API Яндекс.Диска: {e}\n'
            f'Проверь интернет или ссылку.'
        ) from e

    data = resp.json()

    if 'href' not in data:
        raise RuntimeError(
            f'API не вернул ссылку на скачивание. Ответ: {data}\n'
            f'Возможно, ссылка недействительна или файл удалён.'
        )

    return data['href']


# ============================================================
# СКАЧИВАНИЕ С ДОКАЧКОЙ
# ============================================================
def download_file(url: str, out_path: Path, chunk_size: int = 1024 * 1024):
    """
    Скачивает файл с поддержкой докачки.

    Если файл уже есть — докачивает с того места, где остановился.

    Args:
        url: прямая ссылка на файл.
        out_path: куда сохранять.
        chunk_size: размер чанка (по умолчанию 1 МБ).
    """
    headers = {}
    mode = 'wb'
    downloaded = 0

    # Если файл существует — докачиваем
    if out_path.exists():
        downloaded = out_path.stat().st_size
        if downloaded > 0:
            headers['Range'] = f'bytes={downloaded}-'
            mode = 'ab'
            print(f'   Докачка с {downloaded / 1e9:.2f} ГБ')

    try:
        with requests.get(url, headers=headers, stream=True, timeout=60) as r:
            r.raise_for_status()

            # Общий размер: Content-Length + уже скачанное
            content_length = int(r.headers.get('Content-Length', 0))
            total = content_length + downloaded

            with open(out_path, mode) as f, tqdm(
                total=total,
                initial=downloaded,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=out_path.name,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

    except requests.RequestException as e:
        raise RuntimeError(f'Ошибка скачивания {out_path.name}: {e}') from e


# ============================================================
# ГЛАВНАЯ ЛОГИКА
# ============================================================
def download(data_dir: Path, subset: str):
    """
    Скачивает указанный subset датасета.

    Args:
        data_dir: куда сохранять архивы.
        subset: train / test / nanobanana / all.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Определяем, что качать
    if subset == 'all':
        keys = ['train', 'test', 'nanobanana']
    else:
        keys = [subset]

    for key in keys:
        info = YANDEX_LINKS[key]
        out_path = data_dir / info['filename']

        print('=' * 60)
        print(f'📥 Скачиваю: {info["filename"]} (~{info["size_gb"]} ГБ)')
        print(f'   Источник: {info["url"]}')
        print(f'   Сохраняю: {out_path}')
        print('=' * 60)

        # Если файл уже полностью скачан — пропускаем
        if out_path.exists():
            size_gb = out_path.stat().st_size / 1e9
            if size_gb >= info['size_gb'] * 0.95:  # допуск 5%
                print(f'   ✅ Файл уже есть: {size_gb:.2f} ГБ')
                print('   Пропускаю. Удали вручную, если хочешь перекачать.')
                print()
                continue
            else:
                print(f'   ⚠️  Файл есть, но неполный ({size_gb:.2f} ГБ). Докачиваю.')

        # Получаем прямую ссылку
        try:
            href = get_download_href(info['url'])
        except RuntimeError as e:
            print(f'   ❌ {e}')
            raise SystemExit(1)

        # Скачиваем
        try:
            download_file(href, out_path)
        except RuntimeError as e:
            print(f'   ❌ {e}')
            raise SystemExit(1)

        final_size = out_path.stat().st_size / 1e9
        print(f'   ✅ Сохранено: {out_path} ({final_size:.2f} ГБ)')
        print()

    print('=' * 60)
    print('✅ Готово!')
    print()
    print('📦 Распакуй архивы:')
    print('   Windows: правой кнопкой на .zip → «Извлечь всё» → в data/')
    print('   Python:  python -c "import zipfile; '
          'zipfile.ZipFile(\'data/train_stage1.zip\').extractall(\'data/\')"')
    print('=' * 60)


# ============================================================
# ТОЧКА ВХОДА
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Скачивает датасет AI Challenge 2026 с Яндекс.Диска'
    )
    parser.add_argument(
        '--data-dir', type=str, default='data',
        help='Куда сохранять архивы (по умолчанию data/)',
    )
    parser.add_argument(
        '--subset', type=str, default='train',
        choices=['train', 'test', 'nanobanana', 'all'],
        help='Что скачивать (default: train)',
    )
    args = parser.parse_args()

    download(Path(args.data_dir), args.subset)


if __name__ == '__main__':
    main()