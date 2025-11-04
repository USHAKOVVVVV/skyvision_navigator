import requests
import os
from PIL import Image
import math
import time
from io import BytesIO
def correct_latlon_to_tile(lat, lon, zoom):
    """Правильное преобразование координат в тайлы для Яндекс Карт"""
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y_standard = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    y_correct = y_standard + 231
    return x, y_correct
def download_yandex_screenshot_quality():
    """
    Эмуляция скриншотов Яндекс Карт в максимальном качестве
    """
    
    output_dir = "yandex_screenshots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Основные точки Пушкино - УВЕЛИЧИВАЕМ количество точек
    puskino_spots = [
        # Центральные районы
        # Лесные массивы вокруг Климовска
    
   (55.9742, 37.9206), (55.9735, 37.9180), (55.9720, 37.9150),
    (55.9800, 37.9100), (55.9700, 37.9300), (55.9850, 37.9150),
    (55.9650, 37.9250), (55.9750, 37.9400), (55.9770, 37.9250),
    (55.9680, 37.9180), (55.9820, 37.9120), (55.9600, 37.9280),
    (55.9750, 37.9160), (55.9710, 37.9220), (55.9765, 37.9140),
    (55.9740, 37.9190), (55.9725, 37.9230), (55.9760, 37.9170),
    (55.9700, 37.9200), (55.9780, 37.9130), (55.9770, 37.9200),
    (55.9690, 37.9250)
]
 
    zoom = 18
    counter = 1
    
    print("🛰️ Эмулируем скриншоты Яндекс Карт...")
    
    for lat, lon, in puskino_spots:
        if counter > 100:  # МЕНЯЕМ с 300 на 100
            break
            
        # Получаем центральный тайл
        center_x, center_y = correct_latlon_to_tile(lat, lon, zoom)
        
        # Для каждой точки делаем несколько "скриншотов" со смещением
        for shot_num in range(4):  # УВЕЛИЧИВАЕМ с 3 до 4 скриншотов с каждой точки
            if counter > 120:
                break
                
            # Смещаем центр для разнообразия
            offset_x = center_x + (shot_num % 3) - 1
            offset_y = center_y + (shot_num // 3) - 1
            
            # Создаем "скриншот" из 4 тайлов (2x2) для лучшего качества
            composite_size = 512
            final_size = 640
            composite = Image.new('RGB', (composite_size * 2, composite_size * 2))
            
            tiles_downloaded = 0
            
            # Скачиваем 4 тайла вокруг точки
            for i in range(2):
                for j in range(2):
                    tile_x = offset_x + i
                    tile_y = offset_y + j
                    
                    # Пробуем разные URL для максимального качества
                    urls = [
                        f"https://core-sat.maps.yandex.net/tiles?l=sat&x={tile_x}&y={tile_y}&z={zoom}&scale=1&lang=ru_RU",
                        f"https://sat0{(tile_x + tile_y) % 4 + 1}.maps.yandex.net/tiles?l=sat&x={tile_x}&y={tile_y}&z={zoom}",
                    ]
                    
                    for url in urls:
                        try:
                            response = requests.get(url, headers={
                                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                                'Accept': 'image/webp,image/avif,image/*,*/*;q=0.8',
                                'Accept-Language': 'ru-RU,ru;q=0.9,en;q=0.8',
                                'Referer': 'https://yandex.ru/maps/'
                            }, timeout=10)
                            
                            if response.status_code == 200:
                                tile_img = Image.open(BytesIO(response.content))
                                
                                # Проверяем что тайл валидный
                                if tile_img.size[0] >= 256 and tile_img.size[1] >= 256:
                                    # Ресайзим к стандартному размеру если нужно
                                    if tile_img.size != (composite_size, composite_size):
                                        tile_img = tile_img.resize((composite_size, composite_size), Image.Resampling.LANCZOS)
                                    
                                    # Вставляем в композит
                                    composite.paste(tile_img, (i * composite_size, j * composite_size))
                                    tiles_downloaded += 1
                                    break
                                    
                        except Exception as e:
                            continue
                    
                    time.sleep(0.1)
            
            # Если скачали достаточно тайлов, сохраняем "скриншот"
            if tiles_downloaded >= 3:
                if composite.size != (final_size, final_size):
                    composite = composite.resize((final_size, final_size), Image.Resampling.LANCZOS)
                
                filename = f"ivanteevka-{counter}.jpg"
                filepath = os.path.join(output_dir, filename)
                
                composite.save(filepath, 'JPEG', 
                             quality=95,
                             optimize=True,
                             subsampling=0,
                             dpi=(300, 300))
                
                print(f"✅ Снимок {counter:3d}:  {shot_num} ({tiles_downloaded}/4 тайлов)")
                counter += 1
            
            time.sleep(0.3)
    
    print(f"\n🎉 Готово! Создано {counter-1} скриншотов в качестве 640x640")

if __name__ == "__main__":
    # Основной метод - эмуляция скриншотов
    download_yandex_screenshot_quality()
    
    # Если нужно больше фото, запусти второй метод
    # alternative_screenshot_method()
