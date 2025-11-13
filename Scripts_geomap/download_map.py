55.960076, 37.959420
import requests
import math
import os
from PIL import Image
import time

def correct_latlon_to_tile(lat, lon, zoom):
    """Правильное преобразование координат в тайлы для Яндекс Карт"""
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y_standard = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    y_correct = y_standard + 231  # Найденная разница
    return x, y_correct

def download_tile(x, y, zoom):
    """Скачивает один тайл"""
    url = f"https://core-sat.maps.yandex.net/tiles?l=sat&x={x}&y={y}&z={zoom}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://yandex.ru/maps/'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200 and len(response.content) > 5000:
            return response.content
    except:
        pass
    return None

def create_output_filename(lat1, lon1, lat2, lon2, zoom):
    """Создает имя файла на основе координат"""
    # Создаем папку output_img если её нет
    os.makedirs('output_img', exist_ok=True)
    
    # Формируем строку с координатами
    coords_str = f"map_{lat1:.6f}_{lon1:.6f}_to_{lat2:.6f}_{lon2:.6f}"
    # Заменяем точки на 'd' и минусы на 'm'
    coords_str = coords_str.replace('.', 'd').replace('-', 'm')
    filename = f"{coords_str}.jpg"
    return os.path.join('output_img', filename)

def download_and_stitch_area(lat1, lon1, lat2, lon2, zoom):
    """Скачивает и склеивает область между двумя точками"""
    print("\n=== СКАЧИВАНИЕ ОБЛАСТИ ===")
    
    # Создаем имя файла
    output_file = create_output_filename(lat1, lon1, lat2, lon2, zoom)
    
    # Определяем реальные границы
    south = min(lat1, lat2)
    north = max(lat1, lat2)
    west = min(lon1, lon2)
    east = max(lon1, lon2)
    
    print(f"Область:")
    print(f"  Юго-запад: {south:.6f}, {west:.6f}")
    print(f"  Северо-восток: {north:.6f}, {east:.6f}")
    print(f"  Файл: {os.path.basename(output_file)}")
    
    # Преобразуем углы в тайлы
    x_nw, y_nw = correct_latlon_to_tile(north, west, zoom)
    x_se, y_se = correct_latlon_to_tile(south, east, zoom)
    
    print(f"Тайлы углов:")
    print(f"  Северо-запад: {x_nw}, {y_nw}")
    print(f"  Юго-восток: {x_se}, {y_se}")
    
    # Определяем диапазон тайлов
    min_x, max_x = min(x_nw, x_se), max(x_nw, x_se)
    min_y, max_y = min(y_nw, y_se), max(y_nw, y_se)
    
    print(f"Диапазон тайлов:")
    print(f"  X: {min_x} - {max_x}")
    print(f"  Y: {min_y} - {max_y}")
    total_tiles = (max_x - min_x + 1) * (max_y - min_y + 1)
    print(f"  Всего тайлов: {total_tiles}")
    
    # Создаем папку для временных файлов
    os.makedirs("temp_tiles", exist_ok=True)
    
    # Скачиваем тайлы
    downloaded = 0
    print("\nСкачиваем тайлы...")
    
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            tile_data = download_tile(x, y, zoom)
            if tile_data:
                filename = f"temp_tiles/tile_{x}_{y}.jpg"
                with open(filename, 'wb') as f:
                    f.write(tile_data)
                downloaded += 1
            print(f"Скачано: {downloaded}/{total_tiles}", end='\r')
            time.sleep(0.05)
    
    print(f"\nУспешно скачано: {downloaded}/{total_tiles} тайлов")
    
    # Склеиваем тайлы
    if downloaded > 0:
        print("Склеиваем тайлы...")
        success = stitch_tiles(min_x, max_x, min_y, max_y, output_file)
        
        # Очищаем временные файлы
        cleanup_temp_files(min_x, max_x, min_y, max_y)
        
        return success, output_file
    else:
        print("Не удалось скачать ни одного тайла!")
        return False, output_file

def stitch_tiles(min_x, max_x, min_y, max_y, output_file):
    """Склеивает тайлы в одно изображение"""
    tile_size = 256
    width = (max_x - min_x + 1) * tile_size
    height = (max_y - min_y + 1) * tile_size
    
    print(f"Создаем изображение {width}x{height} пикселей...")
    
    result = Image.new('RGB', (width, height))
    
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            tile_path = f"temp_tiles/tile_{x}_{y}.jpg"
            if os.path.exists(tile_path):
                try:
                    tile = Image.open(tile_path)
                    pos_x = (x - min_x) * tile_size
                    pos_y = (y - min_y) * tile_size
                    result.paste(tile, (pos_x, pos_y))
                except Exception as e:
                    print(f"Ошибка склейки тайла {x},{y}: {e}")
    
    result.save(output_file, quality=95)
    print(f"✅ Карта сохранена: {output_file}")
    return True

def cleanup_temp_files(min_x, max_x, min_y, max_y):
    """Очищает временные файлы"""
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            tile_path = f"temp_tiles/tile_{x}_{y}.jpg"
            if os.path.exists(tile_path):
                os.remove(tile_path)
    try:
        os.rmdir("temp_tiles")
    except:
        pass

def main():
    """Основная функция для ввода координат"""
    print("=== СКАЧИВАТЕЛЬ КАРТ ЯНДЕКС ===")
    print("Введите координаты области:")
    
    try:
        print("\n📍 Левый нижний угол (юго-запад):")
        coord_input_1 = input("  Координаты (широта, долгота): ")
        lat1, lon1 = map(float, coord_input_1.split(','))
       
        
        print("\n📍 Правый верхний угол (северо-восток):")
        coord_input_2 = input("  Координаты (широта, долгота): ")
        lat2, lon2 = map(float, coord_input_2.split(','))
        
        zoom = 18       
        print(f"\n⏳ Начинаем скачивание...")
        success, filename = download_and_stitch_area(lat1, lon1, lat2, lon2, zoom)
        
        if success:
            print(f"\n🎉 УСПЕХ! Карта сохранена как:")
            print(f"📁 {filename}")
            print(f"📊 Имя файла: {os.path.basename(filename)}")
        else:
            print(f"\n❌ Не удалось создать карту")
            
    except ValueError:
        print("❌ Ошибка: введите корректные числа")
    except KeyboardInterrupt:
        print("\n⏹️ Прервано пользователем")

# Запуск программы
if __name__ == "__main__":
    main()