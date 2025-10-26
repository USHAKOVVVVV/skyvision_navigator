import cv2
import numpy as np
import json
import os
from ultralytics import YOLO
from datetime import datetime
from tqdm import tqdm

def extract_coords_from_filename(filename):
    """Извлекает координаты из названия файла"""
    # Пример: map_55d954635_37d942221_to_55d962248_37d953572_z18.jpg
    try:
        base_name = os.path.splitext(filename)[0]  # убираем расширение
        parts = base_name.split('_')
        
        # Извлекаем координаты и заменяем 'd' обратно на точки
        lat1 = float(parts[1].replace('d', '.'))
        lon1 = float(parts[2].replace('d', '.'))
        lat2 = float(parts[4].replace('d', '.'))
        lon2 = float(parts[5].replace('d', '.'))
        
        # ОПРЕДЕЛЯЕМ СЕВЕРНУЮ И ЮЖНУЮ ГРАНИЦЫ
        # Широта: северная больше, южная меньше
        north_lat = max(lat1, lat2)
        south_lat = min(lat1, lat2)
        
        # Долгота: западная меньше, восточная больше  
        west_lon = min(lon1, lon2)
        east_lon = max(lon1, lon2)
        
        top_left_gps = (north_lat, west_lon)     # Северо-запад (северная широта, западная долгота)
        bottom_right_gps = (south_lat, east_lon) # Юго-восток (южная широта, восточная долгота)
        
        print(f"🔍 Извлеченные координаты из имени файла:")
        print(f"   Точка 1: {lat1}, {lon1}")
        print(f"   Точка 2: {lat2}, {lon2}")
        print(f"   Северо-запад (top_left): {top_left_gps}")
        print(f"   Юго-восток (bottom_right): {bottom_right_gps}")
        
        return top_left_gps, bottom_right_gps
        
    except Exception as e:
        print(f"❌ Ошибка извлечения координат из имени файла: {e}")
        return None, None


def calculate_centroid(mask):
    """Вычисляет центроид бинарной маски"""
    try:
        moments = cv2.moments(mask)
        if moments["m00"] != 0:
            centroid_x = int(moments["m10"] / moments["m00"])
            centroid_y = int(moments["m01"] / moments["m00"])
            return (centroid_x, centroid_y)
        else:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w, h = cv2.boundingRect(contours[0])
                return (x + w//2, y + h//2)
            else:
                return None
    except Exception as e:
        print(f"Ошибка при вычислении центроида: {e}")
        return None

def pixel_to_gps(pixel_coords, image_width, image_height, top_left_gps, bottom_right_gps):
    """Конвертирует пиксельные координаты в GPS координаты"""
    x_px, y_px = pixel_coords
    
    # ПРАВИЛЬНЫЙ РАСЧЕТ ДИАПАЗОНОВ
    lat_range = top_left_gps[0] - bottom_right_gps[0]  # разница по широте
    lon_range = bottom_right_gps[1] - top_left_gps[1]  # разница по долготе
    
    # Нормализация координат (0-1)
    x_norm = x_px / (image_width - 1)  # делим на (width-1) для правильной интерполяции
    y_norm = y_px / (image_height - 1)  # делим на (height-1) для правильной интерполяции
    
    # ПРАВИЛЬНАЯ ИНТЕРПОЛЯЦИЯ:
    # Широта: от верхней к нижней (y увеличивается вниз)
    latitude = top_left_gps[0] - (y_norm * lat_range)
    # Долгота: от левой к правой (x увеличивается вправо)  
    longitude = top_left_gps[1] + (x_norm * lon_range)
    
    return (round(latitude, 8), round(longitude, 8))  # увеличим точность до 8 знаков

def process_map_with_yolo_tiled(model_path, image_path, output_json_path, output_viz_path, tile_size=640, overlap=64, conf=0.3):
    """Обработка карты YOLO с тайлингом и визуализацией"""
    
    # Извлекаем координаты из названия файла
    top_left_gps, bottom_right_gps = extract_coords_from_filename(os.path.basename(image_path))
    
    if not top_left_gps:
        return
    
    print(f"📍 Координаты из файла: С-З {top_left_gps}, Ю-В {bottom_right_gps}")
    
    # Загружаем модель и изображение
    print("🔄 Загружаю модель YOLO...")
    model = YOLO(model_path)
    print("✅ Модель загружена")
    
    print(f"🖼️ Загружаю изображение: {image_path}")
    original_image = cv2.imread(image_path)
    if original_image is None:
        print("❌ Не могу загрузить изображение")
        return
        
    h, w = original_image.shape[:2]
    print(f"📐 Размер изображения: {w}x{h}")
    
    # Создаем копию для визуализации
    visualization_image = original_image.copy()
    
    # Создаем данные для JSON
    centroids_data = {
        "metadata": {
            "source_image": image_path,
            "image_size": {"width": w, "height": h},
            "gps_bounds": {
                "top_left": list(top_left_gps),
                "bottom_right": list(bottom_right_gps)
            },
            "processing_date": datetime.now().isoformat(),
            "model_used": model_path,
            "processing_params": {
                "tile_size": tile_size,
                "overlap": overlap,
                "confidence_threshold": conf
            }
        },
        "objects": []
    }
    
    # ПРАВИЛЬНЫЙ РАСЧЕТ КОЛИЧЕСТВА ТАЙЛОВ
    x_tiles = 0
    y_tiles = 0
    tile_positions = []
    
    # Собираем все позиции тайлов
    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            x1 = x
            y1 = y
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            
            # УБИРАЕМ УСЛОВИЕ ПРОПУСКА - обрабатываем ВСЕ тайлы
            # даже если они маленькие, чтобы не пропустить края
            tile_positions.append((x1, y1, x2, y2))
    
    total_tiles = len(tile_positions)
    
    print(f"🧩 Количество тайлов: {total_tiles}")
    print(f"🔲 Размер тайла: {tile_size}x{tile_size}")
    print(f"🔄 Перекрытие: {overlap} пикселей")
    
    processed_tiles = 0
    object_count = 0
    
    # Обрабатываем каждый тайл
    print("🔍 Начинаю обработку тайлов...")
    for x1, y1, x2, y2 in tqdm(tile_positions, desc="Обработка тайлов"):
        # Вырезаем тайл (даже если он маленький)
        tile = original_image[y1:y2, x1:x2]
        
        # Обрабатываем тайл моделью
        results = model(tile, conf=conf, verbose=False)
        
        # Обрабатываем результаты для этого тайла
        for r in results:
            if r.masks is not None and len(r.masks) > 0:
                for i, mask in enumerate(r.masks.data):
                    if i < len(r.boxes.cls):
                        class_id = int(r.boxes.cls[i])
                        confidence = float(r.boxes.conf[i])
                        
                        # Конвертируем маску
                        mask_np = mask.cpu().numpy()
                        mask_resized = cv2.resize(mask_np, (x2-x1, y2-y1))
                        mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                        
                        # Вычисляем центроид
                        centroid = calculate_centroid(mask_binary)
                        
                        if centroid:
                            # Преобразуем в глобальные координаты
                            global_centroid_px = (x1 + centroid[0], y1 + centroid[1])
                            
                            # Конвертируем в GPS
                            gps_coords = pixel_to_gps(global_centroid_px, w, h, top_left_gps, bottom_right_gps)
                            
                            # Сохраняем данные
                            object_data = {
                                "class_id": class_id,
                                "confidence": confidence,
                                "gps_coordinates": {
                                    "latitude": gps_coords[0],
                                    "longitude": gps_coords[1]
                                }
                            }
                            centroids_data["objects"].append(object_data)
                            object_count += 1
            
            # ВИЗУАЛИЗАЦИЯ: рисуем полигоны на изображении
            if hasattr(r, 'plot') and r.boxes is not None:
                plotted_tile = r.plot()  # получаем тайл с нарисованными полигонами
                # Вставляем обработанный тайл в визуализацию
                visualization_image[y1:y2, x1:x2] = plotted_tile
        
        processed_tiles += 1
    
    # Сохраняем визуализацию с полигонами
    cv2.imwrite(output_viz_path, visualization_image)
    print(f"🖼️ Визуализация с полигонами сохранена: {output_viz_path}")
    
    # Сохраняем JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(centroids_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Итоги обработки:")
    print(f"   Обработано тайлов: {processed_tiles}/{total_tiles}")
    print(f"   Обнаружено объектов: {object_count}")
    print(f"   JSON сохранен: {output_json_path}")
    print(f"   Визуализация сохранена: {output_viz_path}")
    
    # Статистика по классам
    if object_count > 0:
        class_counts = {}
        for obj in centroids_data["objects"]:
            class_id = obj["class_id"]
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        print("📊 Статистика по классам:")
        for class_id, count in class_counts.items():
            class_name = model.names[class_id]
            print(f"   - {class_name}: {count} объектов")
if __name__ == "__main__":
    # Настройки
    MODEL_PATH = '../runs/segment/yolov8n_gpu_simple_1/weights/best.pt'  # путь к твоей модели
    IMAGE_PATH = "output_img/map_55d948091_37d941703_to_55d967844_37d996474.jpg"  # карта из папки maps
    base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0].replace('map_', '')

# Создаем названия файлов с координатами
    OUTPUT_JSON = os.path.join("output_json", f"json_{base_name}.json")
    OUTPUT_POLYGONS = os.path.join("output_yolo_img", f"polygons_{base_name}.jpg")
    

     # Параметры тайлинга
    TILE_SIZE = 640
    OVERLAP = 10
    CONFIDENCE = 0.25
    
    print(f"📁 Выходные файлы:")
    print(f"   JSON: {OUTPUT_JSON}")
    print(f"   Polygons: {OUTPUT_POLYGONS}")
    
    print("🚀 Запуск обработки YOLO...")
    process_map_with_yolo_tiled(
        MODEL_PATH, 
        IMAGE_PATH, 
        OUTPUT_JSON,
        OUTPUT_POLYGONS,
        tile_size=TILE_SIZE,
        overlap=OVERLAP,
        conf=CONFIDENCE
    )