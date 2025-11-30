import cv2
import numpy as np
import json
import os
from ultralytics import YOLO
from datetime import datetime
from tqdm import tqdm
from scipy import ndimage

def extract_coords_from_filename(filename):
    """Извлекает координаты из названия файла"""
    try:
        base_name = os.path.splitext(filename)[0]
        parts = base_name.split('_')
        
        lat1 = float(parts[1].replace('d', '.'))
        lon1 = float(parts[2].replace('d', '.'))
        lat2 = float(parts[4].replace('d', '.'))
        lon2 = float(parts[5].replace('d', '.'))
        
        north_lat = max(lat1, lat2)
        south_lat = min(lat1, lat2)
        west_lon = min(lon1, lon2)
        east_lon = max(lon1, lon2)
        
        top_left_gps = (north_lat, west_lon)
        bottom_right_gps = (south_lat, east_lon)
        
        print(f"🔍 Извлеченные координаты из имени файла:")
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

def generate_grid_points(mask, grid_size=80):
    """Генерирует точки сетки для больших объектов с проверкой внутри маски"""
    points = []
    try:
        # Получаем высоту и ширину маски
        h, w = mask.shape
        
        # Генерируем точки по всей маске с шагом grid_size
        for y in range(0, h, grid_size):
            for x in range(0, w, grid_size):
                # Проверяем, что точка внутри маски
                if mask[y, x] > 0:
                    points.append((x, y))
        
        # Если точек слишком мало или нет вообще, добавляем дополнительные точки
        if len(points) < 5:
            # Используем морфологические операции для нахождения внутренних точек
            kernel = np.ones((grid_size//2, grid_size//2), np.uint8)
            eroded = cv2.erode(mask, kernel, iterations=1)
            
            # Добавляем точки из эродированной маски
            eroded_points = []
            for y in range(0, h, grid_size//2):
                for x in range(0, w, grid_size//2):
                    if eroded[y, x] > 0:
                        eroded_points.append((x, y))
            
            # Если есть эродированные точки, добавляем их
            if eroded_points:
                points.extend(eroded_points)
            
            # Если все еще нет точек, используем центроид и несколько случайных точек
            if not points:
                centroid = calculate_centroid(mask)
                if centroid:
                    points.append(centroid)
                    # Добавляем несколько точек вокруг центроида
                    for dy in [-grid_size//2, 0, grid_size//2]:
                        for dx in [-grid_size//2, 0, grid_size//2]:
                            if dx == 0 and dy == 0:
                                continue
                            new_x, new_y = centroid[0] + dx, centroid[1] + dy
                            if 0 <= new_x < w and 0 <= new_y < h and mask[new_y, new_x] > 0:
                                points.append((new_x, new_y))
        
        # Убираем дубликаты и ограничиваем максимальное количество точек
        points = list(set(points))
        if len(points) > 50:  # Ограничиваем для очень больших объектов
            step = len(points) // 50
            points = points[::step]
                
    except Exception as e:
        print(f"Ошибка при генерации точек сетки: {e}")
        # Fallback: используем центроид
        centroid = calculate_centroid(mask)
        if centroid:
            points.append(centroid)
    
    return points

def simple_skeletonize(mask):
    """Простая скелетизация без cv2.ximgproc"""
    try:
        # Конвертируем в бинарный формат
        binary_mask = (mask > 0).astype(np.uint8)
        
        # Используем морфологические операции для приблизительной скелетизации
        skeleton = np.zeros_like(binary_mask)
        
        # Итеративное истончение
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        temp = binary_mask.copy()
        
        while True:
            eroded = cv2.erode(temp, element)
            temp_open = cv2.dilate(eroded, element)
            temp_sub = cv2.subtract(temp, temp_open)
            skeleton = cv2.bitwise_or(skeleton, temp_sub)
            temp = eroded.copy()
            
            if cv2.countNonZero(temp) == 0:
                break
                
        return skeleton * 255
        
    except Exception as e:
        print(f"Ошибка при скелетизации: {e}")
        return mask

def generate_road_points(mask, spacing=80):
    """Генерирует точки вдоль дорог с альтернативной скелетизацией"""
    points = []
    try:
        # Используем простую скелетизацию
        skeleton = simple_skeletonize(mask)
        
        # Находим ненулевые точки скелета
        y_coords, x_coords = np.where(skeleton > 0)
        
        if len(x_coords) == 0:
            # Если скелет не найден, используем центроидную линию
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Берем точки из контура с интервалом
                contour = contours[0]
                for i in range(0, len(contour), spacing):
                    point = contour[i][0]
                    points.append((point[0], point[1]))
        
        else:
            # Сортируем точки для последовательного обхода по X
            sorted_indices = np.argsort(x_coords)
            x_sorted = x_coords[sorted_indices]
            y_sorted = y_coords[sorted_indices]
            
            # Выбираем точки с интервалом
            if len(x_sorted) > spacing:
                step = max(1, len(x_sorted) // (len(x_sorted) // spacing))
                for i in range(0, len(x_sorted), step):
                    points.append((x_sorted[i], y_sorted[i]))
            else:
                # Если точек мало, берем все
                for x, y in zip(x_sorted, y_sorted):
                    points.append((x, y))
        
        # Если точек все еще нет, используем центроид и добавляем точки вдоль главной оси
        if not points:
            centroid = calculate_centroid(mask)
            if centroid:
                points.append(centroid)
                # Добавляем точки вдоль предполагаемого направления дороги
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    rect = cv2.minAreaRect(contours[0])
                    angle = rect[2]
                    # Добавляем точки под углом дороги
                    for i in range(1, 4):
                        for sign in [-1, 1]:
                            dx = int(sign * i * spacing * 0.5 * np.cos(np.radians(angle)))
                            dy = int(sign * i * spacing * 0.5 * np.sin(np.radians(angle)))
                            new_point = (centroid[0] + dx, centroid[1] + dy)
                            if (0 <= new_point[0] < mask.shape[1] and 
                                0 <= new_point[1] < mask.shape[0] and 
                                mask[new_point[1], new_point[0]] > 0):
                                points.append(new_point)
                
    except Exception as e:
        print(f"Ошибка при генерации точек дороги: {e}")
        # Fallback: используем центроид
        centroid = calculate_centroid(mask)
        if centroid:
            points.append(centroid)
    
    return points

def pixel_to_gps(pixel_coords, image_width, image_height, top_left_gps, bottom_right_gps):
    """Конвертирует пиксельные координаты в GPS координаты"""
    x_px, y_px = pixel_coords
    
    lat_range = top_left_gps[0] - bottom_right_gps[0]
    lon_range = bottom_right_gps[1] - top_left_gps[1]
    
    x_norm = x_px / (image_width - 1)
    y_norm = y_px / (image_height - 1)
    
    latitude = top_left_gps[0] - (y_norm * lat_range)
    longitude = top_left_gps[1] + (x_norm * lon_range)
    
    return (round(latitude, 8), round(longitude, 8))

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
    
    # Создаем данные для JSON в требуемом формате
    centroids_data = {
        "metadata": {
            "source_image": image_path,
            "image_size": {
                "width": w,
                "height": h
            },
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
    
    # Определяем классы для разных стратегий точек
    GRID_CLASSES = [1, 2, 3]  # field, forest, lake - используют сетку
    ROAD_CLASSES = [4]        # road - точки вдоль дороги
    CENTROID_CLASSES = [0, 5] # building, zrail - одна точка в центре
    
    # Собираем все позиции тайлов
    tile_positions = []
    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            x1 = x
            y1 = y
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            tile_positions.append((x1, y1, x2, y2))
    
    total_tiles = len(tile_positions)
    print(f"🧩 Количество тайлов: {total_tiles}")
    
    processed_tiles = 0
    object_count = 0
    
    # Обрабатываем каждый тайл
    print("🔍 Начинаю обработку тайлов...")
    for x1, y1, x2, y2 in tqdm(tile_positions, desc="Обработка тайлов"):
        tile = original_image[y1:y2, x1:x2]
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
                        
                        # Выбираем стратегию в зависимости от класса
                        if class_id in GRID_CLASSES:
                            # Для полей, лесов и озер - сетка точек
                            points = generate_grid_points(mask_binary, grid_size=160)
                            print(f"   🟩 Класс {class_id}: сгенерировано {len(points)} точек сетки")
                        elif class_id in ROAD_CLASSES:
                            # Для дорог - точки вдоль дороги
                            points = generate_road_points(mask_binary, spacing=80)
                            print(f"   🛣️ Класс {class_id}: сгенерировано {len(points)} точек дороги")
                        else:
                            # Для зданий и железных дорог - одна точка в центре
                            centroid = calculate_centroid(mask_binary)
                            points = [centroid] if centroid else []
                            print(f"   🏠 Класс {class_id}: центроид")
                        
                        # Обрабатываем все точки
                        for point in points:
                            if point:
                                # Преобразуем в глобальные координаты
                                global_point_px = (x1 + point[0], y1 + point[1])
                                
                                # Конвертируем в GPS
                                gps_coords = pixel_to_gps(global_point_px, w, h, top_left_gps, bottom_right_gps)
                                
                                # Сохраняем данные в требуемом формате
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
                plotted_tile = r.plot()
                visualization_image[y1:y2, x1:x2] = plotted_tile
        
        processed_tiles += 1
    
    # Сохраняем визуализацию с полигонами
    cv2.imwrite(output_viz_path, visualization_image)
    print(f"🖼️ Визуализация с полигонами сохранена: {output_viz_path}")
    
    # Сохраняем JSON в требуемом формате
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
    MODEL_PATH = '../runs/segment/yolov8n_gpu_updgrade_1/weights/best.pt'
    IMAGE_PATH = "output_img/map_55d753137_37d282641_to_55d763143_37d308581.jpg"
    base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0].replace('map_', '')

    OUTPUT_JSON = os.path.join("output_json", f"json_{base_name}_upd_yolo.json")
    OUTPUT_POLYGONS = os.path.join("output_yolo_img", f"polygons_{base_name}_upd_yolo.jpg")
    
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