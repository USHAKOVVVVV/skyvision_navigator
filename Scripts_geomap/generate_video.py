import cv2
import numpy as np
import re
import os
import json
from datetime import datetime

def parse_coordinates(filename):
    """Парсинг координат из имени файла"""
    # Извлекаем только имя файла из пути
    basename = os.path.basename(filename)
    
    # Ищем координаты в формате map_ll_lat_ll_lon_to_ur_lat_ur_lon
    # Разрешаем символы d в числах (например, 55d948091)
    pattern = r'map_([\dd\.]+)_([\dd\.]+)_to_([\dd\.]+)_([\dd\.]+)'
    match = re.search(pattern, basename)
    if match:
        # Убираем символы 'd' из координат и преобразуем в float
        ll_lat = float(match.group(1).replace('d', ''))
        ll_lon = float(match.group(2).replace('d', ''))
        ur_lat = float(match.group(3).replace('d', ''))
        ur_lon = float(match.group(4).replace('d', ''))
        
        return {
            'll_lat': ll_lat,
            'll_lon': ll_lon,
            'ur_lat': ur_lat,
            'ur_lon': ur_lon
        }
    else:
        print(f"Не удалось распарсить координаты из: {basename}")
        print(f"Найденные группы: {match.groups() if match else 'нет совпадений'}")
        return None

def latlon_to_pixel(lat, lon, coords, image_width, image_height):
    """Конвертация географических координат в пиксели изображения"""
    lat_ratio = (coords['ur_lat'] - lat) / (coords['ur_lat'] - coords['ll_lat'])
    lon_ratio = (lon - coords['ll_lon']) / (coords['ur_lon'] - coords['ll_lon'])
    
    x = int(lon_ratio * image_width)
    y = int(lat_ratio * image_height)
    
    return x, y

def smoothstep(edge0, edge1, x):
    """Плавная функция интерполяции"""
    x = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return x * x * (3 - 2 * x)


def create_drone_animation_with_gps(input_image, output_video, duration_seconds=30, 
                                   fps=30, output_size=(640, 640), altitude=100):
    """
    Создает анимацию полета дрона с привязкой к GPS координатам
    """
    
    # Парсим координаты из имени файла
    coords = parse_coordinates(input_image)
    if not coords:
        print("Не удалось распарсить координаты из имени файла")
        return
    
    print(f"Координаты карты:")
    print(f"Нижний левый: ({coords['ll_lat']:.6f}, {coords['ll_lon']:.6f})")
    print(f"Верхний правый: ({coords['ur_lat']:.6f}, {coords['ur_lon']:.6f})")
    
    # Загружаем основное изображение
    main_image = cv2.imread(input_image)
    if main_image is None:
        print("Не удалось загрузить изображение")
        return
    
    h, w = main_image.shape[:2]
    
    # Генерация более плавного маршрута с большим количеством точек
    waypoints = [
        # Старт - нижний левый угол (медленный взлет)
        (coords['ll_lat'] + 0.05 * (coords['ur_lat'] - coords['ll_lat']), 
         coords['ll_lon'] + 0.05 * (coords['ur_lon'] - coords['ll_lon'])),
        
        # Плавный подъем
        (coords['ll_lat'] + 0.15 * (coords['ur_lat'] - coords['ll_lat']), 
         coords['ll_lon'] + 0.1 * (coords['ur_lon'] - coords['ll_lon'])),
        
        # Первая точка - слева посередине
        (coords['ll_lat'] + 0.25 * (coords['ur_lat'] - coords['ll_lat']), 
         coords['ll_lon'] + 0.15 * (coords['ur_lon'] - coords['ll_lon'])),
        
        # Плавный поворот
        (coords['ll_lat'] + 0.35 * (coords['ur_lat'] - coords['ll_lat']), 
         coords['ll_lon'] + 0.25 * (coords['ur_lon'] - coords['ll_lon'])),
        
        # Центр
        (coords['ll_lat'] + 0.45 * (coords['ur_lat'] - coords['ll_lat']), 
         coords['ll_lon'] + 0.4 * (coords['ur_lon'] - coords['ll_lon'])),
        
        # Вторая точка
        (coords['ll_lat'] + 0.55 * (coords['ur_lat'] - coords['ll_lat']), 
         coords['ll_lon'] + 0.55 * (coords['ur_lon'] - coords['ll_lon'])),
        
        # Плавный поворот к финишу
        (coords['ll_lat'] + 0.65 * (coords['ur_lat'] - coords['ll_lat']), 
         coords['ll_lon'] + 0.7 * (coords['ur_lon'] - coords['ll_lon'])),
        
        # Подготовка к посадке
        (coords['ll_lat'] + 0.75 * (coords['ur_lat'] - coords['ll_lat']), 
         coords['ll_lon'] + 0.8 * (coords['ur_lon'] - coords['ll_lon'])),
        
        # Финиш - верхний правый угол (медленная посадка)
        (coords['ur_lat'] - 0.05 * (coords['ur_lat'] - coords['ll_lat']), 
         coords['ur_lon'] - 0.05 * (coords['ur_lon'] - coords['ll_lon']))
    ]
    
    # Создаем визуализацию маршрута
 
    
    # Создаем видео writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, output_size)
    
    total_frames = int(fps * duration_seconds)
    
    # Создаем структуру для хранения данных GPS
    gps_data = {
        "video_info": {
            "filename": output_video,
            "duration_seconds": duration_seconds,
            "fps": fps,
            "total_frames": total_frames,
            "creation_time": datetime.now().isoformat(),
            "map_coordinates": coords
        },
        "flight_data": []
    }
    
    print("Генерация плавной анимации с GPS привязкой...")
    print(f"Маршрут через {len(waypoints)} точек")
    print(f"Длительность: {duration_seconds} сек, Кадров: {total_frames}")
    
    # Предварительно вычисляем все позиции для максимальной плавности
    positions = []
    for frame_num in range(total_frames):
        progress = frame_num / total_frames
        
        # Очень плавная функция прогресса с замедлением в начале и конце
        eased_progress = smoothstep(0, 1, progress)
        
        # Определяем текущий сегмент маршрута
        segment_progress = eased_progress * (len(waypoints) - 1)
        segment_index = min(int(segment_progress), len(waypoints) - 2)
        local_progress = segment_progress - segment_index
        
        # Супер-плавная интерполяция между точками
        super_smooth_progress = 0.5 - 0.5 * np.cos(local_progress * np.pi)
        
        # Текущая позиция в GPS координатах
        start_lat, start_lon = waypoints[segment_index]
        end_lat, end_lon = waypoints[segment_index + 1]
        
        current_lat = start_lat + (end_lat - start_lat) * super_smooth_progress
        current_lon = start_lon + (end_lon - start_lon) * super_smooth_progress
        
        # Конвертируем в пиксели
        x_center, y_center = latlon_to_pixel(current_lat, current_lon, coords, w, h)
        
        # Плавное изменение высоты (масштаба) с замедлением в начале и конце
        start_scale = 0.3   # Очень высоко в начале
        mid_scale = 0.6     # Средняя высота
        end_scale = 0.4     # Немного опускаемся в конце
        
        if eased_progress < 0.3:  # Медленный взлет
            height_progress = eased_progress / 0.3
            current_scale = start_scale + (mid_scale - start_scale) * smoothstep(0, 1, height_progress)
        elif eased_progress < 0.7:  # Плавный полет
            current_scale = mid_scale
        else:  # Медленная посадка
            height_progress = (eased_progress - 0.7) / 0.3
            current_scale = mid_scale + (end_scale - mid_scale) * smoothstep(0, 1, height_progress)
        
        positions.append((x_center, y_center, current_lat, current_lon, current_scale))
        
        # Сохраняем данные для JSON (каждый кадр)
        frame_data = {
            "frame_number": frame_num,
            "timestamp_seconds": round(frame_num / fps, 3),
            "timestamp_formatted": f"{int(frame_num // fps):02d}:{int(frame_num % fps):02d}",
            "gps_coordinates": {
                "latitude": round(current_lat, 6),
                "longitude": round(current_lon, 6)
            },
            "pixel_coordinates": {
                "x": x_center,
                "y": y_center
            },
            "altitude_meters": int(altitude * current_scale/0.3),
            "scale_factor": round(current_scale, 3),
            "progress_percent": round(progress * 100, 1)
        }
        gps_data["flight_data"].append(frame_data)
    
    # Генерируем кадры видео
    print("Генерация кадров...")
    for frame_num in range(total_frames):
        x_center, y_center, current_lat, current_lon, current_scale = positions[frame_num]
        progress = frame_num / total_frames
        
        # Вычисляем размер для текущего кадра
        crop_w = int(output_size[0] / current_scale)
        crop_h = int(output_size[1] / current_scale)
        
        # Вычисляем область обрезки
        x1 = max(0, x_center - crop_w // 2)
        y1 = max(0, y_center - crop_h // 2)
        x2 = min(w, x_center + crop_w // 2)
        y2 = min(h, y_center + crop_h // 2)
        
        # Вырезаем и масштабируем
        if x2 > x1 and y2 > y1:
            cropped = main_image[y1:y2, x1:x2]
            
            # Плавное масштабирование с интерполяцией
            resized = cv2.resize(cropped, output_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Добавляем информацию о полете
            cv2.putText(resized, f"Altitude: {int(altitude * current_scale/0.3)}m", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(resized, f"GPS: {current_lat:.6f}, {current_lon:.6f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(resized, f"Speed: 4 m/s", (10, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(resized, f"Progress: {progress*100:.1f}%", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
       
            
             
    
    # Сохраняем GPS данные в JSON файл
       # Преобразуем координаты перед сохранением
    gps_data_converted = convert_coordinates_data(gps_data)
    
    # Сохраняем GPS данные в JSON файл
    json_filename = output_video.replace('.mp4', '_gps_data.json')
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(gps_data_converted, f, indent=2, ensure_ascii=False)
    # Также сохраняем упрощенную версию (только ключевые кадры)
    # Также сохраняем упрощенную версию (только ключевые кадры)
 
    
    print(f"✅ Видео сохранено: {output_video}")
    print(f"📊 Длительность: {duration_seconds} сек")
    print(f"🎞️ Кадров: {total_frames}")
    print(f"🔄 FPS: {fps}")
    print(f"📏 Размер: {output_size[0]}x{output_size[1]}")
    print(f"🛩️ Высота: {altitude} м")
    print(f"🗺️ GPS данные сохранены: {json_filename}")
 

# Функция для чтения GPS данных из JSON
def read_gps_data(json_file):
    """Чтение GPS данных из JSON файла"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

# Функция для поиска GPS координат по времени
def find_gps_by_time(json_file, timestamp_seconds):
    """Найти GPS координаты по времени в секундах"""
    data = read_gps_data(json_file)
    
    for frame_data in data["flight_data"]:
        if abs(frame_data["timestamp_seconds"] - timestamp_seconds) < 0.1:
            return frame_data["gps_coordinates"]
    
    return None
def convert_coordinate(coord):
    """Преобразует координату из формата 55949078.650004 в 55.949079"""
    if isinstance(coord, (int, float)):
        # Если координата в неправильном формате (типа 55949078.650004)
        if coord > 1000:
            coord_str = str(int(coord))
            # Берем первые 2 цифры как градусы, остальные как десятичные
            degrees = int(coord_str[:2])
            decimal_part = float("0." + coord_str[2:])
            return degrees + decimal_part
    return float(coord)

def convert_coordinates_data(data):
    """Конвертирует все координаты в данных в правильный GPS формат"""
    converted = data.copy()
    
    # Конвертируем координаты карты
    if 'video_info' in converted and 'map_coordinates' in converted['video_info']:
        map_coords = converted['video_info']['map_coordinates']
        map_coords['ll_lat'] = convert_coordinate(map_coords['ll_lat'])
        map_coords['ll_lon'] = convert_coordinate(map_coords['ll_lon'])
        map_coords['ur_lat'] = convert_coordinate(map_coords['ur_lat'])
        map_coords['ur_lon'] = convert_coordinate(map_coords['ur_lon'])
    
    # Конвертируем координаты полета
    if 'flight_data' in converted:
        for frame in converted['flight_data']:
            if 'gps_coordinates' in frame:
                frame['gps_coordinates']['latitude'] = convert_coordinate(frame['gps_coordinates']['latitude'])
                frame['gps_coordinates']['longitude'] = convert_coordinate(frame['gps_coordinates']['longitude'])
    
    return converted
# Использование
if __name__ == "__main__":
    # Укажите путь к вашему файлу карты
    input_map = "output_img/map_55d753137_37d282641_to_55d763143_37d308581.jpg"
    
    if os.path.exists(input_map):
        create_drone_animation_with_gps(
            input_image=input_map,
            output_video="drone_flight_smooth.mp4",
            duration_seconds=30,
            fps=60,
            output_size=(640, 640),
            altitude=100
        )
        
        # Пример использования функций для чтения данных
        print("\n📖 Пример чтения GPS данных:")
        gps_data = read_gps_data("drone_flight_smooth_gps_data.json")
        print(f"Всего записей: {len(gps_data['flight_data'])}")
        print(f"Первая запись: {gps_data['flight_data'][0]}")
        print(f"Последняя запись: {gps_data['flight_data'][-1]}")
        
    else:
        print(f"Файл {input_map} не найден!")
        print("Убедитесь, что файл карты существует в текущей директории")