import cv2
import numpy as np
import re
import os

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

def create_route_visualization(main_image, coords, waypoints, output_path="route_visualization.jpg"):
    """Создает визуализацию маршрута на карте"""
    route_viz = main_image.copy()
    h, w = route_viz.shape[:2]
    
    # Рисуем маршрут
    route_points = []
    for lat, lon in waypoints:
        x, y = latlon_to_pixel(lat, lon, coords, w, h)
        route_points.append((x, y))
    
    # Рисуем линию маршрута
    for i in range(len(route_points) - 1):
        cv2.line(route_viz, route_points[i], route_points[i+1], (0, 255, 0), 3)
    
    # Рисуем точки маршрута
    for i, (x, y) in enumerate(route_points):
        color = (0, 0, 255) if i == 0 else (255, 0, 0) if i == len(route_points)-1 else (0, 255, 255)
        cv2.circle(route_viz, (x, y), 8, color, -1)
        cv2.putText(route_viz, f"{i+1}", (x-5, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Добавляем легенду
    cv2.putText(route_viz, "Drone Flight Route", (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(route_viz, "Start (Red)", (20, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(route_viz, "Waypoints (Yellow)", (20, 85), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(route_viz, "End (Blue)", (20, 110), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    cv2.imwrite(output_path, route_viz)
    print(f"Визуализация маршрута сохранена: {output_path}")
    return route_viz

def create_drone_animation_with_gps(input_image, output_video, duration_seconds=30, 
                                   fps=60, output_size=(640, 640), altitude=100):
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
    route_viz = create_route_visualization(main_image, coords, waypoints, "flight_route.jpg")
    
    # Создаем видео writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, output_size)
    
    total_frames = int(fps * duration_seconds)
    
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
            
            # Добавляем индикатор текущей позиции на мини-карте
            mini_map_size = 120
            mini_map = cv2.resize(route_viz, (mini_map_size, mini_map_size), interpolation=cv2.INTER_LANCZOS4)
            
            # Отмечаем текущую позицию на мини-карте
            mini_x = int((x_center / w) * mini_map_size)
            mini_y = int((y_center / h) * mini_map_size)
            cv2.circle(mini_map, (mini_x, mini_y), 4, (0, 0, 255), -1)
            
            # Рисуем траекторию пройденного пути на мини-карте
            for i in range(min(frame_num // 10, len(positions) // 10)):
                idx = i * 10
                if idx < len(positions):
                    px, py, _, _, _ = positions[idx]
                    trail_x = int((px / w) * mini_map_size)
                    trail_y = int((py / h) * mini_map_size)
                    cv2.circle(mini_map, (trail_x, trail_y), 1, (255, 255, 0), -1)
            
            # Вставляем мини-карту в основной кадр
            resized[10:10+mini_map_size, output_size[0]-10-mini_map_size:output_size[0]-10] = mini_map
            
            # Рамка вокруг мини-карты
            cv2.rectangle(resized, 
                         (output_size[0]-10-mini_map_size, 10),
                         (output_size[0]-10, 10+mini_map_size),
                         (255, 255, 255), 2)
            
            out.write(resized)
        
        # Прогресс
        if frame_num % (fps * 5) == 0:  # Сообщение каждые 5 секунд
            print(f"Обработано: {progress*100:.1f}%")
    
    out.release()
    print(f"✅ Видео сохранено: {output_video}")
    print(f"📊 Длительность: {duration_seconds} сек")
    print(f"🎞️ Кадров: {total_frames}")
    print(f"🔄 FPS: {fps}")
    print(f"📏 Размер: {output_size[0]}x{output_size[1]}")
    print(f"🛩️ Высота: {altitude} м")

# Использование
if __name__ == "__main__":
    # Укажите путь к вашему файлу карты
    input_map = "output_img/map_55d948091_37d941703_to_55d967844_37d996474.jpg"
    
    if os.path.exists(input_map):
        create_drone_animation_with_gps(
            input_image=input_map,
            output_video="drone_flight_smooth.mp4",
            duration_seconds=30,  # Увеличил длительность в 2 раза
            fps=60,
            output_size=(640, 640),
            altitude=100
        )
    else:
        print(f"Файл {input_map} не найден!")
        print("Убедитесь, что файл карты существует в текущей директории")