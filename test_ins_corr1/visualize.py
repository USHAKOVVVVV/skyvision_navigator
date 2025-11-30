import json
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def load_json_data(file_path):
    """Загрузка данных из JSON файла"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def gps_to_pixel(lat, lon, map_coords, img_width, img_height):
    """Конвертация GPS координат в пиксели изображения"""
    lat_ratio = (lat - map_coords['ll_lat']) / (map_coords['ur_lat'] - map_coords['ll_lat'])
    lon_ratio = (lon - map_coords['ll_lon']) / (map_coords['ur_lon'] - map_coords['ll_lon'])
    
    x = int(lon_ratio * img_width)
    y = int((1 - lat_ratio) * img_height)  # Инвертируем Y-координату
    
    return x, y

def create_verification_visualization():
    """Создание проверочной визуализации траектории и объектов"""
    
    # Загружаем данные
    flight_data = load_json_data("flight_data_visible_error.json")
    detection_data = load_json_data("json_55d753137_37d282641_to_55d763143_37d308581_upd_yolo.json")
    
    # Загружаем карту
    map_image_path = "map_55d753137_37d282641_to_55d763143_37d308581.jpg"
    map_img = Image.open(map_image_path)
    img_width, img_height = map_img.size
    
    # Создаем график
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))
    ax.imshow(map_img)
    ax.set_title("Проверка координат: траектория полета и обнаруженные объекты", 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Координаты карты из flight_data
    map_coords = flight_data['video_info']['map_coordinates']
    
    # 1. Рисуем траекторию полета
    flight_points = []
    for frame in flight_data['flight_data'][::10]:  # Берем каждый 10-й кадр
        lat = frame['gps_coordinates']['latitude']
        lon = frame['gps_coordinates']['longitude']
        x, y = gps_to_pixel(lat, lon, map_coords, img_width, img_height)
        flight_points.append((x, y))
    
    if flight_points:
        flight_x, flight_y = zip(*flight_points)
        ax.plot(flight_x, flight_y, 'b-', linewidth=3, label='Траектория полета дрона', alpha=0.7)
        
        # Отмечаем начало и конец траектории
        ax.scatter(flight_x[0], flight_y[0], c='green', s=200, marker='o', 
                  label='Старт', edgecolors='white', linewidth=2)
        ax.scatter(flight_x[-1], flight_y[-1], c='red', s=200, marker='o', 
                  label='Финиш', edgecolors='white', linewidth=2)
    
    # 2. Рисуем обнаруженные объекты
    class_colors = {
        1: 'orange',  # Класс 1
        2: 'purple',  # Класс 2  
        3: 'cyan',    # Класс 3
    }
    
    class_labels = {
        1: 'Объект класса 1',
        2: 'Объект класса 2',
        3: 'Объект класса 3'
    }
    
    drawn_classes = set()
    
    for obj in detection_data['objects']:
        class_id = obj['class_id']
        lat = obj['gps_coordinates']['latitude']
        lon = obj['gps_coordinates']['longitude']
        confidence = obj['confidence']
        
        x, y = gps_to_pixel(lat, lon, map_coords, img_width, img_height)
        
        color = class_colors.get(class_id, 'gray')
        label = class_labels.get(class_id, f'Класс {class_id}')
        
        # Рисуем объект
        ax.scatter(x, y, c=color, s=150, marker='s', 
                  label=label if class_id not in drawn_classes else "", 
                  alpha=0.8, edgecolors='black', linewidth=1)
        
        # Подписываем confidence
        ax.annotate(f'{confidence:.2f}', (x, y), xytext=(5, 5), 
                   textcoords='offset points', fontsize=8, color='black',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        
        drawn_classes.add(class_id)
    
    # 3. Рисуем границы карты для проверки
    corners = [
        (map_coords['ll_lat'], map_coords['ll_lon']),  # Нижний левый
        (map_coords['ll_lat'], map_coords['ur_lon']),  # Нижний правый  
        (map_coords['ur_lat'], map_coords['ur_lon']),  # Верхний правый
        (map_coords['ur_lat'], map_coords['ll_lon']),  # Верхний левый
        (map_coords['ll_lat'], map_coords['ll_lon'])   # Замыкаем
    ]
    
    corner_points = []
    for lat, lon in corners:
        x, y = gps_to_pixel(lat, lon, map_coords, img_width, img_height)
        corner_points.append((x, y))
    
    corner_x, corner_y = zip(*corner_points)
    ax.plot(corner_x, corner_y, 'r--', linewidth=2, label='Границы карты', alpha=0.5)
    
    # 4. Добавляем информационную панель
    info_text = f"""
    Информация о данных:
    • Траектория: {len(flight_data['flight_data'])} кадров
    • Обнаружено объектов: {len(detection_data['objects'])}
    • Координаты карты:
      LL: ({map_coords['ll_lat']:.6f}, {map_coords['ll_lon']:.6f})
      UR: ({map_coords['ur_lat']:.6f}, {map_coords['ur_lon']:.6f})
    • Размер карты: {img_width} x {img_height} px
    """
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    
    # Настройки графика
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Пиксели X', fontsize=12)
    ax.set_ylabel('Пиксели Y', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('coordinate_verification.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Проверочная визуализация создана: coordinate_verification.png")
    print("📊 Статистика:")
    print(f"   - Кадров траектории: {len(flight_data['flight_data'])}")
    print(f"   - Обнаруженных объектов: {len(detection_data['objects'])}")
    print(f"   - Размер карты: {img_width}x{img_height} пикселей")
    
    # Проверка координат
    print("\n🔍 Проверка координат:")
    first_frame = flight_data['flight_data'][0]
    print(f"   - Первый кадр: lat={first_frame['gps_coordinates']['latitude']:.6f}, lon={first_frame['gps_coordinates']['longitude']:.6f}")
    
    last_frame = flight_data['flight_data'][-1]
    print(f"   - Последний кадр: lat={last_frame['gps_coordinates']['latitude']:.6f}, lon={last_frame['gps_coordinates']['longitude']:.6f}")
    
    # Проверка вхождения координат в границы карты
    def check_coordinates_in_bounds(lat, lon):
        return (map_coords['ll_lat'] <= lat <= map_coords['ur_lat'] and 
                map_coords['ll_lon'] <= lon <= map_coords['ur_lon'])
    
    first_in_bounds = check_coordinates_in_bounds(
        first_frame['gps_coordinates']['latitude'],
        first_frame['gps_coordinates']['longitude']
    )
    last_in_bounds = check_coordinates_in_bounds(
        last_frame['gps_coordinates']['latitude'], 
        last_frame['gps_coordinates']['longitude']
    )
    
    print(f"   - Первый кадр в границах карты: {'✅' if first_in_bounds else '❌'}")
    print(f"   - Последний кадр в границах карты: {'✅' if last_in_bounds else '❌'}")

# Запуск проверки
if __name__ == "__main__":
    create_verification_visualization()