import json
import matplotlib.pyplot as plt

def visualize_gps_points(json_path, check_point=None):
    """Визуализирует GPS точки из JSON и проверочную точку"""
    
    # Загружаем данные из JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Извлекаем все точки объектов
    points = []
    for obj in data['objects']:
        lat = obj['gps_coordinates']['latitude']
        lon = obj['gps_coordinates']['longitude']
        points.append((lat, lon))
    
    # Извлекаем границы карты
    bounds = data['metadata']['gps_bounds']
    top_left = bounds['top_left']
    bottom_right = bounds['bottom_right']
    
    # Создаем график
    plt.figure(figsize=(12, 10))
    
    # Рисуем точки из JSON
    if points:
        lats, lons = zip(*points)
        plt.scatter(lons, lats, c='blue', s=50, alpha=0.7, label='Обнаруженные объекты')
    
    # Рисуем проверочную точку
    if check_point:
        check_lat, check_lon = check_point
        plt.scatter([check_lon], [check_lat], c='red', s=200, marker='*', 
                   label=f'Проверочная точка: {check_lat}, {check_lon}')
    
    # Рисуем границы карты
    plt.plot([top_left[1], bottom_right[1], bottom_right[1], top_left[1], top_left[1]], 
             [top_left[0], top_left[0], bottom_right[0], bottom_right[0], top_left[0]], 
             'g--', alpha=0.5, label='Границы карты')
    
    # Настройки графика
    plt.xlabel('Долгота')
    plt.ylabel('Широта')
    plt.title('Визуализация GPS координат из JSON')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Сохраняем график
    output_path = json_path.replace('.json', '_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Визуализация сохранена: {output_path}")
    
    # Выводим статистику
    print(f"\n📈 Статистика:")
    print(f"   Всего точек: {len(points)}")
    print(f"   Границы карты: {top_left} -> {bottom_right}")
    if check_point:
        print(f"   Проверочная точка: {check_point}")

# Использование:
if __name__ == "__main__":
    JSON_PATH = "Scripts_geomap/output_json/json_55d948091_37d941703_to_55d967844_37d996474.json"  # путь к вашему JSON
    CHECK_POINT = (55.960646, 37.959188)  # ваша проверочная точка
    
    visualize_gps_points(JSON_PATH, CHECK_POINT)