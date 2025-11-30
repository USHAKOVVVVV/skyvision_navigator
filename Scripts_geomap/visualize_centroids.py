import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

def visualize_polygons_from_json(json_path, output_image_path):
    """Визуализация всех полигонов из JSON"""
    
    # Загружаем данные
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metadata = data['metadata']
    objects = data['objects']
    
    # Извлекаем границы
    gps_bounds = metadata['gps_bounds']
    top_left = gps_bounds['top_left']
    bottom_right = gps_bounds['bottom_right']
    
    print(f"📍 GPS границы: С-З {top_left}, Ю-В {bottom_right}")
    print(f"📊 Всего объектов: {len(objects)}")
    
    # Создаем карту
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Цвета для классов
    class_colors = {
        0: 'red',      # building
        1: 'yellow',    # field
        2: 'darkgreen', # forest
        3: 'blue',     # lake
        4: 'gray',     # road
        5: 'orange'    # zrail
    }
    
    class_names = {
        0: 'Building',
        1: 'Field',
        2: 'Forest',
        3: 'Lake',
        4: 'Road', 
        5: 'Railway'
    }
    
    # Собираем данные по классам
    class_data = {}
    for obj in objects:
        class_id = obj['class_id']
        if class_id not in class_data:
            class_data[class_id] = []
        class_data[class_id].append(obj)
    
    # Рисуем полигоны (точки с областями)
    for class_id, objects_list in class_data.items():
        if objects_list:
            lons = [obj['gps_coordinates']['longitude'] for obj in objects_list]
            lats = [obj['gps_coordinates']['latitude'] for obj in objects_list]
            confidences = [obj['confidence'] for obj in objects_list]
            
            color = class_colors.get(class_id, 'purple')
            
            # Рисуем точки с размерами по confidence
            sizes = [10 + conf * 50 for conf in confidences]  # размер зависит от confidence
            scatter = ax.scatter(lons, lats, 
                               c=color, 
                               s=sizes,
                               alpha=0.6,
                               edgecolors='black',
                               linewidth=0.5,
                               label=class_names.get(class_id, f'Class {class_id}'))
    
    # Настройки карты
    ax.set_xlabel('Долгота (Longitude)', fontsize=12)
    ax.set_ylabel('Широта (Latitude)', fontsize=12)
    ax.set_title('Визуализация всех полигонов объектов', fontsize=16, pad=20)
    
    # Сетка
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Легенда
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Устанавливаем границы
    ax.set_xlim(top_left[1], bottom_right[1])
    ax.set_ylim(bottom_right[0], top_left[0])
    
    # Сохраняем
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Визуализация полигонов сохранена: {output_image_path}")
    
    # Статистика
    print("📊 Статистика по классам:")
    for class_id, objects_list in class_data.items():
        class_name = class_names.get(class_id, f'Class {class_id}')
        print(f"   - {class_name}: {len(objects_list)} объектов")

if __name__ == "__main__":
    
     
    JSON_PATH = "output_json/json_55d753137_37d282641_to_55d763143_37d308581_upd_yolo.json"
    base_name = os.path.splitext(os.path.basename(JSON_PATH))[0].replace('json_', '')
    OUTPUT_CENTROIDS = os.path.join("output_yolo_img", f"centroids_{base_name}.jpg")
    
    visualize_polygons_from_json(JSON_PATH, OUTPUT_CENTROIDS)