import cv2
import numpy as np
import json
import os
import random
from math import radians, cos, sin, sqrt, atan2
import matplotlib.pyplot as plt


class RANSACMatcher:
    def __init__(self, num_iterations=1000, inlier_threshold=0.05, min_inliers=2):
        self.num_iterations = num_iterations
        self.inlier_threshold = inlier_threshold
        self.min_inliers = min_inliers
    
    def find_similarity_transform(self, src_points, dst_points):
        """Находит преобразование подобия между двумя наборами точек"""
        if len(src_points) < 2 or len(dst_points) < 2:
            return None
            
        if len(src_points) != len(dst_points):
            raise ValueError("Количество исходных и целевых точек должно совпадать")
            
        try:
            # Центрируем точки
            src_center = np.mean(src_points, axis=0)
            dst_center = np.mean(dst_points, axis=0)
            
            src_centered = src_points - src_center
            dst_centered = dst_points - dst_center
            
            # Вычисляем масштаб и поворот
            src_norm = np.linalg.norm(src_centered, axis=1)
            dst_norm = np.linalg.norm(dst_centered, axis=1)
            
            if np.mean(src_norm) == 0 or np.mean(dst_norm) == 0:
                return None
                
            scale = np.mean(dst_norm) / np.mean(src_norm)
            
            # Вычисляем угол поворота через SVD
            H = src_centered.T @ dst_centered
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            # Если определитель отрицательный, корректируем отражение
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            # Матрица преобразования подобия
            transform = np.eye(3)
            transform[0:2, 0:2] = R * scale
            transform[0:2, 2] = dst_center - scale * R @ src_center
            
            return transform
            
        except np.linalg.LinAlgError:
            return None
    
    def apply_transform(self, points, transform):
        """Применяет аффинное преобразование к точкам"""
        if len(points) == 0:
            return np.array([])
            
        homogeneous_points = np.column_stack([points, np.ones(len(points))])
        transformed = homogeneous_points @ transform.T
        return transformed[:, :2]

    def find_feature_matches(self, drone_points, drone_classes, map_points, map_classes):
        """Находит соответствия ТОЛЬКО для объектов в виртуальном кадре"""
        matches = []
        
        for i, (drone_point, drone_class) in enumerate(zip(drone_points, drone_classes)):
            # Ищем объекты того же класса в виртуальном кадре
            map_indices = [j for j, map_class in enumerate(map_classes) if map_class == drone_class]
            
            if not map_indices:
                continue
                
            # Находим ближайший объект того же класса
            min_distance = float('inf')
            best_match_idx = -1
            
            for j in map_indices:
                distance = np.linalg.norm(drone_point - map_points[j])
                if distance < min_distance:
                    min_distance = distance
                    best_match_idx = j
            
            # Если нашли достаточно близкий объект, добавляем в matches
            if best_match_idx != -1 and min_distance < 0.3:
                matches.append((i, best_match_idx, min_distance))
        
        print(f"DEBUG: Found {len(matches)} feature matches in virtual frame")
        return matches

    def ransac_match(self, drone_points, drone_classes, map_points, map_classes):
        """
        RANSAC алгоритм для сопоставления виртуального кадра карты с реальным кадром дрона
        """
        if len(drone_points) < 2 or len(map_points) < 2:
            return None, [], 0.0
        
        print(f"RANSAC: {len(drone_points)} drone points, {len(map_points)} map points")
        
        # Находим возможные соответствия
        feature_matches = self.find_feature_matches(drone_points, drone_classes, map_points, map_classes)
        print(f"DEBUG: Found {len(feature_matches)} feature matches")
        
        if len(feature_matches) < 2:
            return None, [], 0.0
        
        best_transform = None
        best_inliers = []
        best_error = float('inf')
        
        for iteration in range(self.num_iterations):
            # Выбираем случайные соответствия
            if len(feature_matches) >= 2:
                sample_matches = random.sample(feature_matches, 2)
            else:
                continue
            
            # Формируем точки для преобразования
            src_pts = []  # Точки на дроне
            dst_pts = []  # Точки на карте
            
            for match in sample_matches:
                drone_idx, map_idx, _ = match
                src_pts.append(drone_points[drone_idx])
                dst_pts.append(map_points[map_idx])
            
            # Вычисляем преобразование ОТ ДРОНА К КАРТЕ
            transform = self.find_similarity_transform(np.array(src_pts), np.array(dst_pts))
            
            if transform is None:
                continue
            
            # Применяем преобразование ко всем точкам дрона
            transformed_drone = self.apply_transform(drone_points, transform)
            
            if len(transformed_drone) == 0:
                continue
            
            # Находим инлаеры
            inliers = []
            total_error = 0
            
            for i, trans_point in enumerate(transformed_drone):
                drone_class = drone_classes[i]
                
                # Ищем ближайшую точку на карте того же класса
                min_distance = float('inf')
                for j, (map_point, map_class) in enumerate(zip(map_points, map_classes)):
                    if map_class == drone_class:
                        distance = np.linalg.norm(trans_point - map_point)
                        if distance < min_distance:
                            min_distance = distance
                
                if min_distance < self.inlier_threshold:
                    inliers.append(i)
                    total_error += min_distance
            
            if len(inliers) >= self.min_inliers:
                avg_error = total_error / len(inliers) if inliers else float('inf')
                
                if len(inliers) > len(best_inliers) or (
                    len(inliers) == len(best_inliers) and avg_error < best_error):
                    best_inliers = inliers
                    best_transform = transform
                    best_error = avg_error
        
        confidence = len(best_inliers) / len(drone_points) if best_inliers else 0.0
        print(f"DEBUG: RANSAC result: {len(best_inliers)} inliers, confidence {confidence:.3f}")
        
        return best_transform, best_inliers, confidence


class DroneCoordinateCorrector:
    def __init__(self, model_path, search_radius_meters=100, visualize=True):
        from ultralytics import YOLO
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель YOLO не найдена: {model_path}")
        self.model = YOLO(model_path)
        self.search_radius = search_radius_meters
        self.ransac = RANSACMatcher(
            num_iterations=1000,
            inlier_threshold=0.05,
            min_inliers=2
        )
        self.visualize = visualize
        if self.visualize:
            os.makedirs("visualization", exist_ok=True)
    
    def load_map_data(self, json_path):
        """Загружает данные карты из JSON файла"""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON файл карты не найден: {json_path}")
            
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        required_keys = ['objects', 'metadata']
        if not all(key in data for key in required_keys):
            raise ValueError("Неверная структура JSON файла карты")
            
        return data

    def calculate_centroid(self, mask):
        """Вычисляет центроид маски"""
        try:
            moments = cv2.moments(mask)
            if moments["m00"] != 0:
                centroid_x = int(moments["m10"] / moments["m00"])
                centroid_y = int(moments["m01"] / moments["m00"])
                return (centroid_x, centroid_y)
            else:
                contours, _ = cv2.findContours(
                    (mask > 0.5).astype(np.uint8), 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    x, y, w, h = cv2.boundingRect(contours[0])
                    return (x + w//2, y + h//2)
                else:
                    return None
        except Exception as e:
            print(f"Ошибка вычисления центроида: {e}")
            return None

    def detect_objects_on_frame(self, frame):
        """Детектирует объекты на кадре и возвращает нормализованные координаты"""
        if frame is None:
            raise ValueError("Пустой кадр")
            
        results = self.model(frame, conf=0.3, verbose=False)
        
        centroids = []
        classes = []
        
        for r in results:
            if r.boxes is not None and r.masks is not None:
                for i, (box, mask) in enumerate(zip(r.boxes, r.masks.data)):
                    class_id = int(box.cls[0].item())
                    class_name = self.model.names[class_id]
                    
                    mask_np = mask.cpu().numpy()
                    centroid = self.calculate_centroid(mask_np)
                    if centroid:
                        # Нормализуем относительно центра кадра (0.5, 0.5 - центр)
                        x_norm = (centroid[0] / frame.shape[1] - 0.5) * 2.0  # [-1, 1]
                        y_norm = (centroid[1] / frame.shape[0] - 0.5) * 2.0  # [-1, 1]
                        centroids.append([x_norm, y_norm])
                        classes.append(class_name)
        
        return np.array(centroids), classes

    def calculate_gps_distance(self, gps1, gps2):
        """Вычисляет расстояние между двумя GPS точками в метрах"""
        lat1, lon1 = radians(gps1[0]), radians(gps1[1])
        lat2, lon2 = radians(gps2[0]), radians(gps2[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))

        return 6371000 * c

    def create_virtual_frame_from_map(self, center_gps, map_data, drone_classes):
        """
        Создает виртуальный кадр из объектов карты в радиусе 100 метров от INS координат
        """
        map_objects = []
        map_classes = []
        
        expected_classes = set(drone_classes)
        class_id_to_name = {
            0: 'building', 1: 'field', 2: 'lake', 3: 'road', 4: 'zrail'
        }
        
        center_lat, center_lon = center_gps
        
        for obj in map_data['objects']:
            obj_gps = (obj['gps_coordinates']['latitude'], 
                    obj['gps_coordinates']['longitude'])
            
            # ФИЛЬТР 1: Только объекты в радиусе 100 метров
            distance = self.calculate_gps_distance(center_gps, obj_gps)
            if distance > self.search_radius:
                continue
            
            # ФИЛЬТР 2: Только объекты тех классов, которые есть на дроне
            class_id = obj.get('class_id', -1)
            obj_class = class_id_to_name.get(class_id, 'unknown')
            if obj_class not in expected_classes:
                continue
            
            # Нормализуем координаты относительно центра виртуального кадра
            frame_size_degrees = self.search_radius / 111000.0
            
            # ИСПРАВЛЕННОЕ преобразование (зеркалим по Y):
          
            x_norm = (obj_gps[1] - center_lon) / frame_size_degrees # восток -> +X
            y_norm = -(obj_gps[0] - center_lat) / frame_size_degrees # север -> -Y 
            map_objects.append([x_norm, y_norm])
            map_classes.append(obj_class)
        
        print(f"DEBUG: Virtual frame around {center_gps} ({self.search_radius}m radius)")
        print(f"DEBUG: Found {len(map_objects)} objects in virtual frame")
        
        return np.array(map_objects), map_classes

    def calculate_correction(self, transform, ins_gps):
        """Вычисляет коррекцию координат на основе преобразования RANSAC"""
        # Центр реального кадра дрона в нормализованных координатах [0, 0]
        drone_center = np.array([[0.0, 0.0]])
        
        # Применяем преобразование к центру кадра
        map_center = self.ransac.apply_transform(drone_center, transform)[0]
        
        print(f"DEBUG: Map center in virtual frame: {map_center}")
        
        # Конвертируем смещение из нормализованных координат в метры
        meters_per_normalized_unit = self.search_radius / 1.0
        
        # Смещение в метрах (учитываем зеркалирование по Y)
        displacement_x_m = map_center[0] * meters_per_normalized_unit   # Восток
        displacement_y_m =  map_center[1] * meters_per_normalized_unit  # Север (минус из-за зеркалирования)
        
        # Конвертируем метры в градусы
        lat_correction_deg = displacement_y_m / 111000.0  # Север = +широта
        lon_correction_deg = displacement_x_m / (111000.0 * cos(radians(ins_gps[0])))  # Восток = +долгота
        
        # Корректируем координаты
        corrected_lat = ins_gps[0] + lat_correction_deg
        corrected_lon = ins_gps[1] + lon_correction_deg
        
        print(f"DEBUG: Correction: ({displacement_x_m:.1f}, {displacement_y_m:.1f}) meters")
        
        return (corrected_lat, corrected_lon)
    def get_ins_coordinates(self, frame_data):
        """Вычисляет INS координаты из данных кадра"""
        if 'original_gps' in frame_data:
            base_gps = frame_data['original_gps']
        else:
            base_gps = frame_data['gps_coordinates']
        
        ins_error = frame_data.get('ins_error', {})
        ins_lat = base_gps['latitude'] - ins_error.get('lat_error', 0)
        ins_lon = base_gps['longitude'] - ins_error.get('lon_error', 0)
        
        return (ins_lat, ins_lon)
    def visualize_ransac_matching(self, drone_points, drone_classes, map_points, map_classes, 
                            transform, inliers, frame_number, confidence, ins_gps, corrected_gps):
        """Визуализация RANSAC сопоставления для отладки"""
        if not self.visualize:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Цвета для классов
        class_colors = {
            'building': 'red',
            'field': 'green', 
            'lake': 'blue',
            'road': 'gray',
            'zrail': 'orange'
        }
        
        # 1. Нормализованные координаты
        ax1.set_title(f'RANSAC Matching - Frame {frame_number}\nConfidence: {confidence:.3f}, Inliers: {len(inliers)}')
        ax1.set_xlim(-1.2, 1.2)
        ax1.set_ylim(-1.2, 1.2)
        ax1.grid(True)
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax1.set_aspect('equal')
        
        # Точки карты (синие кружки)
        for i, (point, cls) in enumerate(zip(map_points, map_classes)):
            color = class_colors.get(cls, 'black')
            ax1.scatter(point[0], point[1], c=color, marker='o', s=100, alpha=0.7, 
                    label='Map' if i == 0 else "")
            ax1.text(point[0] + 0.02, point[1] + 0.02, f'{cls}', fontsize=8, alpha=0.8)
        
        # Точки дрона до преобразования (красные квадраты)
        for i, (point, cls) in enumerate(zip(drone_points, drone_classes)):
            color = class_colors.get(cls, 'black')
            ax1.scatter(point[0], point[1], c=color, marker='s', s=100, alpha=0.7, 
                    label='Drone (orig)' if i == 0 else "")
            ax1.text(point[0] + 0.02, point[1] + 0.02, f'Drone {cls}', fontsize=8, alpha=0.8)
        
        # Преобразованные точки дрона (зеленые треугольники)
        if transform is not None:
            transformed_drone = self.ransac.apply_transform(drone_points, transform)
            for i, (point, cls) in enumerate(zip(transformed_drone, drone_classes)):
                color = class_colors.get(cls, 'black')
                marker = '^'
                size = 100
                alpha = 0.7
                
                # Подсвечиваем инлаеры
                if i in inliers:
                    marker = 'D'  # Ромб для инлаеров
                    size = 120
                    alpha = 1.0
                    edgecolor = 'green'
                else:
                    edgecolor = color
                    
                ax1.scatter(point[0], point[1], c=color, marker=marker, s=size, 
                        alpha=alpha, edgecolor=edgecolor, linewidth=2,
                        label='Drone (trans)' if i == 0 else "")
                
                # Линии для инлаеров
                if i in inliers:
                    original_point = drone_points[i]
                    ax1.plot([original_point[0], point[0]], [original_point[1], point[1]], 
                            'g--', alpha=0.6, linewidth=1)
        
        # Центр кадра (INS координаты)
        ax1.scatter(0, 0, c='purple', marker='*', s=200, label='INS Center')
        
        ax1.legend()
        ax1.set_xlabel('Normalized X')
        ax1.set_ylabel('Normalized Y')
        
        # 2. GPS координаты
        ax2.set_title('GPS Coordinates and Correction')
        
        # Собираем точки для границ
        all_lats = [ins_gps[0], corrected_gps[0]]
        all_lons = [ins_gps[1], corrected_gps[1]]
        
        # Объекты карты в GPS координатах
        frame_size_degrees = self.search_radius / 111000.0
        for point in map_points:
            lat = ins_gps[0] + point[1] * frame_size_degrees
            lon = ins_gps[1] + point[0] * frame_size_degrees
            all_lats.append(lat)
            all_lons.append(lon)
            
            # Рисуем точки карты
            ax2.scatter(lon, lat, c='blue', marker='o', s=50, alpha=0.6, label='Map Objects')
        
        # INS и скорректированные позиции
        ax2.scatter(ins_gps[1], ins_gps[0], c='red', marker='*', s=200, label='INS Position')
        ax2.scatter(corrected_gps[1], corrected_gps[0], c='green', marker='*', s=200, label='Corrected Position')
        
        # Линия коррекции
        ax2.plot([ins_gps[1], corrected_gps[1]], [ins_gps[0], corrected_gps[0]], 
                'r-', linewidth=3, label='Correction Vector')
        
        # Объекты дрона в GPS координатах (если есть преобразование)
        if transform is not None:
            transformed_drone = self.ransac.apply_transform(drone_points, transform)
            for i, (point, cls) in enumerate(zip(transformed_drone, drone_classes)):
                lat = ins_gps[0] - point[1] * frame_size_degrees
                lon = ins_gps[1] + point[0] * frame_size_degrees
                
                color = class_colors.get(cls, 'black')
                if i in inliers:
                    ax2.scatter(lon, lat, c=color, marker='D', s=80, alpha=0.8, 
                            label=f'Inlier {cls}' if i == 0 else "")
                else:
                    ax2.scatter(lon, lat, c=color, marker='^', s=60, alpha=0.5, 
                            label=f'Outlier {cls}' if i == 0 else "")
        
        # Настройка вида
        lat_margin = (max(all_lats) - min(all_lats)) * 0.2
        lon_margin = (max(all_lons) - min(all_lons)) * 0.2
        
        ax2.set_xlim(min(all_lons) - lon_margin, max(all_lons) + lon_margin)
        ax2.set_ylim(min(all_lats) - lat_margin, max(all_lats) + lat_margin)
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.grid(True)
        ax2.legend()
        
        # Информация о коррекции
        distance = self.calculate_gps_distance(ins_gps, corrected_gps)
        ax2.text(0.02, 0.98, f'Correction: {distance:.1f}m\nConfidence: {confidence:.3f}\nInliers: {len(inliers)}/{len(drone_points)}', 
                transform=ax2.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top')
        
        plt.tight_layout()
        
        # Сохраняем
        os.makedirs("ransac_visualization", exist_ok=True)
        plt.savefig(f'ransac_visualization/frame_{frame_number:04d}_ransac.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"💾 RANSAC визуализация сохранена: ransac_visualization/frame_{frame_number:04d}_ransac.png")
    def process_frame(self, frame, frame_data, map_data):
        """
        Основной метод обработки кадра
        Возвращает: (corrected_gps, confidence, processing_info)
        """
        frame_number = frame_data['frame_number']
        has_gps = frame_data.get('ins_error', {}).get('has_gps', True)
        
        processing_info = {
            'frame_number': frame_number,
            'has_gps': has_gps,
            'detected_objects': 0,
            'map_objects_in_radius': 0,
            'ransac_inliers': 0,
            'correction_method': 'gps' if has_gps else 'ins'
        }
        
        # Если есть GPS, используем истинные координаты
        if has_gps:
            true_gps = (frame_data['gps_coordinates']['latitude'], 
                    frame_data['gps_coordinates']['longitude'])
            print(f"📡 Кадр {frame_number}: GPS данные")
            return true_gps, 1.0, processing_info
        
        # INS коррекция
        ins_gps = self.get_ins_coordinates(frame_data)
        print(f"🔄 Кадр {frame_number}: INS коррекция")
        print(f"   INS координаты: {ins_gps}")
        
        # Детекция объектов на кадре дрона
        drone_centroids, drone_classes = self.detect_objects_on_frame(frame)
        processing_info['detected_objects'] = len(drone_centroids)
        print(f"   Обнаружено на дроне: {len(drone_centroids)} объектов")
        
        if len(drone_centroids) < 2:
            print("   ❌ Недостаточно объектов для коррекции")
            return ins_gps, 0.0, processing_info
        
        # Создаем виртуальный кадр из объектов карты в радиусе 100 метров
        map_objects, map_classes = self.create_virtual_frame_from_map(ins_gps, map_data, drone_classes)
        processing_info['map_objects_in_radius'] = len(map_objects)
        
        if len(map_objects) < 2:
            print("   ❌ Недостаточно объектов на карте для коррекции")
            return ins_gps, 0.0, processing_info
        
        # RANSAC матчинг
        transform, inliers, confidence = self.ransac.ransac_match(
            drone_centroids, drone_classes, map_objects, map_classes
        )
        processing_info['ransac_inliers'] = len(inliers)
        
        if transform is None or len(inliers) < 2:
            print("   ❌ RANSAC не нашел преобразование")
            if self.visualize:
                self.visualize_ransac_matching(
                    drone_centroids, drone_classes, 
                    map_objects, map_classes,
                    None, [], frame_number,
                    0.0, ins_gps, ins_gps  # corrected_gps = ins_gps при неудаче
                )
            processing_info['correction_method'] = 'ransac_failed'
            return ins_gps, 0.0, processing_info
        
        print(f"   ✅ RANSAC нашел {len(inliers)} инлаеров, уверенность: {confidence:.2f}")
        
        # ВЫЧИСЛЯЕМ КОРРЕКЦИЮ ПЕРЕД ВИЗУАЛИЗАЦИЕЙ
        corrected_gps = self.calculate_correction(transform, ins_gps)
        processing_info['correction_method'] = 'ransac_success'
        
        correction_distance = self.calculate_gps_distance(ins_gps, corrected_gps)
        print(f"   📏 Коррекция: {correction_distance:.1f} метров")
        
        # ТЕПЕРЬ ВИЗУАЛИЗАЦИЯ ПОСЛЕ ТОГО КАК corrected_gps ОПРЕДЕЛЕН
        if self.visualize and transform is not None:
            self.visualize_ransac_matching(
                drone_centroids, drone_classes, 
                map_objects, map_classes,
                transform, inliers, frame_number,
                confidence, ins_gps, corrected_gps
            )
        
        return corrected_gps, confidence, processing_info