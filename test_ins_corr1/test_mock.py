import cv2 
import numpy as np
import json
import os
import sys
import random
from datetime import datetime
from ransac import DroneCoordinateCorrector
import matplotlib.pyplot as plt


class MockRANSACCorrector:
    """Мок-класс для генерации RANSAC коррекции каждые 60 фреймов"""
    
    def __init__(self):
        self.correction_history = []
        self.last_correction_frame = -60
        self.corrected_error_after_ransac = 0
        
    def generate_periodic_correction(self, frame_data, frame_number):
        """Генерирует RANSAC коррекцию каждые 60 фреймов"""
        
        if frame_data.get('ins_error', {}).get('has_gps', True):
            # Для кадров с GPS возвращаем истинные координаты
            return (frame_data['gps_coordinates']['latitude'],
                   frame_data['gps_coordinates']['longitude']), 0.95
        
        # Истинные координаты
        true_lat = frame_data['gps_coordinates']['latitude']
        true_lon = frame_data['gps_coordinates']['longitude']
        
        # Вычисляем INS координаты (истинные + ошибка)
        lat_error = frame_data['ins_error'].get('total_lat_error', 0)
        lon_error = frame_data['ins_error'].get('total_lon_error', 0)
        
        ins_lat = true_lat + lat_error * 0.00001
        ins_lon = true_lon + lon_error * 0.00001
        
        # Проверяем, нужно ли применять коррекцию (каждые 60 фреймов)
        true_gps = (frame_data['gps_coordinates']['latitude'], frame_data['gps_coordinates']['longitude'])
        ins_gps = (ins_lat, ins_lon)
        ins_error = self.calculate_gps_distance(ins_gps, true_gps)  # ← ДОБАВИТЬ ЭТУ СТРОКУ
        
        # Проверяем, нужно ли применять коррекцию (каждые 60 фреймов)
        should_correct = (frame_number - self.last_correction_frame) >= 60
        
        if should_correct:
            # Применяем коррекцию - исправляем 80% ошибки
             # СЛУЧАЙНАЯ ЭФФЕКТИВНОСТЬ КОРРЕКЦИИ (30%-80%)
            correction_ratio = random.uniform(0.3, 0.7)  # Было фиксированное 0.8
            
            # ИНогда коррекция вообще не работает (10% случаев)
            if random.random() < 0.3:
                correction_ratio = 0.0  # Полный провал коррекции
            
            confidence = 0.85
            
            corrected_lat = ins_lat - lat_error * 0.00001 * correction_ratio
            corrected_lon = ins_lon - lon_error * 0.00001 * correction_ratio
              # ЗАПОМИНАЕМ ошибку после коррекции как новую базовую точку
            true_gps = (frame_data['gps_coordinates']['latitude'], frame_data['gps_coordinates']['longitude'])
            self.corrected_error_after_ransac = self.calculate_gps_distance((corrected_lat, corrected_lon), true_gps)
            if random.random() < 0.2:
                self.corrected_error_after_ransac += random.uniform(1.0, 5.0)  # Добавляем случайную ошибку
            self.last_correction_frame = frame_number
        else:
            # Без коррекции - растем от последней скорректированной ошибки
            frames_since_correction = frame_number - self.last_correction_frame
            
            # УВЕЛИЧИВАЕМ СКОРОСТЬ РОСТА ОШИБКИ
            growth_factor = frames_since_correction ** 1.5 * 0.02 # Было 0.02, стало 0.1 (в 5 раз быстрее)
            
            # УВЕЛИЧИВАЕМ СЛУЧАЙНЫЕ КОЛЕБАНИЯ
            random_fluctuation = random.uniform(-0.3, 2)  # Было (-0.5, 0.5), стало (-2.0, 2.0)
            
            # Новая ошибка = базовая скорректированная + рост + колебания
            current_error = self.corrected_error_after_ransac + growth_factor + random_fluctuation
            
            # Ограничиваем снизу (не может быть меньше базовой)
            current_error = max(self.corrected_error_after_ransac, current_error)
            
            # Пересчитываем координаты исходя из новой ошибки
            error_ratio = current_error / ins_error if ins_error > 0 else 1.0
            corrected_lat = ins_lat - lat_error * 0.00001 * error_ratio
            corrected_lon = ins_lon - lon_error * 0.00001 * error_ratio
            
            confidence = max(0.1, 0.3 - (frames_since_correction * 0.005))
        return (corrected_lat, corrected_lon), confidence
    def calculate_gps_distance(self, coord1, coord2):
            """Добавить этот метод в класс"""
            lat1, lon1 = coord1
            lat2, lon2 = coord2
            dlat = (lat2 - lat1) * 111320
            dlon = (lon2 - lon1) * 111320 * 0.7
            return np.sqrt(dlat**2 + dlon**2)
      
class UnifiedCorrectionVisualizer:
    def __init__(self, corrector, map_data, flight_data):
        self.corrector = corrector
        self.map_data = map_data
        self.flight_data = flight_data
        self.results = []
        self.mock_ransac = MockRANSACCorrector()
        
        # Цвета для визуализации
        self.colors = {
            'building': (0, 255, 0),      # зеленый
            'field': (0, 165, 255),       # оранжевый
            'lake': (255, 0, 0),          # синий
            'road': (128, 128, 128),      # серый
            'zrail': (0, 255, 255),       # желтый
            'default': (255, 255, 255)    # белый
        }
    
    def draw_detection(self, frame, centroids, classes):
 
        """Использует стандартный YOLO для детекции и отрисовки"""
        # Просто запускаем YOLO на кадре и рисуем стандартные результаты
        results = self.corrector.model(frame)  # Используем модель из corrector
        detected_frame = results[0].plot()     # Стандартная отрисовка YOLO
        
        return detected_frame
    def calculate_errors(self, corrected_gps, frame_data, has_gps):
        if has_gps:
            return 0, 0
        
        # Истинные координаты
        true_gps = (
            frame_data['gps_coordinates']['latitude'],
            frame_data['gps_coordinates']['longitude']
        )
        
        # INS координаты до коррекции
        ins_lat = frame_data['original_gps']['latitude'] if 'original_gps' in frame_data else true_gps[0]
        ins_lon = frame_data['original_gps']['longitude'] if 'original_gps' in frame_data else true_gps[1]
        ins_gps = (ins_lat, ins_lon)
        
        # Вычисляем ошибки
        ins_error = self.calculate_gps_distance(ins_gps, true_gps)
        corrected_error = self.calculate_gps_distance(corrected_gps, true_gps)
        
        return ins_error, corrected_error
    
    def calculate_gps_distance(self, coord1, coord2):
        """Вычисляет расстояние между GPS координатами в метрах"""
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        # Упрощенный расчет расстояния
        dlat = (lat2 - lat1) * 111320
        dlon = (lon2 - lon1) * 111320 * np.cos(np.radians((lat1 + lat2) / 2))
        
        return np.sqrt(dlat**2 + dlon**2)
    
    def get_ins_coordinates(self, frame_data):
        """Возвращает INS координаты (с ошибкой)"""
        if frame_data.get('ins_error', {}).get('has_gps', True):
            return (frame_data['gps_coordinates']['latitude'], 
                   frame_data['gps_coordinates']['longitude'])
        
        true_lat = frame_data['gps_coordinates']['latitude']
        true_lon = frame_data['gps_coordinates']['longitude']
        
        lat_error = frame_data['ins_error'].get('total_lat_error', 0)
        lon_error = frame_data['ins_error'].get('total_lon_error', 0)
        
        ins_lat = true_lat + lat_error * 0.00001
        ins_lon = true_lon + lon_error * 0.00001
        
        return ins_lat, ins_lon

    def create_dual_graphs(self, width, height, ins_error_history, corrected_error_history, frame_numbers):
       
        """Создает два графика: INS ошибка и INS+RANSAC коррекция"""
        graph = np.zeros((height, width, 3), dtype=np.uint8)
        
        if len(ins_error_history) < 2:
            return graph
        
        # Разделяем высоту для двух графиков
        graph_height = (height - 80) // 2
        graph1_y = 10
        graph2_y = graph1_y + graph_height + 25
        
        # График 1: Только INS ошибка + RANSAC для сравнения
        self._create_single_graph(graph, 10, graph1_y, width-20, graph_height, 
                                ins_error_history, frame_numbers, "INS Error (with RANSAC reference)", (255, 0, 0),
                                show_ransac_reference=True, ransac_error_history=corrected_error_history)  # ← ДОБАВИТЬ ПАРАМЕТРЫ
        
        # График 2: INS ошибка + RANSAC коррекция
        self._create_single_graph(graph, 10, graph2_y, width-20, graph_height, 
                                corrected_error_history, frame_numbers, "INS + RANSAC Correction", (0, 255, 0))
        
        # Статистика для обоих графиков
        if ins_error_history and corrected_error_history:
            self._add_statistics(graph, width, height, ins_error_history, corrected_error_history)
        
        return graph
    
    def _create_single_graph(self, graph, x, y, width, height, error_history, frame_numbers, title, color, 
                        show_ransac_reference=False, ransac_error_history=None):
        """Создает один график ошибок"""
        if len(error_history) < 2:
            return
        
        # Автомасштабирование графика
        max_error = max(error_history) * 1.2
        if show_ransac_reference and ransac_error_history:
            max_error = max(max_error, max(ransac_error_history) * 1.2)  # Учитываем RANSAC ошибку в масштабе
        min_error = max(0, min(error_history) * 0.8)
        avg_error = np.mean(error_history)
        
        # Рисуем рамку графика
        cv2.rectangle(graph, (x, y), (x+width, y+height), (100, 100, 100), 1)
        
        # Заголовок графика
        cv2.putText(graph, title, (x+10, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        # Шкала по оси Y с метками в метрах
        scale_steps = 5
        for i in range(scale_steps + 1):
            error_value = min_error + (max_error - min_error) * i / scale_steps
            y_pos = y + height - 20 - int(i / scale_steps * (height - 40))
            
            # Линия сетки
            cv2.line(graph, (x+20, y_pos), (x+width, y_pos), (50, 50, 50), 1)
            # Метка значения
            cv2.putText(graph, f"{error_value:.1f}m", (x+5, y_pos+3), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Ограничиваем историю для отображения
        display_history = min(50, len(error_history))
        start_idx = len(error_history) - display_history
        
        # ДОБАВИТЬ ЛИНИЮ RANSAC НА ПЕРВЫЙ ГРАФИК
        if show_ransac_reference and ransac_error_history and len(ransac_error_history) >= display_history:
            ransac_points = []
            for i in range(display_history):
                idx = start_idx + i
                if idx < len(ransac_error_history):
                    x_pos = x + 20 + (i * (width - 40) // display_history)
                    error_normalized = (ransac_error_history[idx] - min_error) / (max_error - min_error)
                    y_pos = y + height - 20 - int(error_normalized * (height - 40))
                    ransac_points.append((x_pos, y_pos))
            
            # Рисуем пунктирную линию RANSAC
            for i in range(1, len(ransac_points)):
                if i % 2 == 0:  # Пунктир
                    cv2.line(graph, ransac_points[i-1], ransac_points[i], (0, 255, 0), 1)
        
        # Подписи фреймов и секунд внизу графика
        if frame_numbers and len(frame_numbers) >= display_history:
            step = max(1, display_history // 5)
            for i in range(0, display_history, step):
                if start_idx + i < len(frame_numbers):
                    frame_num = frame_numbers[start_idx + i]
                    seconds = frame_num // 60
                    x_pos = x + 20 + (i * (width - 40) // display_history)
                    # Подпись фрейма
                    cv2.putText(graph, f"{frame_num}", (x_pos-10, y+height-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    # Подпись секунды под фреймом
                    cv2.putText(graph, f"{seconds}s", (x_pos-8, y+height+10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Рисуем основной график ошибок
        points = []
        for i in range(display_history):
            idx = start_idx + i
            x_pos = x + 20 + (i * (width - 40) // display_history)
            error_normalized = (error_history[idx] - min_error) / (max_error - min_error)
            y_pos = y + height - 20 - int(error_normalized * (height - 40))
            points.append((x_pos, y_pos))
        
        for i in range(1, len(points)):
            cv2.line(graph, points[i-1], points[i], color, 2)
        
        # Подписи статистики
        cv2.putText(graph, f"Max: {max(error_history):.1f}m", (x+width-80, y+15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(graph, f"Avg: {avg_error:.1f}m", (x+width-80, y+30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(graph, f"Min: {min(error_history):.1f}m", (x+width-80, y+45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def _add_statistics(self, graph, width, height, ins_errors, corrected_errors):
        """Добавляет статистику для обоих графиков"""
        # Статистика INS ошибки
        ins_avg = np.mean(ins_errors)
        ins_max = np.max(ins_errors)
        ins_min = np.min(ins_errors)
        
        # Статистика скорректированной ошибки
        corr_avg = np.mean(corrected_errors)
        corr_max = np.max(corrected_errors)
        corr_min = np.min(corrected_errors)
        
        # Общая статистика
        improvement = ins_avg - corr_avg
        improvement_percent = (improvement / ins_avg) * 100 if ins_avg > 0 else 0
        
        stats_text = [
            "=== ERROR STATISTICS ===",
            f"INS - Avg: {ins_avg:.1f}m, Max: {ins_max:.1f}m, Min: {ins_min:.1f}m",
            f"RANSAC - Avg: {corr_avg:.1f}m, Max: {corr_max:.1f}m, Min: {corr_min:.1f}m",
            f"Improvement: {improvement:+.1f}m ({improvement_percent:.1f}%)"
        ]
        
        # Выводим статистику в правом верхнем углу
        for i, line in enumerate(stats_text):
            y_pos = 60 + i * 15
            color = (255, 255, 255)
            if "===" in line:
                color = (255, 255, 0)
            elif "INS" in line:
                color = (255, 0, 0)
            elif "RANSAC" in line:
                color = (0, 255, 0)
            elif "Improvement" in line:
                color = (0, 255, 0) if improvement > 0 else (255, 0, 0)
            
            cv2.putText(graph, line, (width-300, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
            cv2.putText(graph, line, (width-300, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    def create_info_panel(self, video_height, frame_data, processing_info, 
                     corrected_gps, confidence, ins_error, corrected_error,
                     ins_error_history=None, corrected_error_history=None, 
                     confidence_history=None, frame_numbers_history=None):
        """Создает информационную панель с двумя графиками"""
        panel_width = 500
        panel_height = video_height
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        
        frame_num = frame_data['frame_number']
        timestamp = frame_data['timestamp_formatted']
        has_gps = processing_info['has_gps']
        
        # Основная информация
        info_lines = [
            f"=== DRONE NAVIGATION SYSTEM ===",
            f"FRAME: {frame_num} | TIME: {timestamp}",
            f"DATA TYPE: {'GPS' if has_gps else 'INS + RANSAC'}",
            f"CONFIDENCE: {confidence:.3f}",
            "",
            f"=== COORDINATES ===",
            f"Lat: {corrected_gps[0]:.6f}",
            f"Lon: {corrected_gps[1]:.6f}",
        ]
        
        if not has_gps:
            ins_gps = self.get_ins_coordinates(frame_data)
            info_lines.extend([
                "",
                f"=== INS CORRECTION ===",
                f"INS Lat: {ins_gps[0]:.6f}",
                f"INS Lon: {ins_gps[1]:.6f}",
                f"Method: RANSAC correction",
                f"Correction every: 60 frames (1 sec)",
            ])
        
        info_lines.extend([
            "",
            f"=== OBJECT DETECTION ===",
            f"Detected Objects: {processing_info.get('detected_objects', 0)}",
        ])
        
        if not has_gps:
            improvement = ins_error - corrected_error
            status = "IMPROVED" if improvement > 0 else "DEGRADED"
            
            info_lines.extend([
                "",
                f"=== ERROR ANALYSIS ===",
                f"INS Error: {ins_error:.1f}m",
                f"RANSAC Error: {corrected_error:.1f}m",
                f"Improvement: {improvement:+.1f}m",
                f"Status: {status}",
            ])
        
        # Выводим текст
        max_text_height = panel_height - 400  # Увеличили отступ для двух графиков
        for i, line in enumerate(info_lines):
            if i * 20 > max_text_height:
                break
                
            y_pos = 30 + i * 20
            color = (255, 255, 255)
            
            if "===" in line:
                color = (255, 255, 0)
            elif i < 5:
                color = (0, 255, 0) if has_gps else (0, 165, 255)
            elif "Status: IMPROVED" in line:
                color = (0, 255, 0)
            elif "Status: DEGRADED" in line:
                color = (0, 0, 255)
            
            cv2.putText(panel, line, (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
            cv2.putText(panel, line, (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Добавляем два графика для INS кадров
        if (not has_gps and ins_error_history is not None and 
            corrected_error_history is not None and frame_numbers_history is not None):
            
            graph_height = 350  # Высота для двух графиков
            graph_width = panel_width - 20
            graph_y = panel_height - graph_height - 10
            
            dual_graphs = self.create_dual_graphs(
                graph_width, graph_height, 
                ins_error_history, corrected_error_history, frame_numbers_history
            )
            
            panel[graph_y:graph_y+graph_height, 10:10+graph_width] = dual_graphs
        
        return panel

    def process_video(self, video_path, output_dir):
        """Основной процесс обработки видео с двумя графиками"""
        os.makedirs(output_dir, exist_ok=True)
        
        correction_data = {
            'video_info': self.flight_data['video_info'],
            'processing_info': {
                'processing_date': datetime.now().isoformat(),
                'search_radius_meters': 100
            },
            'corrected_frames': []
        }
        
        # Списки для истории ошибок
        ins_error_history = []
        corrected_error_history = []
        confidence_history = []
        frame_numbers_history = [] 
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return
        
        frame_count = 0
        processed_frames = 0
        
        print("Starting unified processing with dual graphs...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Находим данные для кадра
            frame_data = None
            for data in self.flight_data['flight_data']:
                if data['frame_number'] == frame_count:
                    frame_data = data
                    break
            
            if frame_data is None:
                frame_count += 1
                continue
            
            # Определяем тип данных (GPS или INS)
            has_gps = frame_data.get('ins_error', {}).get('has_gps', True)
            
            if has_gps:
                # GPS данные - используем истинные координаты
                corrected_gps = (frame_data['gps_coordinates']['latitude'],
                               frame_data['gps_coordinates']['longitude'])
                confidence = 0.95
                processing_info = {
                    'has_gps': True,
                    'correction_method': 'GPS (reference)',
                    'detected_objects': random.randint(1, 4)
                }
            else:
                # INS данные - генерируем RANSAC коррекцию каждые 60 фреймов
                corrected_gps, confidence = self.mock_ransac.generate_periodic_correction(frame_data, frame_count)
                processing_info = {
                    'has_gps': False,
                    'correction_method': 'RANSAC correction (every 60 frames)',
                    'detected_objects': random.randint(1, 4)
                }
            
            # Вычисляем ошибки
            ins_error, corrected_error = self.calculate_errors(
                corrected_gps, frame_data, has_gps
            )
            
            # Обновляем историю ошибок для INS кадров
            if not has_gps:
                ins_error_history.append(ins_error)
                corrected_error_history.append(corrected_error)
                confidence_history.append(confidence)
                frame_numbers_history.append(frame_count)
            
            # Визуализация детекций
            drone_centroids, drone_classes = self.corrector.detect_objects_on_frame(frame)
            vis_frame = self.draw_detection(frame, drone_centroids, drone_classes)
            
            # Информационная панель с двумя графиками
            info_panel = self.create_info_panel(
                vis_frame.shape[0], frame_data, processing_info,
                corrected_gps, confidence, ins_error, corrected_error,
                ins_error_history, corrected_error_history, confidence_history,
                frame_numbers_history
            )
            
            # Объединяем и показываем
            final_frame = np.hstack([vis_frame, info_panel])
            cv2.imshow('Drone Navigation - Dual Graphs', final_frame)
            
            # Сохраняем результаты
            frame_result = {
                'frame_number': frame_count,
                'has_gps': has_gps,
                'correction_method': processing_info['correction_method'],
                'confidence': confidence,
                'corrected_coordinates': {
                    'latitude': corrected_gps[0],
                    'longitude': corrected_gps[1]
                },
                'true_gps_coordinates': frame_data['gps_coordinates'],
                'processing_info': processing_info
            }
            
            if not has_gps:
                frame_result.update({
                    'ins_error': ins_error,
                    'corrected_error': corrected_error,
                    'improvement': ins_error - corrected_error
                })
            
            correction_data['corrected_frames'].append(frame_result)
            
            # Управление
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                print("Paused. Press any key to continue...")
                cv2.waitKey(0)
            elif key == ord('s'):
                screenshot_path = os.path.join(output_dir, f'screenshot_frame_{frame_count:04d}.png')
                cv2.imwrite(screenshot_path, final_frame)
                print(f"Screenshot saved: {screenshot_path}")
            
            frame_count += 1
            processed_frames += 1
            
            if frame_count % 50 == 0:
                ins_frames = len([f for f in correction_data['corrected_frames'] if not f['has_gps']])
                print(f"Progress: Frame {frame_count}, INS frames: {ins_frames}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Сохраняем результаты
        results_path = os.path.join(output_dir, 'unified_correction_results.json')
        with open(results_path, 'w') as f:
            json.dump(correction_data, f, indent=2)
        
        print(f"\n🎉 Processing completed!")
        print(f"📁 Processed frames: {processed_frames}")
        print(f"📊 Results saved to: {results_path}")


def main():
    # Конфигурация
    MODEL_PATH = '../runs/segment/yolov8n_gpu_updgrade_1/weights/best.pt'
    MAP_JSON = 'json_55d753137_37d282641_to_55d763143_37d308581_upd_yolo.json'
    FLIGHT_DATA_JSON = 'flight_data_visible_error.json'
    VIDEO_PATH = 'drone_flight_smooth.mp4'
    OUTPUT_DIR = 'unified_results'
    
    # Инициализация
    corrector = DroneCoordinateCorrector(MODEL_PATH, search_radius_meters=100, visualize=True)
    map_data = corrector.load_map_data(MAP_JSON)
    
    with open(FLIGHT_DATA_JSON, 'r') as f:
        flight_data = json.load(f)
    
    # Запуск
    visualizer = UnifiedCorrectionVisualizer(corrector, map_data, flight_data)
    visualizer.process_video(VIDEO_PATH, OUTPUT_DIR)

if __name__ == "__main__":
    main()