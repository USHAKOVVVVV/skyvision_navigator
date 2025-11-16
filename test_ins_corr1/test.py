import cv2
import numpy as np
import json
import os
import sys
from datetime import datetime
from ransac import DroneCoordinateCorrector


class UnifiedCorrectionVisualizer:
    def __init__(self, corrector, map_data, flight_data):
        self.corrector = corrector
        self.map_data = map_data
        self.flight_data = flight_data
        self.results = []
        
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
        """Рисует детекции на кадре"""
        result_frame = frame.copy()
        
        for i, (centroid_norm, cls_name) in enumerate(zip(centroids, classes)):
            # Конвертируем нормализованные координаты в пиксели
            centroid_x = int((centroid_norm[0] / 2.0 + 0.5) * frame.shape[1])
            centroid_y = int((centroid_norm[1] / 2.0 + 0.5) * frame.shape[0])
            centroid = (centroid_x, centroid_y)
            
            color = self.colors.get(cls_name, self.colors['default'])
            
            # Рисуем центроид
            cv2.circle(result_frame, centroid, 8, color, -1)
            cv2.circle(result_frame, centroid, 4, (255, 255, 255), -1)
            
            # Подпись класса
            cv2.putText(result_frame, cls_name, 
                       (centroid[0] + 10, centroid[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (0, 0, 0), 3)
            cv2.putText(result_frame, cls_name, 
                       (centroid[0] + 10, centroid[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       color, 1)
        
        return result_frame
    
    def calculate_errors(self, corrected_gps, frame_data, has_gps):
        """Вычисляет ошибки INS до и после коррекции"""
        if has_gps:
            return 0, 0
        
        # Истинные координаты
        true_gps = (
            frame_data['gps_coordinates']['latitude'],
            frame_data['gps_coordinates']['longitude']
        )
        
        # INS координаты до коррекции
        ins_gps = self.corrector.get_ins_coordinates(frame_data)
        
        # Вычисляем ошибки
        ins_error = self.corrector.calculate_gps_distance(ins_gps, true_gps)
        corrected_error = self.corrector.calculate_gps_distance(corrected_gps, true_gps)
        
        return ins_error, corrected_error
    
    def create_realtime_error_graph(self, width, height, ins_error_history, corrected_error_history, confidence_history):
        """Создает график ошибок в реальном времени"""
        graph = np.zeros((height, width, 3), dtype=np.uint8)
        
        if len(ins_error_history) < 2:
            return graph
        
        # Максимальная ошибка для масштабирования
        all_errors = ins_error_history + corrected_error_history
        max_error = max(max(all_errors) * 1.2, 50)
        max_error = min(max_error, 200)
        
        # Рисуем оси
        cv2.line(graph, (50, height-30), (width-10, height-30), (100, 100, 100), 1)
        cv2.line(graph, (50, 20), (50, height-30), (100, 100, 100), 1)
        
        # Подписи осей
        cv2.putText(graph, "Error (m)", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(graph, f"{max_error:.0f}m", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(graph, "0m", (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(graph, "Frames", (width//2-20, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Рисуем сетку
        for i in range(1, 5):
            y_pos = height-30 - int((i * max_error / 5) / max_error * (height-50))
            cv2.line(graph, (50, y_pos), (width-10, y_pos), (50, 50, 50), 1)
            cv2.putText(graph, f"{i*max_error/5:.0f}", (5, y_pos+3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        
        # Ограничиваем историю для отображения
        display_history = min(50, len(ins_error_history))
        start_idx = len(ins_error_history) - display_history
        
        # Рисуем график INS ошибки
        ins_points = []
        for i in range(display_history):
            idx = start_idx + i
            x = 50 + (i * (width - 60) // display_history)
            y = height-30 - int((ins_error_history[idx] / max_error) * (height-50))
            ins_points.append((x, y))
        
        for i in range(1, len(ins_points)):
            cv2.line(graph, ins_points[i-1], ins_points[i], (0, 100, 255), 2)
        
        # Рисуем график скорректированной ошибки
        if len(corrected_error_history) >= 2:
            corrected_points = []
            for i in range(display_history):
                idx = start_idx + i
                if idx < len(corrected_error_history):
                    x = 50 + (i * (width - 60) // display_history)
                    y = height-30 - int((corrected_error_history[idx] / max_error) * (height-50))
                    corrected_points.append((x, y))
            
            for i in range(1, len(corrected_points)):
                cv2.line(graph, corrected_points[i-1], corrected_points[i], (0, 255, 0), 2)
        
        # Рисуем точки уверенности
        if confidence_history:
            conf_points = []
            for i in range(display_history):
                idx = start_idx + i
                if idx < len(confidence_history):
                    x = 50 + (i * (width - 60) // display_history)
                    y = 20 + int((1 - confidence_history[idx]) * 30)
                    conf_points.append((x, y))
            
            for point in conf_points:
                cv2.circle(graph, point, 2, (255, 255, 0), -1)
        
        # Легенда
        legend_y = 15
        cv2.putText(graph, "INS Error", (width-120, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)
        cv2.putText(graph, "Corrected", (width-120, legend_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(graph, "Confidence", (width-120, legend_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Текущие значения
        if ins_error_history:
            current_ins_error = ins_error_history[-1]
            current_corrected_error = corrected_error_history[-1] if corrected_error_history else 0
            current_confidence = confidence_history[-1] if confidence_history else 0
            improvement = current_ins_error - current_corrected_error
            
            stats_text = [
                f"INS: {current_ins_error:.1f}m",
                f"Corr: {current_corrected_error:.1f}m", 
                f"Conf: {current_confidence:.2f}",
                f"Imp: {improvement:+.1f}m"
            ]
            
            for i, text in enumerate(stats_text):
                color = (0, 255, 0) if "Imp: +" in text else (255, 255, 255)
                cv2.putText(graph, text, (width-120, legend_y+60 + i*15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        
        cv2.putText(graph, "REALTIME ERROR TRACKING", (width//2-80, 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return graph

    def create_info_panel(self, video_height, frame_data, processing_info, 
                     corrected_gps, confidence, ins_error, corrected_error,
                     ins_error_history=None, corrected_error_history=None, confidence_history=None):
        """Создает информационную панель с графиком в реальном времени"""
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
            f"DATA TYPE: {'GPS' if has_gps else 'INS + CORRECTION'}",
            f"CONFIDENCE: {confidence:.3f}",
            "",
            f"=== COORDINATES ===",
            f"Lat: {corrected_gps[0]:.6f}",
            f"Lon: {corrected_gps[1]:.6f}",
        ]
        
        if not has_gps:
            ins_gps = self.corrector.get_ins_coordinates(frame_data)
            info_lines.extend([
                "",
                f"=== INS CORRECTION ===",
                f"INS Lat: {ins_gps[0]:.6f}",
                f"INS Lon: {ins_gps[1]:.6f}",
                f"Method: {processing_info['correction_method']}",
            ])
        
        info_lines.extend([
            "",
            f"=== OBJECT DETECTION ===",
            f"Drone Objects: {processing_info['detected_objects']}",
            f"Map Objects: {processing_info['map_objects_in_radius']}",
            f"RANSAC Inliers: {processing_info['ransac_inliers']}",
        ])
        
        if not has_gps:
            improvement = ins_error - corrected_error
            status = "IMPROVED" if improvement > 0 else "DEGRADED"
            
            info_lines.extend([
                "",
                f"=== ERROR ANALYSIS ===",
                f"INS Error: {ins_error:.1f}m",
                f"Corrected: {corrected_error:.1f}m",
                f"Improvement: {improvement:+.1f}m",
                f"Status: {status}",
            ])
        
        # Выводим текст
        max_text_height = panel_height - 200
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
        
        # Добавляем график для INS кадров
        if (not has_gps and ins_error_history is not None and 
            corrected_error_history is not None and confidence_history is not None):
            
            graph_height = 180
            graph_width = panel_width - 20
            graph_y = panel_height - graph_height - 10
            
            realtime_graph = self.create_realtime_error_graph(
                graph_width, graph_height, 
                ins_error_history, corrected_error_history, confidence_history
            )
            
            panel[graph_y:graph_y+graph_height, 10:10+graph_width] = realtime_graph
        
        return panel

    def process_video(self, video_path, output_dir):
        """Основной процесс обработки видео с графиком в реальном времени"""
        os.makedirs(output_dir, exist_ok=True)
        
        correction_data = {
            'video_info': self.flight_data['video_info'],
            'processing_info': {
                'processing_date': datetime.now().isoformat(),
                'search_radius_meters': self.corrector.search_radius
            },
            'corrected_frames': []
        }
        
        # Списки для истории ошибок
        ins_error_history = []
        corrected_error_history = []
        confidence_history = []
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return
        
        frame_count = 0
        processed_frames = 0
        
        print("Starting unified processing with realtime error tracking...")
        
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
            
            # Обрабатываем кадр
            corrected_gps, confidence, processing_info = self.corrector.process_frame(
                frame, frame_data, self.map_data
            )
            
            # Вычисляем ошибки
            ins_error, corrected_error = self.calculate_errors(
                corrected_gps, frame_data, processing_info['has_gps']
            )
            
            # Обновляем историю ошибок для INS кадров
            if not processing_info['has_gps']:
                ins_error_history.append(ins_error)
                corrected_error_history.append(corrected_error)
                confidence_history.append(confidence)
            
            # Визуализация
            drone_centroids, drone_classes = self.corrector.detect_objects_on_frame(frame)
            vis_frame = self.draw_detection(frame, drone_centroids, drone_classes)
            
            # Информационная панель с графиком
            info_panel = self.create_info_panel(
                vis_frame.shape[0], frame_data, processing_info,
                corrected_gps, confidence, ins_error, corrected_error,
                ins_error_history, corrected_error_history, confidence_history
            )
            
            # Объединяем и показываем
            final_frame = np.hstack([vis_frame, info_panel])
            cv2.imshow('Drone Coordinate Correction - Realtime Error Tracking', final_frame)
            
            # Сохраняем результаты
            frame_result = {
                'frame_number': frame_count,
                'has_gps': processing_info['has_gps'],
                'correction_method': processing_info['correction_method'],
                'confidence': confidence,
                'corrected_coordinates': {
                    'latitude': corrected_gps[0],
                    'longitude': corrected_gps[1]
                },
                'true_gps_coordinates': frame_data['gps_coordinates'],
                'processing_info': processing_info
            }
            
            if not processing_info['has_gps']:
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
        
        # Создаем финальный анализ
        print("\n📊 Creating final error analysis...")
        self.create_final_error_analysis(correction_data, output_dir)
        
        # Выводим статистику
        self.print_final_statistics(correction_data)
        
        print(f"\n🎉 Processing completed!")
        print(f"📁 Processed frames: {processed_frames}")
        print(f"📊 Results saved to: {results_path}")

    def create_final_error_analysis(self, correction_data, output_dir):
        """Создает финальный график анализа ошибок"""
        frames_with_ins = [f for f in correction_data['corrected_frames'] if not f['has_gps']]
        
        if not frames_with_ins:
            print("No INS frames for error analysis")
            return
        
        frame_numbers = [f['frame_number'] for f in frames_with_ins]
        ins_errors = [f['ins_error'] for f in frames_with_ins]
        corrected_errors = [f['corrected_error'] for f in frames_with_ins]
        confidences = [f['confidence'] for f in frames_with_ins]
        improvements = [f['improvement'] for f in frames_with_ins]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # График 1: Ошибки во времени
        ax1.plot(frame_numbers, ins_errors, 'r-', linewidth=2, label='INS Error', alpha=0.8, marker='o', markersize=3)
        ax1.plot(frame_numbers, corrected_errors, 'g-', linewidth=2, label='Corrected Error', alpha=0.8, marker='s', markersize=3)
        
        ax1.fill_between(frame_numbers, ins_errors, corrected_errors, 
                        where=np.array(corrected_errors) < np.array(ins_errors), 
                        facecolor='green', alpha=0.3, label='Improvement Area')
        ax1.fill_between(frame_numbers, ins_errors, corrected_errors, 
                        where=np.array(corrected_errors) >= np.array(ins_errors), 
                        facecolor='red', alpha=0.2, label='Degradation Area')
        
        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('Position Error (meters)')
        ax1.set_title('INS vs Corrected Position Errors Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Статистика
        avg_ins_error = np.mean(ins_errors)
        avg_corrected_error = np.mean(corrected_errors)
        avg_improvement = np.mean(improvements)
        success_rate = sum(1 for imp in improvements if imp > 0) / len(improvements) * 100
        
        stats_text = f'Statistics:\nAvg INS Error: {avg_ins_error:.1f}m\nAvg Corrected: {avg_corrected_error:.1f}m\nAvg Improvement: {avg_improvement:+.1f}m\nSuccess Rate: {success_rate:.1f}%'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
                verticalalignment='top', fontsize=10)
        
        # График 2: Уверенность и улучшения
        ax2.bar(frame_numbers, improvements, color=['green' if x > 0 else 'red' for x in improvements], 
                alpha=0.6, label='Improvement')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(frame_numbers, confidences, 'b-', linewidth=2, label='RANSAC Confidence', alpha=0.8)
        ax2_twin.set_ylabel('Confidence', color='blue')
        ax2_twin.set_ylim(0, 1)
        ax2_twin.tick_params(axis='y', labelcolor='blue')
        
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('Improvement (meters)')
        ax2.set_title('Improvement and RANSAC Confidence')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        
        final_plot_path = os.path.join(output_dir, 'final_error_analysis.png')
        plt.savefig(final_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Final error analysis saved: {final_plot_path}")

    def print_final_statistics(self, correction_data):
        """Выводит финальную статистику"""
        frames_with_ins = [f for f in correction_data['corrected_frames'] if not f['has_gps']]
        
        if not frames_with_ins:
            print("No INS frames processed")
            return
        
        improvements = [f['improvement'] for f in frames_with_ins]
        confidences = [f['confidence'] for f in frames_with_ins]
        
        successful_corrections = sum(1 for imp in improvements if imp > 0)
        success_rate = successful_corrections / len(improvements) * 100
        avg_improvement = np.mean(improvements)
        avg_confidence = np.mean(confidences)
        
        print("\n" + "="*50)
        print("FINAL PROCESSING STATISTICS")
        print("="*50)
        print(f"Total frames processed: {len(correction_data['corrected_frames'])}")
        print(f"INS correction frames: {len(frames_with_ins)}")
        print(f"Successful corrections: {successful_corrections} ({success_rate:.1f}%)")
        print(f"Average improvement: {avg_improvement:+.1f}m")
        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"Best improvement: {max(improvements):.1f}m")
        print(f"Worst degradation: {min(improvements):.1f}m")
        
        methods = {}
        for frame in frames_with_ins:
            method = frame['correction_method']
            methods[method] = methods.get(method, 0) + 1
        
        print(f"\nCorrection methods distribution:")
        for method, count in methods.items():
            percentage = count / len(frames_with_ins) * 100
            print(f"  {method}: {count} frames ({percentage:.1f}%)")


def main():
    # Конфигурация
    MODEL_PATH = '../runs/segment/yolov8n_gpu_updgrade_1/weights/best.pt'
    MAP_JSON = 'json_55d948091_37d941703_to_55d967844_37d996474_upd_yolo.json'
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