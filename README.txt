SkyVision/                        # Корневая директория проекта
.
├── data/                         # Данные для тестирования ransak
│   ├── drone_images/             # Изображения с дрона для тестирования
│
├── old_data/                 # Архивные данные предыдущих экспериментов
│   
├── ransac_correction/            # Модуль коррекции координат по RANSAC
│   ├── coordinate_corrector.py   # Основной класс корректора координат
│   ├── detection_result.jpg      # Результат детекции объектов на изображении
│   ├── drone_detection_result.jpg # Визуализация детекции на фото дрона
│   ├── matching_visualization.jpg # Визуализация сопоставления точек RANSAC
│   └── ransac_test.ipynb         # Jupyter notebook для тестирования RANSAC
│
├── runs/segment/                 # Результаты обучения сегментации YOLO
│   ├── train/                    # Результаты тренировочного запуска
│   ├── val/                      # Результаты валидационного запуска
│   ├── yolov8n_gpu_simple_1/    # Модель YOLOv8n (simple версия)
│   └── yolov8n_gpu_updgrade_1/  # Модель YOLOv8n (upgraded версия)
│
├── Scripts_geomap/              # Директория со скриптами для работы для парсинга карты
    ├── generate_cords/          # Директория для генерации координат с видео
    ├── output_img/              # Директория для выходных изображений склеиной карты
    ├── output/json              # Директория с json результатами обработки YOLO всей карты
    ├── output_yolo_img/         # Директория для изображений с разметкой YOLO и центроидами
    ├── generate_img_for_model.py# Скрипт для парсинга фото с яндекса для разметки
    ├── generate_video.py        # Скрипт для склейки видео имитации полета дрона
    ├── download_map.py          # Скрипт для парсинга и склейки карты 
    ├── process_map_yolo.py      # Скрипт для обработки YOLO и класиификации объектов.Делает визуализацию полигонов и json 
    ├── yolo_video.py            # Скрипт для запуска детекции по видео 
    └── visualize_centroids.py   # Скрипт для визуализации центроид объектов из json
    drone_flight_smooth.mp4(gitigniore) 

