import cv2
from ultralytics import YOLO

# Загружаем вашу модель YOLO
model = YOLO('../runs/segment/yolov8n_gpu_updgrade_1/weights/best.pt')

# Открываем видео с дроном
cap = cv2.VideoCapture('drone_flight_smooth.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Детекция на кадре
    results = model(frame)
    
    # Рисуем результаты
    detected_frame = results[0].plot()
    
    # Показываем результат
    cv2.imshow('YOLO Detection', detected_frame)
    
    # Выход по 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()