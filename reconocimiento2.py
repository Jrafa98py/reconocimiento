import cv2  # captura videos desde la camara
from deepface import DeepFace  # analiza el genero
import numpy as np  # matrices y arreglos númericos
import time  # para manejaar los tiempos de captura
import os  # para los archivos y carpetas
from ultralytics import YOLO  # detecta objetos en este caso los vehiculos

# Variables globales para el manejo del mouse
is_mouse_down = False
mouse_x, mouse_y = -1, -1

def mouse_callback(event, x, y, flags, param):
    """
    Función de callback para detectar clics del mouse.
    """
    global is_mouse_down, mouse_x, mouse_y
    if event == cv2.EVENT_LBUTTONDOWN:
        is_mouse_down = True
        mouse_x, mouse_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        is_mouse_down = False
        mouse_x, mouse_y = -1, -1

# Crear carpetas para guardar imágenes si no existen
if not os.path.exists("capturas"):
    os.makedirs("capturas")

# Inicialización para reconocimiento facial
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
last_genders = {}
last_capture_times = {}
capture_interval = 120  # 2 minutos en segundos

# Inicialización para reconocimiento de vehículos
last_vehicle_captures = {}
vehicle_capture_interval = 60  # 1 minuto en segundos para vehículos

# Cargar modelo YOLO para detección de vehículos
yolo_model = YOLO('yolov8n.pt')

# Iniciar cámara (si usas un DVR, cambia 0 por la URL RTSP)
cap = cv2.VideoCapture(0)

# Crear la ventana y asignar la función de callback del mouse
cv2.namedWindow('Reconocimiento')
cv2.setMouseCallback('Reconocimiento', mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detección de vehículos
    results = yolo_model(frame, verbose=False)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            class_id = int(box.cls[0])
            class_name = yolo_model.names[class_id]

            if class_name in ['car', 'motorcycle', 'truck', 'bus']:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, class_name.upper(), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Lógica para capturar y guardar la imagen del vehículo
                vehicle_id = f"{class_name}_{int(x1)}_{int(y1)}"
                current_time = time.time()
                
                if current_time - last_vehicle_captures.get(vehicle_id, 0) > vehicle_capture_interval:
                    filename = f"capturas/{class_name}_{int(current_time)}.jpg"
                    cv2.imwrite(filename, frame)
                    last_vehicle_captures[vehicle_id] = current_time

    # --- Detección facial y de género ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for i, (x, y, w, h) in enumerate(faces):
        face_img = frame[y:y+h, x:x+w]

        try:
            result = DeepFace.analyze(face_img, actions=['gender'], enforce_detection=False)
            gender = result[0]['dominant_gender']

            label = 'HOMBRE' if gender == 'Man' else 'MUJER'
            color = (255, 0, 0) if gender == 'Man' else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            face_id = f"{x}_{y}_{w}_{h}"
            current_time = time.time()

            if (face_id not in last_genders or last_genders[face_id] != gender or
                    current_time - last_capture_times.get(face_id, 0) > capture_interval):
                
                filename = f"capturas/{label}_{int(current_time)}.jpg"
                cv2.imwrite(filename, frame)
                last_genders[face_id] = gender
                last_capture_times[face_id] = current_time

        except Exception as e:
            cv2.putText(frame, "Error al analizar", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # --- Lógica de la interfaz de usuario ---
    # 1. Mensaje para salir del programa
    cv2.putText(frame, "Presiona 'q' para salir", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 2. Botón de salida
    button_size = 25
    button_x = frame.shape[1] - button_size - 10
    button_y = 10
    
    cv2.rectangle(frame, (button_x, button_y), (button_x + button_size, button_y + button_size), (0, 0, 255), cv2.FILLED)
    cv2.putText(frame, "X", (button_x + 5, button_y + button_size - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 3. Lógica para detectar el clic en el botón
    if is_mouse_down and mouse_x != -1 and mouse_y != -1:
        if button_x <= mouse_x <= button_x + button_size and button_y <= mouse_y <= button_y + button_size:
            print("Cerrando el programa por clic en el botón.")
            break
            
    # Mostrar el frame procesado
    cv2.imshow('Reconocimiento', frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Cerrando el programa por la tecla 'q'.")
        break

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()