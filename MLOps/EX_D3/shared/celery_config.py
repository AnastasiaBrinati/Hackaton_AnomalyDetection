from shared.celery_app import celery_app
import time
from ultralytics import YOLO
import numpy as np
import base64
import cv2
import json

# Variabile globale per il modello (caricato solo quando necessario)
model = None

def get_model():
    """Carica il modello YOLO solo quando necessario (lazy loading)"""
    global model
    if model is None:
        model = YOLO("yolov8n.pt")
        print("Modello YOLO caricato")
    return model

@celery_app.task(name="detect_objects")
def detect_objects(image_b64):
    try:
        # Carica il modello solo quando necessario
        model = get_model()
        
        # Decodificare l'immagine base64 in un array numpy
        image_bytes = base64.b64decode(image_b64)
        image = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        # Inferenza con YOLO
        results = model.predict(image) #conf=0.25, iou=0.7, imgsz=640

        # Estrarre e formattare i risultati
        results_list = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                class_name = model.names[class_id]
                results_list.append({
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2]
                })
        return json.dumps(results_list)
    
    except Exception as e:
        print(f"Errore durante l'inferenza: {e}")
        return json.dumps({"error": str(e)})
    



