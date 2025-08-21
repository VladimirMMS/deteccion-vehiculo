# -*- coding: utf-8 -*-
import cv2
import torch
from time import sleep

# ---- Config ----
INFER_SIZE = 640
PREFERRED_INDEXES = [1, 0, 2, 3]  # orden de prueba de índices
TRY_BY_NAME = False               # pon True si conoces el nombre exacto del dispositivo
DEVICE_NAME = "video=Integrated Camera"  # ejemplo: "video=USB2.0 HD UVC WebCam"

# ---- Carga modelo YOLOv5s (COCO) ----
model = torch.hub.load(
    'ultralytics/yolov5',
    'yolov5s',
    source='github',
    trust_repo=True,
    force_reload=False
)
model.conf = 0.25
model.iou = 0.45

def try_open(index, backend):
    cap = cv2.VideoCapture(index, backend)
    if cap.isOpened():
        # a veces isOpened es True pero read() falla; probamos un frame
        ok, _ = cap.read()
        if ok:
            print(f"✅ Cámara abierta: index={index}, backend={backend}")
            return cap
        cap.release()
    print(f"❌ No abre: index={index}, backend={backend}")
    return None

def open_any_camera():
    # 1) Media Foundation primero (Windows moderno)
    for idx in PREFERRED_INDEXES:
        cap = try_open(idx, cv2.CAP_MSMF)
        if cap: return cap

    # 2) DirectShow después
    for idx in PREFERRED_INDEXES:
        cap = try_open(idx, cv2.CAP_DSHOW)
        if cap: return cap

    # 3) Por nombre (DirectShow) si lo habilitas y conoces el nombre exacto
    if TRY_BY_NAME:
        cap = cv2.VideoCapture(DEVICE_NAME, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                print(f"✅ Cámara por nombre: {DEVICE_NAME}")
                return cap
            cap.release()
        print(f"❌ No abre por nombre: {DEVICE_NAME}")

    return None

cap = open_any_camera()
if cap is None:
    raise RuntimeError(
        "No se pudo abrir ninguna cámara. Revisa:\n"
        "• Permisos en Configuración > Privacidad y seguridad > Cámara\n"
        "• Cierra apps que usen la cámara (Teams/Zoom/OBS/Discord)\n"
        "• Prueba otros puertos USB o índices (0,1,2...)\n"
        "• Actualiza controladores de la webcam"
    )

# (opcional) fija resolución si tu cámara lo soporta
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
sleep(0.1)  # pequeño delay

while True:
    ok, frame = cap.read()
    if not ok:
        print("⚠️ No se pudo leer frame (¿dispositivo desconectado?)")
        break

    results = model(frame, size=INFER_SIZE)
    annotated = results.render()[0]
    cv2.imshow('YOLOv5 - Cámara', annotated)

    if cv2.waitKey(5) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
