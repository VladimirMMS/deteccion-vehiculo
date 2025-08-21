# -*- coding: utf-8 -*-
import pathlib
pathlib.PosixPath = pathlib.WindowsPath  # por si el .pt viene de Linux/Mac

import cv2
import torch
import time
import numpy as np
from pathlib import Path

# ============== CONFIG ==============
# Físicos: pon aquí los índices reales de tus 4 cámaras
CAMS = [
    {"index": 0, "backend": cv2.CAP_MSMF, "name": "NORTE"},  # detector
    {"index": 1, "backend": cv2.CAP_MSMF, "name": "SUR"},    # detector
    {"index": 2, "backend": cv2.CAP_MSMF, "name": "ESTE"},   # detector
    {"index": 3, "backend": cv2.CAP_MSMF, "name": "OESTE"},  # monitoreo
]
DETECTED_CAM_IDS = [0, 1, 2]   # estas 3 corren detección
MONITOR_CAM_IDS  = [3]         # esta queda sin detector

# Modelo: usa tus pesos o COCO
USE_COCO  = False
ROOT      = Path(r"C:/Users/vladi/OneDrive/Desktop/detection-objects")
WEIGHTS   = ROOT / "best.pt"        # tus pesos (si USE_COCO=False)
YOLO_VAR  = 'yolov5s'               # si USE_COCO=True
IMG_SIZE  = 960                     # 640/960/1280 (más grande => más recall)
CONF_TH   = 0.18
IOU_TH    = 0.45

# Filtrado: 4 gomas + bus
VEHICLE_NAME_SET = {"car", "truck", "bus", "auto", "automovil", "automóvil",
                    "camioneta", "pickup", "van", "suv", "taxi", "guagua", "autobús", "autobus"}

# descartamos cajas muy pequeñas (ajusta a tu escena/resolución)
MIN_BOX_AREA = 9000

# Mostrar ventanas
SHOW = True
# ===================================


def try_open(index: int, backend: int):
    cap = cv2.VideoCapture(index, backend)
    if cap.isOpened():
        ok, _ = cap.read()
        if ok:
            return cap
        cap.release()
    alt = cv2.CAP_DSHOW if backend == cv2.CAP_MSMF else cv2.CAP_MSMF
    cap = cv2.VideoCapture(index, alt)
    if cap.isOpened():
        ok, _ = cap.read()
        if ok:
            return cap
        cap.release()
    raise RuntimeError(f"No pude abrir cámara index={index} con MSMF/DSHOW")


def load_model():
    if USE_COCO:
        m = torch.hub.load('ultralytics/yolov5', YOLO_VAR, trust_repo=True, source='github')
    else:
        m = torch.hub.load('ultralytics/yolov5', 'custom', path=str(WEIGHTS), trust_repo=True, source='github')
    m.conf = CONF_TH
    m.iou  = IOU_TH
    return m


def get_vehicle_ids(names):
    """
    Mapea ids de clases cuyo nombre coincide con 4 gomas + bus.
    Para COCO, devolverá típicamente {2(car),5(bus),7(truck)}.
    """
    if isinstance(names, dict):
        name_map = {int(k): str(v).lower() for k, v in names.items()}
    else:
        name_map = {i: str(n).lower() for i, n in enumerate(names)}
    vids = []
    for i, n in name_map.items():
        base = n.strip()
        if base in VEHICLE_NAME_SET or any(k in base for k in ["car", "truck", "bus", "auto", "van", "suv", "taxi", "pickup"]):
            vids.append(i)
    return sorted(set(vids)), name_map


def count_and_boxes(det_tensor, vehicle_ids, min_area, frame_shape):
    """
    det_tensor: results.xyxy[0] -> [x1,y1,x2,y2,conf,cls]
    Filtra por vehicle_ids y área mínima.
    """
    if det_tensor is None or len(det_tensor) == 0:
        return 0, []
    boxes = []
    H, W = frame_shape[:2]
    for *xyxy, conf, cls in det_tensor:
        cls = int(cls.item()) if hasattr(cls, "item") else int(cls)
        if vehicle_ids and cls not in vehicle_ids:
            continue
        x1, y1, x2, y2 = [float(v.item() if hasattr(v, "item") else v) for v in xyxy]
        if (x2 - x1) * (y2 - y1) < min_area:
            continue
        boxes.append((int(x1), int(y1), int(x2), int(y2), float(conf), cls))
    return len(boxes), boxes


def draw_boxes(img, boxes, names, as_green=True):
    out = img.copy()
    color = (0, 255, 0) if as_green else (0, 0, 255)
    for (x1, y1, x2, y2, conf, cls) in boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{names.get(cls, str(cls))} {conf:.2f}"
        cv2.putText(out, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out


def mosaic_2x2(frames, W=640, H=360, titles=None, counts=None):
    # reescala y compone
    views = []
    for cid in range(4):
        img = frames.get(cid, np.zeros((H, W, 3), dtype=np.uint8))
        img = cv2.resize(img, (W, H))
        name = (titles or {}).get(cid, f"CAM {cid}")
        cnt  = (counts or {}).get(cid, 0)
        cv2.putText(img, f"{name}   count={cnt}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        views.append(img)
    top = np.hstack([views[0], views[1]])
    bot = np.hstack([views[2], views[3]])
    return np.vstack([top, bot])


def main():
    cv2.setUseOptimized(True)

    # Abre cámaras
    caps = {}
    for cam_id, cfg in enumerate(CAMS):
        caps[cam_id] = try_open(cfg["index"], cfg["backend"])
        caps[cam_id].set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Modelo
    model = load_model()
    names_raw = model.names
    names = names_raw if isinstance(names_raw, dict) else {i: n for i, n in enumerate(names_raw)}
    vehicle_ids, names_lc = get_vehicle_ids(names_raw)

    if vehicle_ids:
        # filtrado en el modelo (ahorra postproceso)
        model.classes = vehicle_ids
        print("Clases de vehículos usadas:", {i: names[i] for i in vehicle_ids})
    else:
        model.classes = None
        print("⚠️ No pude mapear classes de vehículos por nombre; se contarán TODAS las clases.")

    # Warmup con primera cámara detectora
    for cam_id in DETECTED_CAM_IDS:
        ok, frame = caps[cam_id].read()
        if ok:
            with torch.inference_mode():
                _ = model(frame, size=IMG_SIZE)
            break

    last_frames = {cid: None for cid in range(4)}
    last_counts = {cid: 0   for cid in range(4)}

    with torch.inference_mode():
        while True:
            # 1) Captura
            for cam_id, cap in caps.items():
                ok, frame = cap.read()
                last_frames[cam_id] = frame if ok else np.zeros((480, 640, 3), dtype=np.uint8)

            # 2) Detección secuencial en cámaras con detector
            for cam_id in DETECTED_CAM_IDS:
                frame = last_frames[cam_id]
                t0 = time.time()
                results = model(frame, size=IMG_SIZE)
                det = results.xyxy[0] if hasattr(results, "xyxy") and len(results.xyxy) else None
                cnt, boxes = count_and_boxes(det, vehicle_ids, MIN_BOX_AREA, frame.shape)
                last_counts[cam_id] = cnt
                last_frames[cam_id] = draw_boxes(frame, boxes, names, as_green=True)
                # opcional: muestra fps por cámara
                fps = 1.0 / max(time.time() - t0, 1e-6)
                cv2.putText(last_frames[cam_id], f"FPS ~ {fps:.1f}", (10, 52),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2, cv2.LINE_AA)

            # 3) Monitoreo (sin detector)
            for cam_id in MONITOR_CAM_IDS:
                frame = last_frames[cam_id]
                cv2.putText(frame, "MONITOREO", (10, 52),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
                last_frames[cam_id] = frame

            # 4) UI
            if SHOW:
                mosaic = mosaic_2x2(last_frames, titles={i: CAMS[i]["name"] for i in range(4)}, counts=last_counts)
                cv2.imshow("Multicam - Vehiculos (3 detectores + 1 monitor)", mosaic)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break

    # Cierre
    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
