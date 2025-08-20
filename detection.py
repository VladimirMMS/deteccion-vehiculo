# -*- coding: utf-8 -*-
import pathlib
pathlib.PosixPath = pathlib.WindowsPath  # por si el .pt viene de Linux/Mac

import cv2
import torch
import time
import numpy as np
from pathlib import Path

ROOT = Path(r"C:/Users/vladi/OneDrive/Desktop/detection-objects")
WEIGHTS = ROOT / "best.pt"  # ajusta si está en otra carpeta
CAM_INDEXES = [1]
BACKENDS    = [cv2.CAP_MSMF, cv2.CAP_DSHOW]

IMG_SIZE = 640
CONF_TH  = 0.25
IOU_TH   = 0.45

def open_camera(indexes, backends):
    for backend in backends:
        for idx in indexes:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    print(f"✅ Cámara abierta (index={idx}, backend={backend})")
                    return cap
                cap.release()
            print(f"❌ No abre: index={idx}, backend={backend}")
    raise RuntimeError("No se pudo abrir la cámara. Cierra Teams/Zoom/OBS/Discord y revisa permisos de cámara.")

def normalize_names(names_obj):
    if isinstance(names_obj, dict):
        return {int(k): str(v) for k, v in names_obj.items()}
    if isinstance(names_obj, (list, tuple)):
        return {i: str(n) for i, n in enumerate(names_obj)}
    return {}

def main():
    cv2.setUseOptimized(True)

    # Carga modelo
    model = torch.hub.load(
        'ultralytics/yolov5',
        'custom',
        path=str(WEIGHTS),
        source='github',
        trust_repo=True,
        force_reload=False
    )
    model.conf = CONF_TH
    model.iou  = IOU_TH
    names = normalize_names(getattr(model, "names", {}))

    # Abre cámara
    cap = open_camera(CAM_INDEXES, BACKENDS)

    # Config cámara
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    _ok, frame = cap.read()
    if _ok:
        with torch.inference_mode():
            _ = model(frame, size=IMG_SIZE)

    t_avg = None
    with torch.inference_mode():
        while True:
            ok, frame = cap.read()
            if not ok:
                print("⚠️ No se pudo leer frame.")
                break

            t0 = time.time()
            results = model(frame, size=IMG_SIZE)

            # >>>> FIX CLAVE: copia + contiguo + aseguramos uint8 y writable
            annotated = results.render()[0].copy()
            annotated = np.ascontiguousarray(annotated)
            if annotated.dtype != np.uint8:
                annotated = annotated.astype(np.uint8, copy=False)
            # <<<<

            # Conteo por clase con guards
            counts_txt = []
            det = results.xyxy[0] if hasattr(results, "xyxy") and len(results.xyxy) else None
            if det is not None and len(det):
                cls_ids = det[:, 5].detach().cpu().numpy().astype(int)
                for c in np.unique(cls_ids):
                    name = names.get(int(c), str(int(c)))
                    counts_txt.append(f"{name}: {(cls_ids == c).sum()}")

            # FPS
            dt = time.time() - t0
            fps = 1.0 / max(dt, 1e-6)
            t_avg = dt if t_avg is None else (0.9 * t_avg + 0.1 * dt)
            fps_avg = 1.0 / max(t_avg, 1e-6)

            # Overlays (ya no fallará por readonly)
            y = 28
            cv2.putText(annotated, f"FPS: {fps:.1f} (avg {fps_avg:.1f})", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
            y += 24
            for line in counts_txt:
                cv2.putText(annotated, line, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
                y += 22

            cv2.imshow("YOLOv5 - Mi modelo", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:   # ESC
                break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Fin")

if __name__ == "__main__":
    main()
