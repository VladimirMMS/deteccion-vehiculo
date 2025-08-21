import argparse
from pathlib import Path
import time

import cv2
import supervision as sv
from inference import get_model

# ---------- Helpers ----------
def is_camera_source(src: str) -> bool:
    # "0", "1", etc. => webcam index
    return src.isdigit()

def build_label_texts(detections: sv.Detections, classes) -> list[str]:
    labels = []
    for cls_id, conf in zip(detections.class_id, detections.confidence):
        name = classes.get(int(cls_id), f"id_{cls_id}")
        labels.append(f"{name} {conf:.2f}")
    return labels

# ---------- Image pipeline ----------
def run_on_image(model, image_path: str, out_path: str | None):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")

    # Inference
    results = model.infer(image)[0]
    detections = sv.Detections.from_inference(results)

    # Annotators
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    labels = build_label_texts(detections, results.classes)

    annotated = box_annotator.annotate(scene=image.copy(), detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

    # Mostrar
    cv2.imshow("Vehicles - Image", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if out_path:
        cv2.imwrite(out_path, annotated)
        print(f"✅ Imagen guardada en: {out_path}")

# ---------- Video / Webcam pipeline ----------
def run_on_stream(model, source: str, out_path: str | None):
    cap = cv2.VideoCapture(int(source)) if is_camera_source(source) else cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente: {source}")

    # Preparar writer si se pide salida de video
    writer = None
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    t0 = time.time()
    frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.infer(frame)[0]
        detections = sv.Detections.from_inference(results)
        labels = build_label_texts(detections, results.classes)

        annotated = box_annotator.annotate(scene=frame, detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        frames += 1
        if frames % 10 == 0:
            dt = time.time() - t0
            fps = frames / dt if dt > 0 else 0
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Vehicles - Stream", annotated)
        if writer is not None:
            writer.write(annotated)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    if writer:
        writer.release()
        print(f"✅ Video guardado en: {out_path}")
    cv2.destroyAllWindows()

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Detección de vehículos con Roboflow Inference + Supervision")
    ap.add_argument("--source", required=True,
                    help="Ruta a imagen/video o índice de webcam (e.g., 0)")
    ap.add_argument("--model-id", default="vehicles-q0x2v/1",
                    help="ID del modelo en Roboflow Inference (p.ej. 'vehicles-q0x2v/1')")
    ap.add_argument("--out", default=None,
                    help="Ruta de salida (imagen o video). Si omites, no guarda.")
    args = ap.parse_args()

    # Carga modelo (usa RF API key si el proyecto es privado)
    model = get_model(model_id=args.model_id)

    # Elegir pipeline según source
    if Path(args.source).exists() and not is_camera_source(args.source):
        # Si el path existe, distinguimos por extensión básica
        ext = Path(args.source).suffix.lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            run_on_image(model, args.source, args.out)
        else:
            run_on_stream(model, args.source, args.out)
    else:
        # Webcam u origen no-existente => intentar cámara
        run_on_stream(model, args.source, args.out)

if __name__ == "__main__":
    main()
