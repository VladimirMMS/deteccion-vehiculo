import cv2

url = "rtsp://10.0.0.167:8554/cam1"  # o cam1, cam2, etc.
cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Stream desde Raspberry", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
