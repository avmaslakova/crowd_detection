import cv2
import torch
from ultralytics import YOLO


class PersonDetector:

    def __init__(self, model="yolov8n.pt", device="cpu"):
      self.device = device
      self.model = YOLO(model)

    def process_video(self, input_path, output_path):
      cap = cv2.VideoCapture(input_path)
      if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {input_path}")

      fps = cap.get(cv2.CAP_PROP_FPS) or 25
      w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
      h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
      out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

      while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = self.model.predict(frame, device=self.device, imgsz=640, conf=0.3)
        boxes = results[0].boxes

        for box in boxes:
            cls = int(box.cls[0])
            if cls != 0:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"person {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

      cap.release()
      out.release()

