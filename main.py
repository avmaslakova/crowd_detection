import sys
from pathlib import Path
import torch

from detector import PersonDetector


def main():
  input_path = Path("crowd.mp4")
  output_path = Path("output.mp4")
  model_path = "yolov8n.pt"

  if not input_path.exists():
    print(f"Входной файл не найден: {input_path}")
    sys.exit(1)

  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Используется устройство: {device}")

  detector = PersonDetector(model=model_path, device=device)

  detector.process_video(str(input_path), str(output_path))

  print(f"Видео сохранено: {output_path}")


if __name__ == "__main__":
  main()
