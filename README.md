<div align="center">

# recyclable-detector

Real-time recyclable material detection using YOLOv8

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/github/license/horse-3903/recyclable-detector?style=flat-square)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/horse-3903/recyclable-detector?style=flat-square)](../../commits)
[![Stars](https://img.shields.io/github/stars/horse-3903/recyclable-detector?style=flat-square)](../../stargazers)

</div>

---

## Overview

`recyclable-detector` is a YOLOv8-based object detection system that classifies materials in real time through a webcam feed or on static images. Trained on the Roboflow [Recyclables and Garbage Detection](https://universe.roboflow.com/recycling-detector/recyclables-and-garbage-detection/dataset/4) dataset, it identifies seven material categories to help automate recyclability decisions at the point of disposal.

## Features

- **7-class material detection**: cardboard, glass, metal, paper, plastic, trash, and mixed materials
- **Real-time webcam inference**: live bounding boxes with confidence scores at 50% threshold
- **Static image/video prediction**: run inference on any file path
- **Model evaluation**: validate on the held-out test set for mAP metrics

## Tech Stack

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logo=ultralytics&logoColor=black)](https://docs.ultralytics.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)

## Getting Started

### Prerequisites

- Python 3.8+
- Webcam (for live detection)

### Installation

```bash
git clone https://github.com/horse-3903/recyclable-detector.git
cd recyclable-detector
pip install ultralytics opencv-python
```

### Real-Time Detection (Webcam)

```bash
python src/video_predict.py
```

Press `q` to quit the live feed window.

```python
import cv2
from ultralytics import YOLO

model = YOLO("./src/model.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()
    cv2.imshow("Recyclable Detector", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

### Static Image Prediction

```python
from ultralytics import YOLO

model = YOLO("./src/model.pt")
results = model.predict(source="path/to/image.jpg", conf=0.5)
results[0].show()
```

### Model Training

```python
from ultralytics import YOLO

model = YOLO("yolov8n.yaml")
model.train(data="path/to/data.yaml", epochs=100)
```

Download the dataset from Roboflow and update the paths in `data.yaml` before training.

### Model Evaluation

```python
from ultralytics import YOLO

model = YOLO("./src/model.pt")
metrics = model.val(data="path/to/data.yaml")
print(metrics.box.map)   # mAP50-95
```

## Class Labels

| ID | Label | Description |
|----|-------|-------------|
| 0 | `cardboard` | Cardboard boxes and packaging |
| 1 | `glass` | Glass bottles and jars |
| 2 | `materials` | Mixed/other recyclable materials |
| 3 | `metal` | Cans and metal objects |
| 4 | `paper` | Paper and newspapers |
| 5 | `plastic` | Plastic bottles and containers |
| 6 | `trash` | Non-recyclable waste |

## Dataset

- **Source:** [Roboflow: Recyclables and Garbage Detection v4](https://universe.roboflow.com/recycling-detector/recyclables-and-garbage-detection/dataset/4)
- **License:** MIT

## License

MIT: see [LICENSE](LICENSE)
