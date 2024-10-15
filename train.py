from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="./data/data.yaml", epochs=200, imgsz=640, batch=16, amp=False)
model_path = "model.pt"
model.save(model_path)
print(f"Model saved to {model_path}")