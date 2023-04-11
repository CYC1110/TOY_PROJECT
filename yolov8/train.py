from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
model.train(data='data.yaml', epochs=5, imgsz=640, save=True, name='my_yolov8')