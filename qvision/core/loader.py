from ultralytics import YOLO

# Load the original model
model = YOLO('yolov8n.pt')

# Export the model to FP16
model.export(format='engine', half=True)