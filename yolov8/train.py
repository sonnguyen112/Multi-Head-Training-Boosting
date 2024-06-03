from ultralytics import YOLOCustom

# Create a new YOLO model from scratch
# model = YOLO("yolov8n.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLOCustom("yolov8s.yaml").load("yolov8s.pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data="../datasets/datasets_yolo/mix_det/data.yaml", epochs=80, batch=20, plots=True)