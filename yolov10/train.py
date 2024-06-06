from ultralytics import YOLOv10, YOLOv10Custom

model = YOLOv10Custom('yolov10s-custom.yaml').load('yolov10s.pt')
# If you want to finetune the model with pretrained weights, you could load the 
# pretrained weights like below
# model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
# or
# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
# model = YOLOv10('yolov10{n/s/m/b/l/x}.pt')

model.train(data='../datasets/datasets_yolo/debug_dataset/data.yaml', epochs=1, batch=1, imgsz=640)