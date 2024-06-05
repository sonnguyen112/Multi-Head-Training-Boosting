from ultralytics import YOLOv10

# model = YOLOv10.from_pretrained('yolov10x.pt')
# or
# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
model = YOLOv10('yolov10x.pt')

model.predict()