from ultralytics import YOLO,YOLOWorld

# Load a COCO-pretrained YOLO11n model
#model = YOLO("yolo11n.pt")
model = YOLOWorld("yolov8s-worldv2.pt")

results = model.train(data="./EV_components/dataset.yaml",epochs=100, imgsz=640, batch=16, device="cuda:0", name="yolo11n_finetuned")
print(results)