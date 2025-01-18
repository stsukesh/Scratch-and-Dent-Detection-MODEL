from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('D:\projects\AI\object detection YOLO\yolo11n.pt')
    model.train(data="D:\projects\AI\object detection YOLO\data.yaml", imgsz=640, batch=8, epochs=100, workers=1, device=0)