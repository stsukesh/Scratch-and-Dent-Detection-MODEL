from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'D:\projects\AI\object detection YOLO\best.pt')
    try:
        while True:
            model.predict(source=0, show=True, save=False)
    except KeyboardInterrupt:
        print("Exited")