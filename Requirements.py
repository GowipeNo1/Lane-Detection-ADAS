from ultralytics import YOLO

def train_model():
    # Load the model
    model = YOLO("yolov8n.pt") 
    
    # Run the training
    # Added workers=0 or a small number to further prevent Windows threading issues
    model.train(
        data=r"F:\Desktop\Lane detection project\datasets\data.yaml", 
        epochs=100, 
        imgsz=640, 
        device=0,
        workers=4  # You can set this to 0 if the error persists
    )

if __name__ == '__main__':
    train_model()