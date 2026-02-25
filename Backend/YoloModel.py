from ultralytics import YOLO
from utils.settings import CLASS_PATH, MODEL_SETTINGS, MODEL_NAME


if __name__ == "__main__":
    # Load the custom architecture YAML, then load pretrained weights
    model = YOLO(MODEL_SETTINGS).load(MODEL_NAME) 

    # Train using your custom configuration
    model.train(data=CLASS_PATH, epochs=50)