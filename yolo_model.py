from ultralytics import YOLO


if __name__ == "__main__":
    # Load the custom architecture YAML, then load pretrained weights
    model = YOLO("model_settings.yaml").load("yolo26s.pt") 

    # Train using your custom configuration
    model.train(data="class_paths.yaml", epochs=50)