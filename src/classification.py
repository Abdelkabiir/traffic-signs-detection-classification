from tensorflow.keras.models import load_model
import numpy as np
import cv2
from typing import Tuple, List


class SignClassifier:
    def __init__(self, model_path: str, image_size: Tuple[int, int] = (30, 30)):
        """
        Initialize the traffic sign classifier.

        Args:
            model_path: Path to the trained classification model
            image_size: Input image size expected by the model
        """
        self.model = load_model(model_path)
        self.image_size = image_size

        # Define sign categories and classes
        self.sign_names = [
            "prohibitory",
            "danger",
            "mandatory",
            "other",
        ]

        self.signs_classes = [
            "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
            "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
            "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
            "No passing", "No passing for vehicles over 3.5 metric tons",
            "Right-of-way at the next intersection", "Priority road", "Yield", "Stop",
            "No vehicles", "Vehicles over 3.5 metric tons prohibited", "No entry",
            "General caution", "Dangerous curve to the left", "Dangerous curve to the right",
            "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
            "Road work", "Traffic signals", "Pedestrians", "Children crossing",
            "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
            "End of all speed and passing limits", "Turn right ahead", "Turn left ahead",
            "Ahead only", "Go straight or right", "Go straight or left", "Keep right",
            "Keep left", "Roundabout mandatory", "End of no passing",
            "End of no passing by vehicles over 3.5 metric tons"
        ]

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for classification.

        Args:
            image: Input image in BGR format

        Returns:
            Preprocessed image ready for classification
        """
        resized = cv2.resize(image, self.image_size)
        return resized.reshape(-1, self.image_size[0], self.image_size[1], 3)

    def classify(self, image: np.ndarray) -> Tuple[str, int]:
        """
        Classify a traffic sign image.

        Args:
            image: Input image in BGR format

        Returns:
            Tuple of (sign class name, class index)
        """
        preprocessed = self.preprocess_image(image)
        prediction = np.argmax(self.model.predict(preprocessed))
        return self.signs_classes[prediction], prediction

    def get_sign_category(self, class_id: int) -> str:
        """
        Get the category name for a given class ID.

        Args:
            class_id: Class ID from detection

        Returns:
            Category name
        """
        if class_id < len(self.sign_names):
            return self.sign_names[class_id]
        return "other"