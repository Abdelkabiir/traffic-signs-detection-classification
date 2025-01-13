import cv2
import numpy as np
from typing import List, Tuple, Dict


class SignDetector:
    def __init__(self, weights_path: str, cfg_path: str, confidence_threshold: float = 0.5):
        """
        Initialize the traffic sign detector with YOLO model.

        Args:
            weights_path: Path to YOLO weights file
            cfg_path: Path to YOLO configuration file
            confidence_threshold: Minimum confidence threshold for detections
        """
        self.net = cv2.dnn.readNet(weights_path, cfg_path)
        self.confidence_threshold = confidence_threshold

        # Get output layer names
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for YOLO model.

        Args:
            image: Input image in BGR format

        Returns:
            Preprocessed image blob
        """
        return cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    def detect(self, image: np.ndarray) -> Tuple[List, List, List]:
        """
        Detect traffic signs in the image.

        Args:
            image: Input image in BGR format

        Returns:
            Tuple containing lists of boxes, confidences, and class IDs
        """
        height, width = image.shape[:2]

        # Preprocess image and get detections
        blob = self.preprocess_image(image)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        # Initialize lists for detection results
        boxes = []
        confidences = []
        class_ids = []

        # Process detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.confidence_threshold:
                    # Calculate bounding box coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        return boxes, confidences, class_ids

    def apply_nms(self, boxes: List, confidences: List, class_ids: List) -> List[int]:
        """
        Apply Non-Maximum Suppression to remove overlapping detections.

        Args:
            boxes: List of bounding boxes
            confidences: List of confidence scores
            class_ids: List of class IDs

        Returns:
            List of indices of kept boxes
        """
        return cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.4)