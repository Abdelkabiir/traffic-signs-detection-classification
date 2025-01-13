import cv2
import numpy as np
from typing import Dict, Tuple, List


class Visualizer:
    def __init__(self):
        """Initialize the visualizer with color schemes for different sign categories."""
        self.class_colors = {
            0: (0, 0, 255),  # Red for danger
            1: (0, 255, 255),  # Yellow for prohibitory
            2: (255, 0, 0),  # Blue for mandatory
        }
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_detection(self,
                       image: np.ndarray,
                       box: List[int],
                       label: str,
                       confidence: float,
                       class_id: int) -> np.ndarray:
        """
        Draw a single detection on the image.

        Args:
            image: Input image
            box: Bounding box coordinates [x, y, w, h]
            label: Label text to display
            confidence: Detection confidence
            class_id: Class ID for color selection

        Returns:
            Image with detection visualization
        """
        x, y, w, h = box
        color = self.class_colors.get(class_id, (0, 165, 255))

        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        # Draw label with confidence
        label_with_conf = f"{label} ({confidence:.2f}%)"
        cv2.putText(image, label_with_conf, (x, y - 10), self.font, 0.5, color, 2)

        return image

    def draw_detections(self,
                        image: np.ndarray,
                        boxes: List[List[int]],
                        labels: List[str],
                        confidences: List[float],
                        class_ids: List[int]) -> np.ndarray:
        """
        Draw multiple detections on the image.

        Args:
            image: Input image
            boxes: List of bounding box coordinates
            labels: List of label texts
            confidences: List of confidence scores
            class_ids: List of class IDs

        Returns:
            Image with all detections visualized
        """
        for box, label, confidence, class_id in zip(boxes, labels, confidences, class_ids):
            image = self.draw_detection(image, box, label, confidence * 100, class_id)
        return image