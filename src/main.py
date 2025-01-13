import cv2
import argparse
import sys
from pathlib import Path

# Add the project root directory to Python path
root_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(root_dir)

from detection import SignDetector
from classification import SignClassifier
from utils.visualization import Visualizer


def main(weights_path: str, cfg_path: str, classification_model_path: str):
    """
    Main function to run the traffic sign detection and classification system.

    Args:
        weights_path: Path to YOLO weights file
        cfg_path: Path to YOLO configuration file
        classification_model_path: Path to classification model file
    """
    # Initialize components
    detector = SignDetector(weights_path, cfg_path)
    classifier = SignClassifier(classification_model_path)
    visualizer = Visualizer()

    # Initialize video capture
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Detect signs
        boxes, confidences, class_ids = detector.detect(frame)

        # Apply Non-Maximum Suppression
        indexes = detector.apply_nms(boxes, confidences, class_ids)

        # Process valid detections
        if len(indexes) > 0:
            indexes = indexes.flatten()

            for i in indexes:
                # Get detection info
                x, y, w, h = boxes[i]
                class_id = class_ids[i]
                confidence = confidences[i]

                # Get sign category
                category = classifier.get_sign_category(class_id)

                # Classify the detected sign
                if h > 0 and w > 0:
                    # Extract and classify the sign
                    sign_image = frame[y:y + h, x:x + w]
                    if sign_image.size > 0:
                        sign_class, _ = classifier.classify(sign_image)

                        # Visualize the detection
                        frame = visualizer.draw_detection(
                            frame,
                            boxes[i],
                            sign_class,
                            confidence,
                            class_id
                        )

        # Display the frame
        cv2.imshow("Traffic Sign Detection", frame)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Traffic Sign Detection and Classification')
    parser.add_argument('--weights', required=True, help='Path to YOLO weights file')
    parser.add_argument('--cfg', required=True, help='Path to YOLO cfg file')
    parser.add_argument('--model', required=True, help='Path to classification model file')

    args = parser.parse_args()
    main(args.weights, args.cfg, args.model)