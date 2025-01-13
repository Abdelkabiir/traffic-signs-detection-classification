# Traffic Sign Detection System

A real-time traffic sign detection and classification system using YOLOv4 and deep learning, capable of identifying and classifying 43 different types of traffic signs through your computer's webcam.

## Features

- Real-time traffic sign detection using YOLOv4
- Classification of 43 different traffic sign types
- Live webcam processing
- Color-coded detection boxes (Red for danger, Yellow for prohibitory, Blue for mandatory)
- Confidence score display
- Easy-to-use command line interface

## Repository Structure

```
traffic-sign-detection/
│
├── data/                      # Model files and weights
│   ├── yolov4_trained.weights
│   ├── yolov4_trained.cfg
│   └── classification.h5
│
├── src/                      # Source code
│   ├── __init__.py
│   ├── main.py               # Main application script
│   ├── detection.py          # YOLO detection utilities
│   └── classification.py     # Sign classification utilities
│
├── utils/                    # Utility functions
│   ├── __init__.py
│   └── visualization.py      # Visualization helpers
│
├── requirements.txt          # Project dependencies
├── README.md                 # Project documentation
└── LICENSE                   # License file
```

## Prerequisites

- Python 3.7+
- OpenCV
- TensorFlow 2.x
- Keras
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Abdelkabiir/traffic-signs-detection-classification.git
cd traffic-sign-detection
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Download and set up the model files:

The project requires three model files that should be placed in the `data/` directory:

a) Classification Model (`saved_model_v5.h5`):
   - Generated from the German Traffic Signs Recognition Dataset (GTSRB)
   - You can train your own model using the Kaggle notebook:
   - [German Traffic Signs Recognition Notebook](https://www.kaggle.com/code/abdelkabiirr/german-traffic-signs-recognition)

b) YOLO Detection Files (`yolov4_trained.weights` and `yolov4_trained.cfg`):
   - Available from the darknet repository
   - Download from: [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
   - Use the YOLOv4-tiny versions of the weights and configuration files

## Usage

Run the detection system using:

```bash
python src/main.py --weights="./data/yolov4_trained.weights" --cfg="./data/yolov4_trained.cfg" --model="./data/saved_model_v5.h5"
```

### Command Line Arguments

- `--weights`: Path to YOLO weights file
- `--cfg`: Path to YOLO configuration file
- `--model`: Path to classification model file (H5 format)

## Traffic Sign Classes

The system can detect and classify 43 different types of traffic signs, including:
- Speed limits (various speeds)
- No passing zones
- Priority signs
- Warning signs
- Mandatory signs
- And more...

## How It Works

1. **Detection Stage**: 
   - Uses YOLOv4 to detect traffic signs in the video feed
   - Provides bounding boxes around detected signs
   - Color codes the detections based on sign category

2. **Classification Stage**:
   - Crops detected signs from the frame
   - Resizes to 30x30 pixels
   - Feeds through a CNN classifier
   - Outputs specific sign classification

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License

## Acknowledgments

- YOLOv4 for object detection
- German Traffic Sign Recognition Benchmark (GTSRB) dataset