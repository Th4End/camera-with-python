# Camera with Python

A Python project using OpenCV and MediaPipe for real-time hand and face detection via webcam.

## Features

- **Hand Detection**: Detection and tracking of up to 4 hands simultaneously
- **Face Detection**: Face detection with bounding boxes
- **Real-time Interface**: Live display via webcam

## Project Files

- `cam.py`: Complete version with hand AND face detection
- `camera.py`: Simplified version with hand detection only

## Installation

### Prerequisites
- Python 3.7 or higher
- Connected webcam

### Virtual Environment Setup

#### Windows
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Deactivate (when done)
deactivate
```

#### Linux/macOS
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Deactivate (when done)
deactivate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Complete Version (hands + faces)
```bash
python cam.py
```

### Simplified Version (hands only)
```bash
python camera.py
```

## Controls

- **Press 'a'** to quit the application
- Camera window displays in real-time

## Main Dependencies

- **OpenCV**: Image processing and video capture
- **MediaPipe**: Hand and face detection
- **NumPy**: Numerical computations

## Technical Notes

- Default resolution: 640x480 pixels
- Minimum detection confidence: 0.7
- Minimum tracking confidence: 0.4
- Support for up to 4 hands simultaneously