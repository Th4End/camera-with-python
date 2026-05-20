# Camera with Python

A Python project using OpenCV and MediaPipe for real-time hand and face detection via webcam.

## Features

- **Hand Detection**: Detection and tracking of up to 4 hands simultaneously
- **Face Detection**: Face detection with bounding boxes
- **Gesture Recognition**: Real-time gesture recognition using LSF (Langue des Signes Française)
- **Real-time Interface**: Live display via webcam

## Project Files

- `cam.py`: Complete version with hand AND face detection
- `camera.py`: Simplified version with hand detection only
- `signe.py`: Gesture recognition using MediaPipe Gesture Recognizer
- `predict.py`: Sign language prediction using trained ML model
- `hand_detection.py`: Data collection tool for capturing hand landmarks
- `train.py`: Train the LSF classifier model
- `download_models.py`: Download required MediaPipe models

## Installation

### Prerequisites
- Python 3.7 or higher
- Connected webcam
- Internet connection (for downloading MediaPipe models)

### Quick Start (All Platforms)

1. **Clone or download this project**

2. **Run the installation script:**
   
   **Windows:**
   ```bash
   install.bat
   ```
   
   **Linux/macOS:**
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

   This will:
   - Create a Python virtual environment
   - Install all required dependencies
   - Download MediaPipe model files (hand_landmarker.task, gesture_recognizer.task)

### Manual Setup

#### Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

#### Download MediaPipe Models

```bash
python download_models.py
```

This downloads:
- `hand_landmarker.task` - Hand landmark detection model
- `gesture_recognizer.task` - Gesture recognition model

## Usage

### Complete Version (hands + faces)
```bash
python cam.py
```

### Simplified Version (hands only)
```bash
python camera.py
```

### Gesture Recognition (LSF)
```bash
python signe.py
```

### Sign Language Prediction
Requires a trained model (`lsf_model.pkl`). Train it first with:
```bash
python hand_detection.py
python train.py
python predict.py
```

## Workflow: Training Custom Sign Language Model

1. **Collect hand landmark data:**
   ```bash
   python hand_detection.py
   ```
   - Press 's' to save hand positions
   - Press 'a' to quit
   - Data saved to `data/` folder

2. **Train the model:**
   ```bash
   python train.py
   ```
   - Creates `lsf_model.pkl`
   - Displays training accuracy

3. **Use the trained model:**
   ```bash
   python predict.py
   ```

## Controls

- **Press 'a'** to quit the application
- **Press 's'** (in hand_detection.py) to capture hand landmarks
- Camera window displays in real-time

## Main Dependencies

- **OpenCV**: Image processing and video capture
- **MediaPipe**: Hand, face, and gesture detection
- **NumPy**: Numerical computations
- **scikit-learn**: Machine learning models

## Technical Notes

- Default resolution: 640x480 pixels
- Hand detection confidence: 0.7
- Hand tracking confidence: 0.4
- Support for up to 4 hands simultaneously
- ML Model: Random Forest Classifier with 200 estimators

## Troubleshooting

- **"No module named mediapipe"**: Run `pip install -r requirements.txt`
- **"hand_landmarker.task not found"**: Run `python download_models.py`
- **Camera not working**: Ensure webcam is connected and not in use by another app
- **Slow performance**: Reduce video resolution or decrease detection confidence