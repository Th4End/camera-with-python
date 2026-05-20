#!/bin/bash

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Downloading MediaPipe models..."
python download_models.py

echo "Installation complete!"
echo ""
echo "To start the application, run:"
echo "  python cam.py"
echo "  python camera.py"
echo "  python signe.py"
echo "  python predict.py"