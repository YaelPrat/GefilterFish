# Gefilter Fish Project

## Description

This project uses OpenCV and dlib to detect faces and overlay a carrot image on the forehead of detected faces in a video stream. The project utilizes the `shape_predictor_68_face_landmarks.dat` model to identify facial landmarks and accurately position the carrot image.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/YaelPrat/GefilterFish.git
    cd GefilterFish
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the script**:
    ```bash
    python3 main.py
    ```

2. **Interact with the application**:
    - The script will open your webcam and start detecting faces.
    - A carrot image will be overlaid on the forehead of detected faces.
    - Press `ESC` to exit the application.

## Requirements

- Python 3.12
- OpenCV
- numpy
- dlib

You can install all the required packages using the provided `requirements.txt` file.

