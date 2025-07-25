# PrintGuard: Real-Time 3D Printing Defect Detection with Deep Learning

## Overview

PrintGuard is a toolkit for monitoring and detecting 3D printing errors in real time using a ResNet50-based deep learning model. It classifies each video frame as one of three classes: `no_error`, `extrusion_error`, or `layershift_error`. This enables users to monitor print quality and minimize filament waste due to unnoticed defects.

**Key Features:**
- Real-time defect detection with camera or webcam
- Three defect classes supported
- Robust to color, angle, and mild lighting changes
- Eliminates color bias by using grayscale preprocessing
- Modular, easy to extend or customize

## Table of Contents

- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Setup Instructions](#setup-instructions)
- [How to Train the Model](#how-to-train-the-model)
- [How to Run Real-Time Detection](#how-to-run-real-time-detection)
- [Results and Sample Outputs](#results-and-sample-outputs)
- [Project Highlights](#project-highlights)
- [Notes and Tips](#notes-and-tips)

## Project Structure

```
project_root/
├── 3d_printing/
│   ├── 3d1.py                # Training script
│   ├── p4.py                 # Real-time inference script
│   ├── class_indices.pkl     # Label mapping for classes
│   ├── resnet50_3dprint_defect4.h5  # Trained ResNet50 model weights
│   ├── training_curves.jpg   # Training loss/accuracy curves
│   └── Temperature Optimisation/
│       └── ...               # (Additional scripts or resources)
├── .gitattributes
└── base/                     # **You must add this folder with dataset inside!**
    └── full_res/
        ├── extrusion_error/
        │   ├── camera1/
        │   ├── camera2/
        │   └── camera3/
        ├── layershift_error/
        │   ├── camera1/
        │   ├── camera2/
        │   └── camera3/
        └── no_error/
            ├── camera1/
            ├── camera2/
            └── camera3/
```

**Note:** The `base/` directory and all dataset images must be downloaded separately from the Zenodo dataset source and placed as shown.

## Dataset

- Multi-camera labeled dataset (`extrusion_error`, `layershift_error`, `no_error`)
- Each class and camera has its own folder for robust training
- Download the dataset from the Zenodo link below and extract it under `base/full_res/` as shown above.

## Requirements

- **Python**: ≥3.7 (3.8/3.9 recommended)
- **Packages**:
  - tensorflow
  - keras
  - numpy
  - opencv-python
  - matplotlib
  - pickle

**Install dependencies:**
```bash
pip install tensorflow keras numpy opencv-python matplotlib
```

## Setup Instructions

1. **Clone the repository**
    ```bash
    git clone https://github.com/avadacodavra/3D_Printing-Defect_Optimisation.git
    cd 3D_Printing-Defect_Optimisation
    ```
2. **Download the dataset**
    - Get the [3D Printer Defect Detection Dataset (Zenodo)](https://zenodo.org/records/14712897) and extract into `base/full_res/` as per structure above.
3. **Install required packages**
    ```bash
    pip install tensorflow keras numpy opencv-python matplotlib
    ```
4. **Verify directory structure**
    - Ensure `3d1.py`, `p4.py`, and the dataset folders are present as described.

## How to Train the Model

1. **Edit dataset path**  
   - If your dataset path differs from `base/full_res/`, update the path in `3d1.py`.

2. **Train**
    ```bash
    python 3d1.py
    ```
   - Converts all images to grayscale
   - Trains ResNet50 model on dataset
   - Saves outputs:
     - `resnet50_3dprint_defect4.h5` (weights)
     - `class_indices.pkl` (class mapping)
     - `training_curves.jpg` (training/validation curves)

## How to Run Real-Time Detection

1. **Connect your webcam or use Iriun Webcam app on your phone**  
2. **Edit camera index if needed**  
   - In `p4.py`, set the correct webcam index:  
     ```python
     cap = cv2.VideoCapture(1)  # Try 0, 1, 2...
     ```
3. **Run inference**
    ```bash
    python p4.py
    ```
   - Live window shows:
     - Predicted class and confidence over ROI
     - Green label = confident prediction (defect or no error)
     - Red label = low confidence ("No printer detected")
   - Stop anytime with the `q` key.

## Results and Sample Outputs

- Typical validation accuracy: **>92%**
- Robust inference in real-time even on CPUs
- Model is resilient to print color, camera angle, and lighting
- Visual alerts overlayed directly on the camera ROI

## Project Highlights

- **Color agnostic**: All data and frames are grayscale—prevents color bias.
- **Modular**: Easily update class names or extend with more defect categories.
- **Efficient**: Real-time on CPU and small form factor PCs.
- **Strong generalization**: Data augmentation in training reduces overfitting.

## Notes and Tips

- For best results, mount the camera so it clearly views the print area (side or slightly above).
- Ensure good, even lighting without occlusions.
- Update folder structure and retrain if adding new defect categories.
- For even faster inference, MobileNetV2 can be used instead of ResNet50 (requires modification).
- If the webcam is dark or unfocused, adjust its positioning or use camera controls.
