# 🔍 Scratch and Dent Detection Model

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLOv11](https://img.shields.io/badge/YOLOv11-Object%20Detection-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

An advanced computer vision solution for automated detection and localization of scratches and dents on product surfaces using YOLOv11. This project leverages state-of-the-art deep learning techniques to provide real-time, high-accuracy defect detection capabilities.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Technical Stack](#technical-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Model Details](#model-details)
- [Dataset Information](#dataset-information)
- [Performance Metrics](#performance-metrics)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

This project implements a robust real-time object detection system specifically designed to identify and locate scratches and dents on product surfaces. Built on the YOLOv11 architecture, it combines fast inference speeds with high accuracy, making it ideal for manufacturing quality control and inspection workflows.

**Use Cases:**
- ✅ Manufacturing quality assurance
- ✅ Product inspection automation
- ✅ Defect detection in assembly lines
- ✅ Real-time surface damage assessment
- ✅ Automated quality reporting

---

## ⭐ Key Features

### 🚀 Performance
- **Real-time Detection:** Process video streams at high frame rates with minimal latency
- **High Accuracy:** Fine-tuned YOLOv11 model with optimized training parameters
- **Multi-class Detection:** Distinguishes between different types of surface defects
- **GPU Acceleration:** Supports CUDA-enabled GPUs for faster inference

### 🛠️ Flexibility
- **Multiple Input Sources:** Support for webcams, video files, and image sequences
- **Easy Configuration:** YAML-based configuration for dataset and model parameters
- **Pre-trained Weights:** Readily available best-performing model weights
- **Modular Design:** Clean, well-organized codebase for easy customization

### 📊 Integration
- **Live Monitoring:** Check camera devices and configure real-time detection streams
- **Model Weights:** Pre-trained weights (best.pt) ready for immediate use
- **Training Logs:** Complete training history and performance metrics in `/runs` directory

---

## 🛠️ Technical Stack

| Component | Technology |
|-----------|-----------|
| **Deep Learning Framework** | PyTorch |
| **Model Architecture** | YOLOv11 (Nano) |
| **Computer Vision** | OpenCV |
| **Language** | Python 3.8+ |
| **GPU Support** | CUDA (Optional) |
| **Training** | Custom dataset with YOLO training pipeline |

---

## 📁 Project Structure

```
Scratch-and-Dent-Detection-MODEL/
├── dataset/                          # Custom training and validation dataset
│   ├── images/
│   │   ├── train/                   # Training images
│   │   └── val/                     # Validation images
│   └── labels/                       # YOLO format annotations (*.txt)
│
├── runs/                            # Training outputs and results
│   └── detect/
│       └── train9/                  # Latest training run
│           ├── weights/
│           │   ├── best.pt         # Best model weights
│           │   └── last.pt         # Last epoch weights
│           ├── results.csv         # Training metrics
│           └── confusion_matrix.png # Model performance visualization
│
├── train.py                         # Model training script
├── check.py                         # Real-time detection inference script
├── data.yaml                        # Dataset configuration file
├── best.pt                          # Pre-trained model weights
├── yolo11n.pt                       # YOLOv11 Nano base weights
├── Readme.md                        # Original documentation
├── README.md                        # This comprehensive guide
└── SukeshInternshipReport.pdf      # Detailed internship report

```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager
- Git
- NVIDIA GPU with CUDA support (optional, for faster training/inference)

### Step 1: Clone the Repository

```bash
git clone https://github.com/stsukesh/Scratch-and-Dent-Detection-MODEL.git
cd Scratch-and-Dent-Detection-MODEL
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n scratch-dent python=3.10
conda activate scratch-dent
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present, install manually:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python numpy pyyaml pillow
```

### Step 4: Verify Installation

```bash
python -c "from ultralytics import YOLO; print('✓ Installation successful!')"
```

---

## 🎯 Quick Start

### Run Real-Time Detection (Webcam)

```bash
python check.py
```

**Controls:**
- `ESC` or `Q` → Exit
- Press any key to pause/resume

### Train on Custom Dataset

```bash
python train.py
```

**Note:** Ensure `data.yaml` is properly configured before training.

### List Available Cameras

```bash
python ListofCamera.py
```

---

## 📖 Usage Guide

### 1️⃣ Real-Time Detection with check.py

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('best.pt')

# Run detection on webcam
results = model.predict(source=0, conf=0.5)

# Or on a video file
results = model.predict(source='video.mp4', conf=0.5)
```

**Parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `source` | Video source (0 for webcam, file path, or URL) | 0 |
| `conf` | Confidence threshold for predictions | 0.25 |
| `iou` | IoU threshold for NMS | 0.45 |
| `device` | Device to run on ('cpu' or 'cuda') | Auto-detect |

### 2️⃣ Model Training with train.py

```bash
# Basic training
python train.py

# With custom parameters
python train.py --epochs 100 --batch 16 --img 640
```

**Key Training Parameters:**
```python
results = model.train(
    data='data.yaml',           # Path to dataset config
    epochs=100,                 # Number of training epochs
    imgsz=640,                  # Image size
    batch=16,                   # Batch size
    device=0,                   # GPU device ID
    patience=20,                # Early stopping patience
    augment=True,               # Data augmentation
    conf=0.25,                  # Confidence threshold
    save=True                   # Save checkpoints
)
```

### 3️⃣ Configuration with data.yaml

```yaml
path: /path/to/dataset
train: images/train
val: images/val
nc: 2                    # Number of classes
names: ['Scratch', 'Dent']  # Class names
```

---

## 🧠 Model Details

### Architecture: YOLOv11 Nano

```
┌─────────────────────────────────────┐
│   Input: 640x640 RGB Image         │
└────────────┬────────────────────────┘
             │
      ┌──────▼──────┐
      │  Backbone   │  Feature extraction
      └──────┬──────┘
             │
      ┌──────▼──────┐
      │    Neck     │  Feature fusion
      └──────┬──────┘
             │
      ┌──────▼──────┐
      │    Head     │  Detection & classification
      └──────┬──────┘
             │
┌────────────▼──────────────────┐
│  Bounding Boxes + Confidence  │
│  (Scratch/Dent Classification)│
└───────────────────────────────┘
```

### Model Performance
- **Parameters:** ~2.7M (Nano variant)
- **Inference Speed:** ~50ms per image (RTX 3060 Ti)
- **Model Size:** ~6.2 MB
- **Supported Input:** 640x640 pixels (configurable)

---

## 📊 Dataset Information

### Dataset Statistics

```
Total Images: [To be filled from your dataset]
├── Training Images:   [X images]
├── Validation Images: [Y images]
└── Test Images:       [Z images]

Classes: 2
├── 1. Scratch
└── 2. Dent

Format: YOLO (txt format with normalized coordinates)
```

### Dataset Structure

```
dataset/
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   └── val/
│       ├── img_101.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img_001.txt  # Format: <class> <x_center> <y_center> <width> <height>
    │   └── ...
    └── val/
        └── ...
```

### YOLO Label Format
Each `.txt` file contains one line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```
Where coordinates are normalized to [0, 1].

---

## 📈 Performance Metrics

### Training Results (train9)

The model has been trained and evaluated on the custom scratch and dent dataset. Key metrics include:

**Evaluation Metrics:**
- **mAP50:** Mean Average Precision at IoU=0.50
- **mAP50-95:** Mean Average Precision across IoU thresholds (0.50-0.95)
- **Precision:** True positives / (True positives + False positives)
- **Recall:** True positives / (True positives + False negatives)

**View detailed results:**
```bash
# Training metrics
ls runs/detect/train9/

# Plots available
├── results.csv              # Metric history
├── confusion_matrix.png     # Class confusion matrix
├── F1_curve.png            # F1-score curve
├── P_curve.png             # Precision curve
├── R_curve.png             # Recall curve
└── PR_curve.png            # Precision-Recall curve
```

---

## ⚙️ Configuration

### data.yaml Reference

```yaml
# Dataset paths (relative to current directory)
path: dataset
train: dataset/images/train
val: dataset/images/val
test: dataset/images/test  # Optional

# Number of classes
nc: 2

# Class names (order matters!)
names:
  0: Scratch
  1: Dent
```

### train.py Parameters

```python
# Model initialization
model = YOLO('yolo11n.pt')

# Training configuration
results = model.train(
    # Data configuration
    data='data.yaml',
    
    # Training parameters
    epochs=100,
    batch=16,
    imgsz=640,
    
    # Optimization
    optimizer='SGD',          # SGD, Adam, AdamW, RMSProp
    lr0=0.01,                # Initial learning rate
    lrf=0.01,                # Final learning rate
    momentum=0.937,
    weight_decay=0.0005,
    
    # Augmentation
    augment=True,
    flipud=0.5,              # Vertical flip probability
    fliplr=0.5,              # Horizontal flip probability
    mosaic=1.0,              # Mosaic augmentation
    
    # Device & Logging
    device=0,                # GPU device ID
    workers=8,               # DataLoader workers
    save=True,               # Save checkpoints
    patience=20,             # Early stopping patience
    verbose=True,
    
    # Callbacks
    project='runs/detect',
    name='train'
)
```

### check.py Parameters

```python
# Inference configuration
results = model.predict(
    source='0',              # 0 for webcam, file path, or URL
    conf=0.5,                # Confidence threshold
    iou=0.45,                # IoU threshold for NMS
    imgsz=640,               # Inference image size
    device=0,                # GPU device ID
    half=False,              # FP16 inference (faster)
    max_det=300,             # Max detections per image
    classes=None,            # Filter by class (None = all)
    verbose=True,
    save=False,              # Save predictions
    save_txt=False,          # Save predictions as txt
    save_conf=False          # Save confidence scores
)
```

---

## 🔧 Troubleshooting

### Common Issues & Solutions

#### Issue 1: CUDA Out of Memory

**Error:** `CUDA out of memory`

**Solution:**
```python
# Reduce batch size
python train.py --batch 8

# Or reduce image size
python train.py --img 512

# Or use CPU
model = YOLO('best.pt')
results = model.predict(source=0, device='cpu')
```

#### Issue 2: No Module Named 'ultralytics'

**Error:** `ModuleNotFoundError: No module named 'ultralytics'`

**Solution:**
```bash
pip install ultralytics --upgrade
```

#### Issue 3: Webcam Not Detected

**Check available cameras:**
```bash
python ListofCamera.py
```

**Try specific camera index:**
```python
results = model.predict(source=1)  # Try camera index 1, 2, etc.
```

#### Issue 4: Low Detection Performance

**Solutions:**
1. Increase confidence threshold: `conf=0.3` (lower = more detections)
2. Retrain with more data
3. Adjust IoU threshold: `iou=0.5`
4. Check image quality and lighting conditions

---

## 🚀 Future Improvements

- [ ] Integrate with manufacturing IoT platforms
- [ ] Add multi-camera support for conveyor belt systems
- [ ] Implement database logging for defect tracking
- [ ] Create web-based dashboard for monitoring
- [ ] Support for YOLOv11m/l for higher accuracy
- [ ] Real-time alert system for critical defects
- [ ] Export to ONNX/TensorRT for edge deployment
- [ ] Mobile app for on-site inspection
- [ ] Advanced severity classification (minor/moderate/severe)
- [ ] Integration with ERP systems for quality reporting

---

## 📝 Additional Notes

### Model Selection

The project uses **YOLOv11 Nano** for optimal speed-accuracy trade-off:

| Variant | Speed | Accuracy | Use Case |
|---------|-------|----------|----------|
| **Nano** | ⚡⚡⚡ | ⭐⭐⭐ | Real-time (Current) |
| Small | ⚡⚡ | ⭐⭐⭐⭐ | Better accuracy needed |
| Medium | ⚡ | ⭐⭐⭐⭐⭐ | High accuracy required |

### Inference Optimization

For production deployment:

```python
# Export to ONNX (faster inference)
model = YOLO('best.pt')
model.export(format='onnx')

# Or to TensorRT (best GPU performance)
model.export(format='engine')

# Then load and run
from ultralytics import YOLO
onnx_model = YOLO('best.onnx')
results = onnx_model.predict(source=0)
```

---

## 📚 References & Resources

- **YOLOv11 Documentation:** [Ultralytics YOLOv11](https://docs.ultralytics.com/)
- **PyTorch Official:** [pytorch.org](https://pytorch.org/)
- **OpenCV Documentation:** [docs.opencv.org](https://docs.opencv.org/)
- **YOLO Paper:** [You Only Look Once](https://pjreddie.com/darknet/yolo/)

---

## 📄 Citation

If you use this project in your research or work, please cite:

```bibtex
@misc{scratch_dent_detection,
  author = {Sukesh},
  title = {Scratch and Dent Detection Model using YOLOv11},
  year = {2024},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/stsukesh/Scratch-and-Dent-Detection-MODEL}}
}
```

---

## 🙏 Acknowledgments

This project was developed as part of an internship at **FLEX Pvt Ltd**. Special thanks to:

- 🎓 **FLEX Pvt Ltd** - For providing the opportunity and resources
- 👨‍💼 **Supervisors and Mentors** - For guidance and feedback
- 👥 **Team Members** - For collaboration and support
- 🤖 **Ultralytics** - For the excellent YOLOv11 framework
- 📚 **Open Source Community** - For tools and libraries

---

## 📧 Contact & Support

- **Author:** Sukesh
- **GitHub:** [@stsukesh](https://github.com/stsukesh)
- **Project:** [Scratch-and-Dent-Detection-MODEL](https://github.com/stsukesh/Scratch-and-Dent-Detection-MODEL)

For issues, questions, or suggestions, please:
1. Check the [Troubleshooting](#troubleshooting) section
2. Open an issue on GitHub
3. Refer to the internship report (SukeshInternshipReport.pdf)

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

```
MIT License

Copyright (c) 2024 Sukesh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

**⭐ If you find this project helpful, please consider giving it a star!**

---

<div align="center">

Made with ❤️ by Sukesh | [GitHub](https://github.com/stsukesh) | [Portfolio](#)

</div>
