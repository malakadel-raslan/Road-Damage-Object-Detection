# Road Damage Object Detection

> Trained and compared three YOLO versions (v8, v9, v11) to detect road surface damage (potholes, cracks, and related defects) from images, fine-tuned the top-performing model, and deployed it as a live app.

## 📋 Overview
This project benchmarks three generations of YOLO object detectors on an 8-class road damage dataset from Roboflow, then fine-tunes the best-performing model and deploys it through an interactive Gradio interface for real-time road damage detection from uploaded images.

## ✨ Key Features
- Comparison of YOLOv8, YOLOv9, and YOLOv11 on the same dataset
- 8-class road damage detection (potholes, cracks, and related defects)
- Fine-tuning with the AdamW optimizer
- Interactive Gradio deployment for live inference

## 🛠️ Tech Stack
- Python, Ultralytics YOLO (v8/v9/v11)
- Roboflow (dataset)
- Gradio (deployment)

## 🚀 Getting Started
```bash
git clone https://github.com/malakadel-raslan/Road-Damage-Object-Detection.git
cd Road-Damage-Object-Detection
pip install -r requirements.txt
python app.py
```

## 📁 Project Structure
```
├── notebooks/         # Training & model comparison notebooks
├── weights/            # Fine-tuned model weights
├── app.py               # Gradio deployment
└── requirements.txt
```

## 📊 Results
YOLOv8, YOLOv9, and YOLOv11 were trained and compared on the 8-class road damage dataset; the top-performing model was fine-tuned further with AdamW before deployment. See the notebooks for detailed mAP/precision/recall metrics.

## 👤 Author
**Malak Adel Raslan**
[LinkedIn](https://linkedin.com/in/malak-raslan-34152628a) · lokaadel010050@gmail.com
