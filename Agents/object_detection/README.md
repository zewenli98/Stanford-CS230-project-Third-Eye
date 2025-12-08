# SUN RGB-D Object Detection Pipeline

A comprehensive PyTorch-based object detection training pipeline using YOLOv8 for the SUN RGB-D dataset, designed for blind navigation assistance systems.

## 🎯 Overview

This project provides an end-to-end pipeline for training, evaluating, and deploying object detection models on the SUN RGB-D indoor scene dataset. The pipeline is optimized for indoor navigation scenarios with specialized data augmentation and evaluation metrics.

## 📁 Project Structure

```
Agents/
├── data_prep/              # Data processing scripts
│   ├── parse_sunrgbd.py   # Dataset parser (MATLAB/JSON to YOLO)
│   ├── augmentation.py    # Indoor-specific augmentations
│   └── sunrgbd_yolo/      # Processed dataset (generated)
├── training/               # Training scripts
│   └── train.py           # YOLOv8 training script
├── evaluation/             # Evaluation and inference
│   ├── evaluate.py        # Comprehensive evaluation
│   └── inference.py       # Single image/video inference
├── configs/                # Configuration files
│   ├── training_config.yaml
│   └── config.py          # Config loader
├── models/                 # Saved model weights
├── logs/                   # Training logs
└── outputs/                # Evaluation outputs and predictions
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Convert SUN RGB-D dataset to YOLO format:

```bash
python Agents/data_prep/parse_sunrgbd.py \
    --sunrgbd_path ./SUNRGBD \
    --output_path ./Agents/data_prep/sunrgbd_yolo \
    --min_samples 10 \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1
```

**Parameters:**
- `--sunrgbd_path`: Path to SUN RGB-D dataset root
- `--output_path`: Output directory for processed dataset
- `--min_samples`: Minimum samples per class to include
- `--train_ratio`, `--val_ratio`, `--test_ratio`: Dataset split ratios

### 3. Training

Train YOLOv8 model on the processed dataset:

```bash
python Agents/training/train.py \
    --config ./Agents/configs/training_config.yaml \
    --validate
```

**Key Training Features:**
- ✅ YOLOv8 architecture (nano to extra-large)
- ✅ Transfer learning from COCO pretrained weights
- ✅ Automatic mixed precision (AMP) training
- ✅ Cosine annealing learning rate schedule
- ✅ Early stopping with patience
- ✅ Automatic model checkpointing
- ✅ TensorBoard/WandB logging support
- ✅ Indoor-specific data augmentation

### 4. Evaluation

Evaluate trained model with comprehensive metrics:

```bash
python Agents/evaluation/evaluate.py \
    --model ./Agents/models/best.pt \
    --config ./Agents/configs/training_config.yaml \
    --split test \
    --benchmark \
    --visualize ./Agents/data_prep/sunrgbd_yolo/images/test \
    --num_vis 20
```

**Evaluation Outputs:**
- mAP@50 and mAP@50-95
- Per-class precision/recall
- Confusion matrix
- Inference speed benchmarks (FPS)
- Sample prediction visualizations
- Comprehensive evaluation report (JSON + text)

### 5. Inference

Run inference on new images/videos:

**Single Image:**
```bash
python Agents/evaluation/inference.py \
    --model ./Agents/models/best.pt \
    --source path/to/image.jpg \
    --type image \
    --save \
    --show
```

**Batch Processing:**
```bash
python Agents/evaluation/inference.py \
    --model ./Agents/models/best.pt \
    --source path/to/images/ \
    --type batch \
    --save
```

**Video Processing:**
```bash
python Agents/evaluation/inference.py \
    --model ./Agents/models/best.pt \
    --source path/to/video.mp4 \
    --type video \
    --save \
    --show
```

**Webcam (Real-time):**
```bash
python Agents/evaluation/inference.py \
    --model ./Agents/models/best.pt \
    --source 0 \
    --type webcam
```

## ⚙️ Configuration

Edit `Agents/configs/training_config.yaml` to customize:

### Model Settings
```yaml
model:
  name: "yolov8s"  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
  pretrained: true
```

### Training Hyperparameters
```yaml
training:
  epochs: 100
  batch_size: 16
  optimizer: "AdamW"
  lr0: 0.001
  lrf: 0.01
  weight_decay: 0.0005
  warmup_epochs: 3
```

### Data Augmentation
```yaml
augmentation:
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  degrees: 10.0
  translate: 0.1
  scale: 0.5
  fliplr: 0.5
  mosaic: 1.0
  mixup: 0.1
```

### Early Stopping
```yaml
early_stopping:
  enabled: true
  patience: 20
  min_delta: 0.001
```

## 📊 Dataset Format

The pipeline converts SUN RGB-D annotations to YOLO format:

**Input (SUN RGB-D):**
```
SUNRGBD/
├── kv1/NYUdata/NYU0428/
│   ├── image/NYU0428.jpg
│   └── annotation2Dfinal/index.json
├── kv2/...
└── misc/SUNRGBDtoolbox/
```

**Output (YOLO):**
```
sunrgbd_yolo/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── sunrgbd.yaml
└── classes.txt
```

## 🎨 Data Augmentation

Indoor-specific augmentation pipeline includes:
- ✅ Geometric: Perspective, affine, horizontal flip
- ✅ Lighting: Brightness/contrast, shadows, tone curves
- ✅ Quality: Motion blur, Gaussian noise, compression
- ✅ Color: HSV shifts, RGB shifts, channel shuffle
- ✅ Advanced: Mosaic, MixUp (YOLO-specific)

## 📈 Metrics & Visualization

The evaluation script generates:

1. **Detection Metrics:**
   - mAP@50, mAP@50-95
   - Precision, Recall, F1-score
   - Per-class AP comparison

2. **Speed Benchmarks:**
   - Mean/median inference time
   - FPS (frames per second)
   - Inference time distribution

3. **Visualizations:**
   - Per-class AP bar charts
   - Confusion matrix
   - Precision-Recall curves
   - Sample predictions with bounding boxes

4. **Reports:**
   - JSON evaluation report
   - Human-readable text summary
   - Per-class metrics CSV

## 🔧 Advanced Usage

### Custom Model Architecture

To use a different model size:
```yaml
model:
  name: "yolov8x"  # Larger model for better accuracy
```

### Fine-tuning from Custom Checkpoint

```python
from ultralytics import YOLO

model = YOLO("path/to/custom/checkpoint.pt")
model.train(data="sunrgbd.yaml", epochs=50, resume=True)
```

### Export to Different Formats

```bash
# ONNX (for deployment)
python Agents/training/train.py --export onnx

# TorchScript
python Agents/training/train.py --export torchscript

# TensorRT (for NVIDIA GPUs)
python Agents/training/train.py --export engine
```

## 🐛 Troubleshooting

### Out of Memory (OOM)
- Reduce batch size in `training_config.yaml`
- Use a smaller model (e.g., yolov8n instead of yolov8x)
- Disable caching: `cache: false`

### Slow Training
- Enable AMP: `amp: true`
- Increase number of workers: `workers: 8`
- Use GPU if available

### Poor Performance
- Increase epochs
- Adjust learning rate
- Try different augmentation parameters
- Ensure class balance in dataset

## 📝 Best Practices

1. **Data Preparation:**
   - Set `min_samples` appropriately to filter rare classes
   - Use 80-10-10 split for train-val-test

2. **Training:**
   - Start with pretrained weights (`pretrained: true`)
   - Use early stopping to prevent overfitting
   - Monitor validation metrics, not just training loss

3. **Evaluation:**
   - Always benchmark on test set, not validation
   - Consider both mAP and inference speed
   - Visualize predictions to identify failure cases

4. **Deployment:**
   - Export to ONNX for cross-platform deployment
   - Optimize confidence threshold based on use case
   - Consider model quantization for edge devices

## 📚 References

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [SUN RGB-D Dataset](https://rgbd.cs.princeton.edu/)
- Original Paper: *SUN RGB-D: A RGB-D Scene Understanding Benchmark Suite* (CVPR 2015)

## 📄 License

This project is for educational and research purposes. Please cite the original SUN RGB-D paper if you use this dataset.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## ⭐ Acknowledgments

- Ultralytics team for YOLOv8
- SUN RGB-D dataset authors
- PyTorch and OpenCV communities
