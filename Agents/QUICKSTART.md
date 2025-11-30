# Quick Start Guide

Get up and running with the SUN RGB-D object detection pipeline in minutes!

## ⚡ 5-Minute Setup

### 1. Install Dependencies (1 min)

```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline (automatic)

```bash
python main.py --mode full --sunrgbd_path ./SUNRGBD
```

This will automatically:
1. ✅ Convert SUN RGB-D to YOLO format
2. ✅ Train YOLOv8 model
3. ✅ Evaluate on test set
4. ✅ Generate visualizations

---

## 🎯 Step-by-Step Guide

### Option A: Full Automated Pipeline

```bash
# One command to rule them all
python main.py --mode full --sunrgbd_path ./SUNRGBD
```

### Option B: Manual Step-by-Step

**Step 1: Prepare Data**
```bash
python main.py --mode prepare --sunrgbd_path ./SUNRGBD
```

**Step 2: Train Model**
```bash
python main.py --mode train
```

**Step 3: Evaluate**
```bash
python main.py --mode evaluate --model ./Agents/models/best.pt
```

---

## 🔧 Common Use Cases

### 1. Quick Experiment with Small Model

Edit `Agents/configs/training_config.yaml`:
```yaml
model:
  name: "yolov8n"  # Fastest, smallest model
training:
  epochs: 50       # Quick training
  batch_size: 32   # Larger batches for speed
```

### 2. Best Accuracy (Large Model)

```yaml
model:
  name: "yolov8x"  # Largest, most accurate
training:
  epochs: 200      # Longer training
  batch_size: 8    # Smaller batch for large model
```

### 3. Test on Custom Image

```bash
python Agents/evaluation/inference.py \
    --model ./Agents/models/best.pt \
    --source your_image.jpg \
    --type image \
    --save \
    --show
```

### 4. Real-time Webcam Detection

```bash
python Agents/evaluation/inference.py \
    --model ./Agents/models/best.pt \
    --source 0 \
    --type webcam
```

---

## 📊 Where to Find Results

After running the pipeline, check these locations:

```
Agents/
├── models/                      # Trained model weights
│   └── sunrgbd_detection/
│       └── best.pt             # Best model
│
├── outputs/
│   ├── evaluation/             # Evaluation results
│   │   ├── evaluation_report.json
│   │   ├── evaluation_report.txt
│   │   ├── per_class_metrics.csv
│   │   ├── summary_metrics.png
│   │   └── predictions/        # Sample predictions
│   │
│   └── inference/              # Inference outputs
│
└── logs/
    ├── training.log            # Training logs
    └── pipeline.log            # Pipeline logs
```

---

## 🎨 Customization Cheat Sheet

### Change Model Size
```yaml
model:
  name: "yolov8n"  # n (nano), s (small), m (medium), l (large), x (xlarge)
```

### Adjust Training Speed
```yaml
training:
  batch_size: 32   # Higher = faster but more memory
  workers: 8       # More workers = faster data loading
hardware:
  amp: true        # Mixed precision = 2x faster
```

### Modify Augmentation Strength
```yaml
augmentation:
  mosaic: 1.0      # 1.0 = always, 0.0 = never
  mixup: 0.1       # Probability of mixup
  degrees: 10.0    # Rotation range (±degrees)
  scale: 0.5       # Scale variation
```

### Change Validation Frequency
```yaml
validation:
  val_period: 5    # Validate every N epochs
```

---

## 🐛 Troubleshooting

### Error: Out of Memory
**Solution:** Reduce batch size
```yaml
training:
  batch_size: 8  # Or even smaller: 4, 2, 1
```

### Error: Dataset not found
**Solution:** Check path
```bash
python main.py --mode prepare --sunrgbd_path /full/path/to/SUNRGBD
```

### Error: CUDA not available
**Solution:** Use CPU (slower but works)
```yaml
hardware:
  device: "cpu"
```

### Training too slow
**Solutions:**
1. Use smaller model: `yolov8n`
2. Reduce epochs: `epochs: 50`
3. Enable AMP: `amp: true`
4. Increase workers: `workers: 8`

---

## 🚀 Next Steps

1. **Experiment with hyperparameters** in `training_config.yaml`
2. **Try different model sizes** (yolov8n → yolov8x)
3. **Adjust augmentation** for your specific use case
4. **Export model** for deployment (ONNX, TensorRT)
5. **Fine-tune** on your custom indoor dataset

---

## 📚 Learn More

- Full documentation: See `README.md`
- Configuration reference: See `Agents/configs/training_config.yaml`
- API documentation: See docstrings in each `.py` file

---

## 💡 Pro Tips

1. **Start small**: Begin with `yolov8n` and fewer epochs to test your setup
2. **Monitor GPU usage**: Use `nvidia-smi` to check GPU utilization
3. **Use TensorBoard**: Monitor training in real-time
   ```bash
   tensorboard --logdir ./Agents/models
   ```
4. **Save your configs**: Copy `training_config.yaml` before experimenting
5. **Check logs**: Always review `logs/training.log` if something goes wrong

---

Happy training! 🎉
