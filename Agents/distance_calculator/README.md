# Depth Estimation Module

Monocular depth estimation using deep learning for the SUN RGB-D dataset. This module trains a U-Net model to predict depth maps from single RGB images.

## 📁 Structure

```
distance_calculator/
├── data_prepare.py      # Dataset preparation
├── train.py             # Model training
├── evaluation.py        # Model evaluation
├── data_prepare/        # Prepared dataset (generated)
├── models/              # Saved models (generated)
├── logs/                # Training logs (generated)
└── outputs/             # Evaluation outputs (generated)
```

## 🚀 Quick Start

### 1. Prepare Dataset

Convert SUN RGB-D dataset to training format:

```bash
python data_prepare.py \
    --sunrgbd_path ../../SUNRGBD \
    --output_path ./data_prepare \
    --img_height 240 \
    --img_width 320
```

**Output:**
- `data_prepare/train/` - Training RGB and depth images
- `data_prepare/val/` - Validation RGB and depth images
- `data_prepare/test/` - Test RGB and depth images
- `data_prepare/dataset_info.json` - Dataset statistics

### 2. Train Model

Train the depth estimation model:

```bash
python train.py \
    --data_path ./data_prepare \
    --batch_size 8 \
    --epochs 50 \
    --lr 0.001
```

**Features:**
- ✅ Lightweight U-Net architecture
- ✅ Combined L1 + MSE + Gradient loss
- ✅ Learning rate scheduling
- ✅ Automatic checkpointing (saves best model)
- ✅ TensorBoard logging
- ✅ Training curve visualization

**Outputs:**
- `models/best.pth` - Best model checkpoint
- `models/latest.pth` - Latest checkpoint
- `logs/training.log` - Training log file
- `logs/training_curves.png` - Loss curves
- `logs/training_summary.json` - Training summary

### 3. Evaluate Model

Evaluate the trained model:

```bash
python evaluation.py \
    --model ./models/best.pth \
    --data_path ./data_prepare \
    --split test \
    --visualize \
    --num_vis 20
```

**Metrics Computed:**
- ✅ RMSE (Root Mean Square Error)
- ✅ MAE (Mean Absolute Error)
- ✅ Abs Rel (Absolute Relative Error)
- ✅ Sq Rel (Squared Relative Error)
- ✅ Delta Accuracy (δ < 1.25, 1.25², 1.25³)
- ✅ Inference Speed (FPS)

**Outputs:**
- `outputs/evaluation/evaluation_report.json` - Metrics in JSON
- `outputs/evaluation/evaluation_report.txt` - Human-readable report
- `outputs/evaluation/metrics_distribution.png` - Metric histograms
- `outputs/evaluation/comparison_grid.png` - GT vs Pred grid
- `outputs/evaluation/predictions/` - Individual predictions

## 📊 Model Architecture

**U-Net with Skip Connections:**

```
Input RGB (3, H, W)
    ↓
Encoder (32 → 64 → 128 → 256 → 512)
    ↓
Bottleneck (512)
    ↓
Decoder (512 → 256 → 128 → 64 → 32) + Skip Connections
    ↓
Output Depth (1, H, W)
```

**Model Size:** ~7.7M parameters (lightweight for real-time applications)

## 📈 Loss Function

Combined loss with three components:

1. **L1 Loss** (α=0.5): Absolute difference
2. **MSE Loss** (β=0.5): Squared difference
3. **Gradient Loss** (0.1): Edge-aware smoothness

```
Total Loss = α·L1 + β·MSE + 0.1·Gradient
```

## 🔧 Training Configuration

### Default Parameters

```python
{
    'batch_size': 8,
    'epochs': 50,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'optimizer': 'Adam',
    'scheduler': 'ReduceLROnPlateau'
}
```

### Recommended Settings

**Fast Training (Quick Test):**
```bash
python train.py --epochs 20 --batch_size 16
```

**Best Accuracy:**
```bash
python train.py --epochs 100 --batch_size 4 --lr 0.0005
```

**Balanced:**
```bash
python train.py --epochs 50 --batch_size 8 --lr 0.001
```

## 📊 Expected Results

On SUN RGB-D dataset, you should expect:

| Metric | Value |
|--------|-------|
| RMSE | ~0.15-0.25 |
| MAE | ~0.10-0.18 |
| δ < 1.25 | >80% |
| δ < 1.25² | >95% |
| δ < 1.25³ | >98% |
| FPS | 20-60 (depending on hardware) |

## 🎨 Visualization Examples

The evaluation script generates:

1. **Individual Predictions**
   - RGB input
   - Ground truth depth
   - Predicted depth
   - Absolute error map

2. **Comparison Grid**
   - Side-by-side GT vs Predictions

3. **Metrics Distribution**
   - Histograms of all metrics

## 🔄 Data Preprocessing

**RGB Images:**
- Resized to target resolution (default: 320×240)
- Normalized to [0, 1]
- Converted to tensors (C, H, W)

**Depth Images:**
- Converted from mm to meters
- Clamped to [0.1, 10] meters
- Normalized to [0, 1]
- Saved as 16-bit PNG for precision

## 💡 Tips & Best Practices

### Training

1. **Start with small resolution** (240×320) for faster iteration
2. **Monitor validation loss** to avoid overfitting
3. **Use learning rate scheduling** for better convergence
4. **Save checkpoints frequently** in case of interruptions

### Data Preparation

1. **Check dataset statistics** in `dataset_info.json`
2. **Verify RGB-depth alignment** by visualizing samples
3. **Handle invalid depth values** (zeros, infinities)
4. **Balance dataset splits** (80/10/10)

### Evaluation

1. **Always evaluate on test set**, not validation
2. **Visualize predictions** to identify failure modes
3. **Check edge cases** (far objects, occlusions)
4. **Measure inference speed** for deployment planning

## 🐛 Troubleshooting

### Out of Memory

**Solution:** Reduce batch size
```bash
python train.py --batch_size 4
```

### Slow Training

**Solutions:**
- Reduce image resolution
- Increase num_workers
- Use GPU if available

### Poor Accuracy

**Solutions:**
- Train for more epochs
- Reduce learning rate
- Check data quality
- Add more augmentation

### Depth artifacts

**Solutions:**
- Adjust loss weights
- Check data normalization
- Verify depth range

## 📚 Technical Details

### Depth Normalization

```python
# Convert raw depth to meters
depth_meters = depth_raw / 1000.0

# Clamp to valid range
depth_meters = np.clip(depth_meters, 0.1, 10.0)

# Normalize to [0, 1]
depth_normalized = (depth_meters - 0.1) / (10.0 - 0.1)
```

### Evaluation Metrics

**Delta Accuracy:**
```python
δ = max(pred/gt, gt/pred)
accuracy = percentage of pixels where δ < threshold
```

**Common thresholds:** 1.25, 1.25², 1.25³

## 🔗 Integration

To use the trained model in your application:

```python
import torch
from train import UNet

# Load model
model = UNet()
checkpoint = torch.load('models/best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict depth
with torch.no_grad():
    depth_pred = model(rgb_tensor)
```

## 📄 File Formats

**RGB Images:** PNG format, 8-bit per channel
**Depth Images:** PNG format, 16-bit grayscale (for precision)
**Checkpoints:** PyTorch .pth files with model state and training info

## ⚙️ Advanced Usage

### Custom Loss Weights

Edit `train.py`:
```python
self.criterion = DepthLoss(alpha=0.6, beta=0.3)  # Adjust α, β
```

### Different Image Sizes

```bash
python data_prepare.py --img_height 480 --img_width 640
python train.py --data_path ./data_prepare  # Will auto-detect size
```

### Resume Training

```python
# Modify train.py to load checkpoint and resume
checkpoint = torch.load('models/latest.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

## 📊 Monitoring Training

### TensorBoard

```bash
tensorboard --logdir ./logs
```

View at: http://localhost:6006

**Metrics tracked:**
- Training loss (per batch and epoch)
- Validation loss
- Learning rate
- L1, MSE, Gradient losses

## 🎯 Next Steps

1. **Experiment with architectures** (try ResNet encoder, attention mechanisms)
2. **Add data augmentation** (color jittering, crops)
3. **Multi-scale training** for better edge detection
4. **Deploy model** (export to ONNX for production)
5. **Fine-tune on custom data** for specific scenarios

---

## 📖 References

- U-Net: [Ronneberger et al., 2015]
- SUN RGB-D: [Song et al., CVPR 2015]
- Depth Estimation metrics: [Eigen et al., NIPS 2014]
