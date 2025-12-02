"""
Depth Estimation Model Training
Trains a U-Net model to predict depth from RGB images.
"""

import os
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from datetime import datetime
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DepthDataset(Dataset):
    """Dataset for depth estimation."""

    def __init__(self, data_path: str, split: str = 'train', transform=None):
        """
        Initialize dataset.

        Args:
            data_path: Path to prepared dataset
            split: Dataset split (train/val/test)
            transform: Optional transform
        """
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform

        # Get RGB and depth file paths
        self.rgb_dir = self.data_path / split / 'rgb'
        self.depth_dir = self.data_path / split / 'depth'

        self.rgb_files = sorted(list(self.rgb_dir.glob('*.png')))
        self.depth_files = sorted(list(self.depth_dir.glob('*.png')))

        assert len(self.rgb_files) == len(self.depth_files), \
            f"RGB and depth file counts don't match: {len(self.rgb_files)} vs {len(self.depth_files)}"

        logger.info(f"Loaded {split} dataset: {len(self.rgb_files)} samples")

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        # Read RGB image
        rgb = cv2.imread(str(self.rgb_files[idx]))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Read depth image
        depth = cv2.imread(str(self.depth_files[idx]), cv2.IMREAD_ANYDEPTH)

        # Convert to float and normalize
        rgb = rgb.astype(np.float32) / 255.0
        depth = depth.astype(np.float32) / 65535.0

        # Convert to tensors (C, H, W)
        rgb = torch.from_numpy(rgb).permute(2, 0, 1)
        depth = torch.from_numpy(depth).unsqueeze(0)

        return rgb, depth


class UNet(nn.Module):
    """Lightweight U-Net for depth estimation."""

    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(512, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(256, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 64)

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(64, 32)

        # Output
        self.out = nn.Conv2d(32, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def conv_block(self, in_channels, out_channels):
        """Convolutional block with two conv layers."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        # Output
        out = self.out(dec1)
        out = self.sigmoid(out)

        return out


class DepthLoss(nn.Module):
    """Combined loss for depth estimation."""

    def __init__(self, alpha=0.5, beta=0.5):
        super(DepthLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):
        # L1 loss
        l1 = self.l1_loss(pred, target)

        # MSE loss
        mse = self.mse_loss(pred, target)

        # Gradient loss (edge-aware)
        pred_dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        pred_dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        target_dx = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        target_dy = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])

        grad_loss = torch.mean(torch.abs(pred_dx - target_dx)) + \
                   torch.mean(torch.abs(pred_dy - target_dy))

        # Combined loss
        total_loss = self.alpha * l1 + self.beta * mse + 0.1 * grad_loss

        return total_loss, l1, mse, grad_loss


class DepthTrainer:
    """Trainer for depth estimation model."""

    def __init__(self, config: dict):
        """
        Initialize trainer.

        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Create directories
        self.log_dir = Path('./logs')
        self.model_dir = Path('./models')
        self.log_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)

        # Initialize model
        self.model = UNet().to(self.device)
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Loss function
        self.criterion = DepthLoss()

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        epoch_l1 = 0.0
        epoch_mse = 0.0
        epoch_grad = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1}/{self.config['epochs']}")

        for batch_idx, (rgb, depth) in enumerate(pbar):
            rgb = rgb.to(self.device)
            depth = depth.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            pred_depth = self.model(rgb)

            # Compute loss
            loss, l1, mse, grad = self.criterion(pred_depth, depth)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update metrics
            epoch_loss += loss.item()
            epoch_l1 += l1.item()
            epoch_mse += mse.item()
            epoch_grad += grad.item()

            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'l1': l1.item()
            })

            # Log to TensorBoard
            global_step = self.current_epoch * len(train_loader) + batch_idx
            self.writer.add_scalar('Train/Batch_Loss', loss.item(), global_step)

        # Average losses
        n_batches = len(train_loader)
        avg_loss = epoch_loss / n_batches
        avg_l1 = epoch_l1 / n_batches
        avg_mse = epoch_mse / n_batches
        avg_grad = epoch_grad / n_batches

        return avg_loss, avg_l1, avg_mse, avg_grad

    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        val_l1 = 0.0
        val_mse = 0.0

        with torch.no_grad():
            for rgb, depth in val_loader:
                rgb = rgb.to(self.device)
                depth = depth.to(self.device)

                # Forward pass
                pred_depth = self.model(rgb)

                # Compute loss
                loss, l1, mse, _ = self.criterion(pred_depth, depth)

                val_loss += loss.item()
                val_l1 += l1.item()
                val_mse += mse.item()

        # Average losses
        n_batches = len(val_loader)
        avg_loss = val_loss / n_batches
        avg_l1 = val_l1 / n_batches
        avg_mse = val_mse / n_batches

        return avg_loss, avg_l1, avg_mse

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        # Save latest checkpoint
        checkpoint_path = self.model_dir / 'latest.pth'
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.model_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with val_loss: {self.best_val_loss:.4f}")

    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Loss curves
        epochs = range(1, len(self.train_losses) + 1)
        axes[0].plot(epochs, self.train_losses, 'b-', label='Train Loss')
        axes[0].plot(epochs, self.val_losses, 'r-', label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Learning rate
        lrs = [self.optimizer.param_groups[0]['lr']] * len(epochs)
        axes[1].plot(epochs, lrs, 'g-')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_curves.png', dpi=300)
        plt.close()

    def train(self, train_loader, val_loader):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")

        for epoch in range(self.config['epochs']):
            self.current_epoch = epoch
            start_time = time.time()

            # Train
            train_loss, train_l1, train_mse, train_grad = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validate
            val_loss, val_l1, val_mse = self.validate(val_loader)
            self.val_losses.append(val_loss)

            # Update learning rate
            self.scheduler.step(val_loss)

            epoch_time = time.time() - start_time

            # Log metrics
            logger.info(f"Epoch {epoch+1}/{self.config['epochs']} - "
                       f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                       f"Time: {epoch_time:.2f}s")

            # TensorBoard logging
            self.writer.add_scalars('Loss', {
                'train': train_loss,
                'val': val_loss
            }, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            self.save_checkpoint(epoch, is_best)

            # Plot curves every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.plot_training_curves()

        # Final plots
        self.plot_training_curves()

        # Save training summary
        summary = {
            'epochs_trained': self.config['epochs'],
            'best_val_loss': float(self.best_val_loss),
            'final_train_loss': float(self.train_losses[-1]),
            'final_val_loss': float(self.val_losses[-1]),
            'config': self.config
        }

        with open(self.log_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info("Training completed!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

        self.writer.close()


def main():
    """Main training function."""
    import argparse

    parser = argparse.ArgumentParser(description="Train depth estimation model")
    parser.add_argument('--data_path', type=str, default='./data_prepare',
                       help='Path to prepared dataset')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')

    args = parser.parse_args()

    # Configuration
    config = {
        'data_path': args.data_path,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'num_workers': args.num_workers
    }

    # Create datasets
    train_dataset = DepthDataset(config['data_path'], split='train')
    val_dataset = DepthDataset(config['data_path'], split='val')

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    # Create trainer and train
    trainer = DepthTrainer(config)
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
