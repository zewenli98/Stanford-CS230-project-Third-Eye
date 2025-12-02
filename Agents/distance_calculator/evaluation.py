"""
Depth Estimation Model Evaluation
Evaluates trained depth estimation model with comprehensive metrics and visualizations.
"""

import os
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from datetime import datetime
import time

# Import from train.py
from train import UNet, DepthDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DepthMetrics:
    """Compute depth estimation metrics."""

    @staticmethod
    def compute_rmse(pred, target):
        """Root Mean Square Error."""
        return np.sqrt(np.mean((pred - target) ** 2))

    @staticmethod
    def compute_mae(pred, target):
        """Mean Absolute Error."""
        return np.mean(np.abs(pred - target))

    @staticmethod
    def compute_abs_rel(pred, target):
        """Absolute Relative Error."""
        return np.mean(np.abs(pred - target) / (target + 1e-6))

    @staticmethod
    def compute_sq_rel(pred, target):
        """Squared Relative Error."""
        return np.mean(((pred - target) ** 2) / (target + 1e-6))

    @staticmethod
    def compute_delta_accuracy(pred, target, threshold=1.25):
        """
        Percentage of pixels where max(pred/target, target/pred) < threshold.
        Common thresholds: 1.25, 1.25^2, 1.25^3
        """
        ratio = np.maximum(pred / (target + 1e-6), target / (pred + 1e-6))
        return np.mean(ratio < threshold)

    @classmethod
    def compute_all_metrics(cls, pred, target):
        """Compute all metrics."""
        # Flatten arrays
        pred_flat = pred.flatten()
        target_flat = target.flatten()

        # Remove invalid values
        valid_mask = (target_flat > 0) & (pred_flat > 0)
        pred_valid = pred_flat[valid_mask]
        target_valid = target_flat[valid_mask]

        if len(pred_valid) == 0:
            return None

        metrics = {
            'rmse': cls.compute_rmse(pred_valid, target_valid),
            'mae': cls.compute_mae(pred_valid, target_valid),
            'abs_rel': cls.compute_abs_rel(pred_valid, target_valid),
            'sq_rel': cls.compute_sq_rel(pred_valid, target_valid),
            'delta1': cls.compute_delta_accuracy(pred_valid, target_valid, 1.25),
            'delta2': cls.compute_delta_accuracy(pred_valid, target_valid, 1.25**2),
            'delta3': cls.compute_delta_accuracy(pred_valid, target_valid, 1.25**3),
        }

        return metrics


class DepthEvaluator:
    """Evaluator for depth estimation models."""

    def __init__(self, model_path: str, data_path: str, device: str = None):
        """
        Initialize evaluator.

        Args:
            model_path: Path to trained model checkpoint
            data_path: Path to prepared dataset
            device: Device to use (cuda/cpu)
        """
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)

        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Load model
        self.model = self.load_model()

        # Output directory
        self.output_dir = Path('./outputs/evaluation')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Metrics storage
        self.all_metrics = []

    def load_model(self):
        """Load trained model."""
        logger.info(f"Loading model from {self.model_path}")

        # Create model
        model = UNet().to(self.device)

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        logger.info(f"Model loaded (trained for {checkpoint.get('epoch', 'unknown')} epochs)")

        return model

    def evaluate(self, split: str = 'test', batch_size: int = 8, num_workers: int = 4):
        """
        Evaluate model on dataset.

        Args:
            split: Dataset split to evaluate
            batch_size: Batch size for evaluation
            num_workers: Number of data loading workers

        Returns:
            Dictionary of average metrics
        """
        logger.info(f"Evaluating on {split} set...")

        # Create dataset and loader
        dataset = DepthDataset(self.data_path, split=split)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        # Evaluation loop
        all_metrics = []
        inference_times = []

        with torch.no_grad():
            for rgb, depth_gt in tqdm(loader, desc="Evaluating"):
                rgb = rgb.to(self.device)
                depth_gt = depth_gt.cpu().numpy()

                # Measure inference time
                start_time = time.time()
                depth_pred = self.model(rgb)
                inference_time = time.time() - start_time
                inference_times.append(inference_time / len(rgb))

                depth_pred = depth_pred.cpu().numpy()

                # Compute metrics for each sample in batch
                for i in range(len(rgb)):
                    pred = depth_pred[i, 0]
                    gt = depth_gt[i, 0]

                    metrics = DepthMetrics.compute_all_metrics(pred, gt)
                    if metrics is not None:
                        all_metrics.append(metrics)

        # Average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

        # Add inference speed
        avg_metrics['inference_time_ms'] = np.mean(inference_times) * 1000
        avg_metrics['fps'] = 1.0 / np.mean(inference_times)

        # Store metrics
        self.all_metrics = all_metrics

        # Log results
        logger.info("\nEvaluation Results:")
        logger.info(f"  RMSE: {avg_metrics['rmse']:.4f}")
        logger.info(f"  MAE: {avg_metrics['mae']:.4f}")
        logger.info(f"  Abs Rel: {avg_metrics['abs_rel']:.4f}")
        logger.info(f"  Delta < 1.25: {avg_metrics['delta1']:.4f}")
        logger.info(f"  Delta < 1.25²: {avg_metrics['delta2']:.4f}")
        logger.info(f"  Delta < 1.25³: {avg_metrics['delta3']:.4f}")
        logger.info(f"  Inference Time: {avg_metrics['inference_time_ms']:.2f} ms")
        logger.info(f"  FPS: {avg_metrics['fps']:.2f}")

        return avg_metrics

    def visualize_predictions(self, split: str = 'test', num_samples: int = 10):
        """
        Visualize model predictions.

        Args:
            split: Dataset split
            num_samples: Number of samples to visualize
        """
        logger.info(f"Generating visualizations for {num_samples} samples...")

        # Create dataset
        dataset = DepthDataset(self.data_path, split=split)

        # Sample indices
        indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)

        # Create visualization directory
        vis_dir = self.output_dir / 'predictions'
        vis_dir.mkdir(exist_ok=True)

        with torch.no_grad():
            for idx in tqdm(indices, desc="Visualizing"):
                rgb, depth_gt = dataset[idx]

                # Predict
                rgb_batch = rgb.unsqueeze(0).to(self.device)
                depth_pred = self.model(rgb_batch)
                depth_pred = depth_pred[0, 0].cpu().numpy()

                # Convert tensors to numpy
                rgb_np = rgb.permute(1, 2, 0).cpu().numpy()
                depth_gt_np = depth_gt[0].cpu().numpy()

                # Create visualization
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))

                # RGB image
                axes[0].imshow(rgb_np)
                axes[0].set_title('RGB Input')
                axes[0].axis('off')

                # Ground truth depth
                im1 = axes[1].imshow(depth_gt_np, cmap='plasma', vmin=0, vmax=1)
                axes[1].set_title('Ground Truth Depth')
                axes[1].axis('off')
                plt.colorbar(im1, ax=axes[1])

                # Predicted depth
                im2 = axes[2].imshow(depth_pred, cmap='plasma', vmin=0, vmax=1)
                axes[2].set_title('Predicted Depth')
                axes[2].axis('off')
                plt.colorbar(im2, ax=axes[2])

                # Error map
                error = np.abs(depth_pred - depth_gt_np)
                im3 = axes[3].imshow(error, cmap='hot', vmin=0, vmax=0.2)
                axes[3].set_title('Absolute Error')
                axes[3].axis('off')
                plt.colorbar(im3, ax=axes[3])

                plt.tight_layout()
                plt.savefig(vis_dir / f'sample_{idx:04d}.png', dpi=150, bbox_inches='tight')
                plt.close()

        logger.info(f"Visualizations saved to {vis_dir}")

    def plot_metrics_distribution(self):
        """Plot distribution of metrics across samples."""
        if not self.all_metrics:
            logger.warning("No metrics to plot. Run evaluation first.")
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        metrics_to_plot = ['rmse', 'mae', 'abs_rel', 'delta1', 'delta2', 'delta3']

        for idx, metric_name in enumerate(metrics_to_plot):
            values = [m[metric_name] for m in self.all_metrics]

            axes[idx].hist(values, bins=50, edgecolor='black', alpha=0.7)
            axes[idx].axvline(np.mean(values), color='r', linestyle='--',
                            label=f'Mean: {np.mean(values):.4f}')
            axes[idx].set_xlabel(metric_name.upper())
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'{metric_name.upper()} Distribution')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics_distribution.png', dpi=300)
        plt.close()

        logger.info(f"Metrics distribution saved to {self.output_dir / 'metrics_distribution.png'}")

    def create_comparison_grid(self, split: str = 'test', num_samples: int = 9):
        """
        Create a grid comparing predictions and ground truth.

        Args:
            split: Dataset split
            num_samples: Number of samples (should be perfect square)
        """
        logger.info(f"Creating comparison grid...")

        dataset = DepthDataset(self.data_path, split=split)
        indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)

        rows = int(np.sqrt(num_samples))
        cols = rows

        fig, axes = plt.subplots(rows, cols * 2, figsize=(20, 10))

        with torch.no_grad():
            for plot_idx, data_idx in enumerate(indices):
                rgb, depth_gt = dataset[data_idx]

                # Predict
                rgb_batch = rgb.unsqueeze(0).to(self.device)
                depth_pred = self.model(rgb_batch)
                depth_pred = depth_pred[0, 0].cpu().numpy()
                depth_gt_np = depth_gt[0].cpu().numpy()

                # Calculate position in grid
                row = plot_idx // cols
                col_gt = (plot_idx % cols) * 2
                col_pred = col_gt + 1

                # Plot ground truth
                axes[row, col_gt].imshow(depth_gt_np, cmap='plasma', vmin=0, vmax=1)
                axes[row, col_gt].set_title('GT', fontsize=8)
                axes[row, col_gt].axis('off')

                # Plot prediction
                axes[row, col_pred].imshow(depth_pred, cmap='plasma', vmin=0, vmax=1)
                axes[row, col_pred].set_title('Pred', fontsize=8)
                axes[row, col_pred].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'comparison_grid.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Comparison grid saved to {self.output_dir / 'comparison_grid.png'}")

    def save_evaluation_report(self, metrics: dict):
        """
        Save comprehensive evaluation report.

        Args:
            metrics: Dictionary of average metrics
        """
        # JSON report
        report = {
            'model_path': str(self.model_path),
            'data_path': str(self.data_path),
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'num_samples': len(self.all_metrics)
        }

        json_path = self.output_dir / 'evaluation_report.json'
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Text report
        text_path = self.output_dir / 'evaluation_report.txt'
        with open(text_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DEPTH ESTIMATION MODEL EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")

            f.write(f"Model: {self.model_path}\n")
            f.write(f"Dataset: {self.data_path}\n")
            f.write(f"Timestamp: {report['timestamp']}\n")
            f.write(f"Samples Evaluated: {len(self.all_metrics)}\n\n")

            f.write("-"*80 + "\n")
            f.write("DEPTH ACCURACY METRICS\n")
            f.write("-"*80 + "\n")
            f.write(f"RMSE:           {metrics['rmse']:.4f}\n")
            f.write(f"MAE:            {metrics['mae']:.4f}\n")
            f.write(f"Abs Rel:        {metrics['abs_rel']:.4f}\n")
            f.write(f"Sq Rel:         {metrics['sq_rel']:.4f}\n\n")

            f.write("-"*80 + "\n")
            f.write("THRESHOLD ACCURACY\n")
            f.write("-"*80 + "\n")
            f.write(f"δ < 1.25:       {metrics['delta1']:.4f} ({metrics['delta1']*100:.2f}%)\n")
            f.write(f"δ < 1.25²:      {metrics['delta2']:.4f} ({metrics['delta2']*100:.2f}%)\n")
            f.write(f"δ < 1.25³:      {metrics['delta3']:.4f} ({metrics['delta3']*100:.2f}%)\n\n")

            f.write("-"*80 + "\n")
            f.write("INFERENCE SPEED\n")
            f.write("-"*80 + "\n")
            f.write(f"Inference Time: {metrics['inference_time_ms']:.2f} ms\n")
            f.write(f"FPS:            {metrics['fps']:.2f}\n\n")

            f.write("="*80 + "\n")

        logger.info(f"Evaluation report saved to {json_path}")
        logger.info(f"Text report saved to {text_path}")


def main():
    """Main evaluation function."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate depth estimation model")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='./data_prepare',
        help='Path to prepared dataset'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to evaluate'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization of predictions'
    )
    parser.add_argument(
        '--num_vis',
        type=int,
        default=10,
        help='Number of samples to visualize'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu)'
    )

    args = parser.parse_args()

    # Create evaluator
    evaluator = DepthEvaluator(
        model_path=args.model,
        data_path=args.data_path,
        device=args.device
    )

    # Run evaluation
    metrics = evaluator.evaluate(
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Plot metrics distribution
    evaluator.plot_metrics_distribution()

    # Generate visualizations if requested
    if args.visualize:
        evaluator.visualize_predictions(split=args.split, num_samples=args.num_vis)
        evaluator.create_comparison_grid(split=args.split, num_samples=9)

    # Save evaluation report
    evaluator.save_evaluation_report(metrics)

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
