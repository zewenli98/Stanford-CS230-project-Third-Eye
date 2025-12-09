"""
Generate depth images from RGB images using trained model and compare with ground truth.

This script:
1. Loads the trained depth estimation model (best.pth)
2. Generates depth images for all RGB images in queries/prompts.csv
3. Saves generated depth images to queries/generated_images/
4. Compares generated depth images with original ground truth depth images
"""

import os
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import csv
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Add Agents/distance_calculator to path for imports
sys.path.append(str(Path(__file__).parent / 'Agents' / 'distance_calculator'))
from train import UNet

# QUERY_PATH = "./queries-sample10/"
QUERY_PATH = "./queries/"
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DepthMetrics:
    """Compute depth estimation metrics for comparison."""

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
    def compute_mse(pred, target):
        """Mean Squared Error."""
        return np.mean((pred - target) ** 2)

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
        """Compute all metrics for a single image pair."""
        # Flatten arrays
        pred_flat = pred.flatten()
        target_flat = target.flatten()

        # Remove invalid values (depth = 0 or negative)
        valid_mask = (target_flat > 0) & (pred_flat > 0) & np.isfinite(target_flat) & np.isfinite(pred_flat)
        pred_valid = pred_flat[valid_mask]
        target_valid = target_flat[valid_mask]

        if len(pred_valid) == 0:
            logger.warning("No valid pixels found for metric computation")
            return None

        metrics = {
            'rmse': cls.compute_rmse(pred_valid, target_valid),
            'mae': cls.compute_mae(pred_valid, target_valid),
            'mse': cls.compute_mse(pred_valid, target_valid),
            'abs_rel': cls.compute_abs_rel(pred_valid, target_valid),
            'delta1': cls.compute_delta_accuracy(pred_valid, target_valid, 1.25),
            'delta2': cls.compute_delta_accuracy(pred_valid, target_valid, 1.25**2),
            'delta3': cls.compute_delta_accuracy(pred_valid, target_valid, 1.25**3),
            'num_valid_pixels': len(pred_valid)
        }

        return metrics


class DepthGenerator:
    """Generate depth images using trained model."""

    def __init__(self, model_path: str, device: str = None):
        """
        Initialize depth generator.

        Args:
            model_path: Path to trained model checkpoint (best.pth)
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()

    def _load_model(self, model_path: str):
        """Load trained UNet model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        logger.info(f"Loading model from {model_path}")

        # Initialize UNet (input channels=3 for RGB, output channels=1 for depth)
        model = UNet(in_channels=3, out_channels=1)
        model = model.to(self.device)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)

        logger.info("Model loaded successfully")
        return model

    def preprocess_image(self, rgb_image: np.ndarray) -> torch.Tensor:
        """
        Preprocess RGB image for model input.

        Args:
            rgb_image: RGB image as numpy array (H, W, 3)

        Returns:
            Preprocessed tensor (1, 3, H, W)
        """
        # Convert BGR to RGB if needed (OpenCV loads as BGR)
        if rgb_image.shape[2] == 3:
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        # Resize to model input size (assuming 256x256)
        target_size = (256, 256)
        rgb_resized = cv2.resize(rgb_image, target_size)

        # Normalize to [0, 1]
        rgb_normalized = rgb_resized.astype(np.float32) / 255.0

        # Convert to tensor (C, H, W)
        rgb_tensor = torch.from_numpy(rgb_normalized).permute(2, 0, 1)

        # Add batch dimension (1, C, H, W)
        rgb_tensor = rgb_tensor.unsqueeze(0)

        return rgb_tensor.to(self.device)

    def generate_depth(self, rgb_path: str) -> np.ndarray:
        """
        Generate depth map from RGB image.

        Args:
            rgb_path: Path to RGB image

        Returns:
            Depth map as numpy array (H, W) with values in [0, 1]
        """
        # Load RGB image
        rgb_image = cv2.imread(rgb_path)
        if rgb_image is None:
            raise ValueError(f"Could not load image from {rgb_path}")

        original_size = (rgb_image.shape[1], rgb_image.shape[0])  # (W, H)

        # Preprocess
        rgb_tensor = self.preprocess_image(rgb_image)

        # Generate depth
        with torch.no_grad():
            depth_tensor = self.model(rgb_tensor)

        # Convert to numpy
        depth_map = depth_tensor.squeeze().cpu().numpy()

        # Resize back to original size
        depth_map = cv2.resize(depth_map, original_size)

        # Ensure values are in [0, 1]
        depth_map = np.clip(depth_map, 0, 1)

        return depth_map

    def save_depth_image(self, depth_map: np.ndarray, output_path: str):
        """
        Save depth map as 16-bit PNG image.

        Args:
            depth_map: Depth map with values in [0, 1]
            output_path: Path to save depth image
        """
        # Convert to 16-bit (0-65535 range)
        depth_16bit = (depth_map * 65535).astype(np.uint16)

        # Save
        cv2.imwrite(output_path, depth_16bit)
        logger.debug(f"Saved depth image to {output_path}")


def generate_depth_images_from_csv(
    csv_path: str = QUERY_PATH + "prompts.csv",
    images_folder: str = QUERY_PATH + "images",
    model_path: str = "Agents/distance_calculator/models/latest.pth",
    output_folder: str = QUERY_PATH + "generated_images",
    device: str = None
):
    """
    Generate depth images for all RGB images listed in CSV.

    Args:
        csv_path: Path to prompts CSV file
        images_folder: Folder containing RGB images
        model_path: Path to trained model
        output_folder: Folder to save generated depth images
        device: Device to run on ('cuda', 'cpu', or None for auto)

    Returns:
        List of tuples (rgb_path, generated_depth_path, original_depth_path)
    """
    logger.info("="*60)
    logger.info("GENERATING DEPTH IMAGES FROM CSV")
    logger.info("="*60)

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Initialize depth generator
    generator = DepthGenerator(model_path, device)

    # Read CSV
    logger.info(f"Reading CSV from {csv_path}")
    image_pairs = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    logger.info(f"Found {len(rows)} rows in CSV")

    # Generate depth for each image
    results = []
    for row in tqdm(rows, desc="Generating depth images"):
        image_name = row.get('image_name', '')
        depth_image_name = row.get('depth_image', '')

        if not image_name:
            continue

        # Construct paths
        rgb_path = os.path.join(images_folder, image_name)

        if not os.path.exists(rgb_path):
            logger.warning(f"RGB image not found: {rgb_path}")
            continue

        # Generate output filename (use same name as original depth image if available)
        if depth_image_name:
            output_filename = depth_image_name
        else:
            # Generate name from RGB filename
            base_name = os.path.splitext(image_name)[0]
            output_filename = f"{base_name}_depth.png"

        output_path = os.path.join(output_folder, output_filename)

        try:
            # Generate depth
            depth_map = generator.generate_depth(rgb_path)

            # Save depth image
            generator.save_depth_image(depth_map, output_path)

            # Record result
            original_depth_path = os.path.join(images_folder, depth_image_name) if depth_image_name else None
            results.append((rgb_path, output_path, original_depth_path))

        except Exception as e:
            logger.error(f"Error generating depth for {image_name}: {e}")
            continue

    logger.info(f"Generated {len(results)} depth images")
    logger.info(f"Saved to: {output_folder}")

    return results


def compare_depth_images(
    generated_folder: str = QUERY_PATH + "generated_images",
    original_folder: str = QUERY_PATH + "images",
    csv_path: str = QUERY_PATH + "prompts.csv",
    output_folder: str = QUERY_PATH + "depth_comparison"
):
    """
    Compare generated depth images with original ground truth depth images.

    Args:
        generated_folder: Folder containing generated depth images
        original_folder: Folder containing original depth images
        csv_path: Path to prompts CSV
        output_folder: Folder to save comparison results

    Returns:
        Dictionary with overall metrics
    """
    logger.info("="*60)
    logger.info("COMPARING DEPTH IMAGES")
    logger.info("="*60)

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Read CSV to get image pairs
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    all_metrics = []
    comparison_results = []

    for row in tqdm(rows, desc="Comparing depth images"):
        depth_image_name = row.get('depth_image', '')
        image_name = row.get('image_name', '')

        if not depth_image_name:
            continue

        # Construct paths
        generated_path = os.path.join(generated_folder, depth_image_name)
        original_path = os.path.join(original_folder, depth_image_name)

        if not os.path.exists(generated_path):
            logger.warning(f"Generated depth not found: {generated_path}")
            continue

        if not os.path.exists(original_path):
            logger.warning(f"Original depth not found: {original_path}")
            continue

        try:
            # Load depth images
            generated_depth = cv2.imread(generated_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            original_depth = cv2.imread(original_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

            # Normalize to [0, 1]
            if generated_depth.max() > 1:
                generated_depth = generated_depth / 65535.0
            if original_depth.max() > 1:
                original_depth = original_depth / 65535.0

            # Compute metrics
            metrics = DepthMetrics.compute_all_metrics(generated_depth, original_depth)

            if metrics is not None:
                metrics['image_name'] = image_name
                metrics['depth_image_name'] = depth_image_name
                all_metrics.append(metrics)

                # Save comparison visualization
                save_comparison_visualization(
                    generated_depth,
                    original_depth,
                    depth_image_name,
                    metrics,
                    output_folder
                )

        except Exception as e:
            logger.error(f"Error comparing {depth_image_name}: {e}")
            continue

    # Compute overall statistics
    if all_metrics:
        overall_metrics = compute_overall_metrics(all_metrics)

        # Convert to Python types for JSON serialization
        overall_metrics_json = convert_to_python_types(overall_metrics)
        all_metrics_json = convert_to_python_types(all_metrics)

        # Save metrics to JSON
        metrics_path = os.path.join(output_folder, "comparison_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump({
                'overall': overall_metrics_json,
                'per_image': all_metrics_json
            }, f, indent=2)

        logger.info(f"Saved metrics to {metrics_path}")

        # Print summary
        print_metrics_summary(overall_metrics)

        return overall_metrics
    else:
        logger.warning("No valid comparisons were made")
        return None


def save_comparison_visualization(
    generated_depth: np.ndarray,
    original_depth: np.ndarray,
    filename: str,
    metrics: dict,
    output_folder: str
):
    """Save visualization comparing generated and original depth."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Generated depth
    im1 = axes[0, 0].imshow(generated_depth, cmap='plasma', vmin=0, vmax=1)
    axes[0, 0].set_title('Generated Depth')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])

    # Original depth
    im2 = axes[0, 1].imshow(original_depth, cmap='plasma', vmin=0, vmax=1)
    axes[0, 1].set_title('Ground Truth Depth')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])

    # Error map
    error_map = np.abs(generated_depth - original_depth)
    im3 = axes[1, 0].imshow(error_map, cmap='hot', vmin=0, vmax=0.3)
    axes[1, 0].set_title('Absolute Error')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0])

    # Metrics text
    axes[1, 1].axis('off')
    metrics_text = f"""
Metrics:
RMSE: {metrics['rmse']:.4f}
MAE: {metrics['mae']:.4f}
MSE: {metrics['mse']:.4f}
Abs Rel: {metrics['abs_rel']:.4f}
δ < 1.25: {metrics['delta1']:.4f}
δ < 1.25²: {metrics['delta2']:.4f}
δ < 1.25³: {metrics['delta3']:.4f}
Valid Pixels: {metrics['num_valid_pixels']}
    """
    axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center', family='monospace')

    plt.tight_layout()

    # Save
    base_name = os.path.splitext(filename)[0]
    output_path = os.path.join(output_folder, f"{base_name}_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def convert_to_python_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    else:
        return obj


def compute_overall_metrics(all_metrics: list) -> dict:
    """Compute overall statistics from all image metrics."""
    overall = {}

    metric_keys = ['rmse', 'mae', 'mse', 'abs_rel', 'delta1', 'delta2', 'delta3']

    for key in metric_keys:
        values = [m[key] for m in all_metrics if key in m]
        if values:
            overall[f'{key}_mean'] = float(np.mean(values))
            overall[f'{key}_std'] = float(np.std(values))
            overall[f'{key}_median'] = float(np.median(values))
            overall[f'{key}_min'] = float(np.min(values))
            overall[f'{key}_max'] = float(np.max(values))

    overall['num_images'] = len(all_metrics)

    return overall


def print_metrics_summary(overall_metrics: dict):
    """Print summary of metrics."""
    print("\n" + "="*60)
    print("DEPTH COMPARISON SUMMARY")
    print("="*60)
    print(f"Number of images compared: {overall_metrics['num_images']}")
    print("\nMetrics (mean ± std):")
    print(f"  RMSE:    {overall_metrics['rmse_mean']:.4f} ± {overall_metrics['rmse_std']:.4f}")
    print(f"  MAE:     {overall_metrics['mae_mean']:.4f} ± {overall_metrics['mae_std']:.4f}")
    print(f"  MSE:     {overall_metrics['mse_mean']:.4f} ± {overall_metrics['mse_std']:.4f}")
    print(f"  Abs Rel: {overall_metrics['abs_rel_mean']:.4f} ± {overall_metrics['abs_rel_std']:.4f}")
    print(f"\nAccuracy (δ < threshold):")
    print(f"  δ < 1.25:   {overall_metrics['delta1_mean']:.4f} ± {overall_metrics['delta1_std']:.4f}")
    print(f"  δ < 1.25²:  {overall_metrics['delta2_mean']:.4f} ± {overall_metrics['delta2_std']:.4f}")
    print(f"  δ < 1.25³:  {overall_metrics['delta3_mean']:.4f} ± {overall_metrics['delta3_std']:.4f}")
    print("="*60 + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate and compare depth images')
    parser.add_argument('--mode', type=str, choices=['generate', 'compare', 'both'], default='both',
                       help='Mode: generate depth, compare, or both')
    parser.add_argument('--model_path', type=str,
                       default='Agents/distance_calculator/models/best.pth',
                       help='Path to trained model')
    parser.add_argument('--csv_path', type=str, default=QUERY_PATH + 'prompts.csv',
                       help='Path to prompts CSV')
    parser.add_argument('--images_folder', type=str, default=QUERY_PATH + 'images',
                       help='Folder containing RGB and original depth images')
    parser.add_argument('--output_folder', type=str, default=QUERY_PATH + 'generated_images',
                       help='Folder to save generated depth images')
    parser.add_argument('--comparison_output', type=str, default=QUERY_PATH + 'depth_comparison',
                       help='Folder to save comparison results')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default=None,
                       help='Device to run on')

    args = parser.parse_args()

    if args.mode in ['generate', 'both']:
        logger.info("Starting depth image generation...")
        results = generate_depth_images_from_csv(
            csv_path=args.csv_path,
            images_folder=args.images_folder,
            model_path=args.model_path,
            output_folder=args.output_folder,
            device=args.device
        )
        logger.info(f"Generated {len(results)} depth images")

    if args.mode in ['compare', 'both']:
        logger.info("Starting depth image comparison...")
        overall_metrics = compare_depth_images(
            generated_folder=args.output_folder,
            original_folder=args.images_folder,
            csv_path=args.csv_path,
            output_folder=args.comparison_output
        )

    logger.info("Done!")


if __name__ == "__main__":
    main()
