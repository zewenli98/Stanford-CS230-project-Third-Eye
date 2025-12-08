"""
Comprehensive Evaluation Script for YOLOv8 Object Detection
Calculates mAP, per-class metrics, inference speed, and generates visualizations.
"""

import os
import sys
from pathlib import Path
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm
import time
from typing import Dict, List, Tuple
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from ultralytics import YOLO
from configs.config import Config
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YOLOEvaluator:
    """Comprehensive evaluator for YOLOv8 models."""

    def __init__(self, model_path: str, config_path: str = None):
        """
        Initialize evaluator.

        Args:
            model_path: Path to trained model weights
            config_path: Path to configuration file
        """
        self.model_path = Path(model_path)
        self.config = Config(config_path)

        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = YOLO(str(model_path))

        # Output directory
        self.output_dir = Path('./Agents/outputs/evaluation')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Metrics storage
        self.metrics = {}
        self.inference_times = []

    def evaluate_model(self, data_yaml: str = None, split: str = 'test') -> Dict:
        """
        Evaluate model on dataset.

        Args:
            data_yaml: Path to dataset YAML
            split: Dataset split to evaluate on

        Returns:
            Dictionary of metrics
        """
        if data_yaml is None:
            data_yaml = self.config.get('data.yaml_path')

        logger.info(f"Evaluating model on {split} set")

        # Run validation
        metrics = self.model.val(
            data=data_yaml,
            split=split,
            batch=self.config.get('training.batch_size', 16),
            imgsz=self.config.get('data.img_size', 640),
            conf=self.config.get('validation.conf_thres', 0.25),
            iou=self.config.get('validation.iou_thres', 0.45),
            max_det=self.config.get('validation.max_det', 300),
            plots=True,
            save_json=True,
        )

        # Extract metrics
        self.metrics = {
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
            'per_class_ap50': metrics.box.ap50.tolist() if hasattr(metrics.box, 'ap50') else [],
            'per_class_ap': metrics.box.ap.tolist() if hasattr(metrics.box, 'ap') else [],
        }

        # Log metrics
        logger.info("Evaluation Metrics:")
        logger.info(f"  mAP@50: {self.metrics['mAP50']:.4f}")
        logger.info(f"  mAP@50-95: {self.metrics['mAP50-95']:.4f}")
        logger.info(f"  Precision: {self.metrics['precision']:.4f}")
        logger.info(f"  Recall: {self.metrics['recall']:.4f}")

        return self.metrics

    def calculate_per_class_metrics(self, data_yaml: str = None) -> pd.DataFrame:
        """
        Calculate detailed per-class metrics.

        Args:
            data_yaml: Path to dataset YAML

        Returns:
            DataFrame with per-class metrics
        """
        if data_yaml is None:
            data_yaml = self.config.get('data.yaml_path')

        logger.info("Calculating per-class metrics...")

        # Get class names
        import yaml
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        class_names = data_config['names']

        # Create DataFrame
        if 'per_class_ap50' in self.metrics and len(self.metrics['per_class_ap50']) > 0:
            df = pd.DataFrame({
                'Class': class_names[:len(self.metrics['per_class_ap50'])],
                'AP@50': self.metrics['per_class_ap50'],
                'AP@50-95': self.metrics['per_class_ap'][:len(self.metrics['per_class_ap50'])],
            })

            # Sort by AP@50-95
            df = df.sort_values('AP@50-95', ascending=False)

            # Save to CSV
            csv_path = self.output_dir / 'per_class_metrics.csv'
            df.to_csv(csv_path, index=False)
            logger.info(f"Per-class metrics saved to {csv_path}")

            return df
        else:
            logger.warning("No per-class metrics available")
            return pd.DataFrame()

    def benchmark_inference_speed(self, num_warmup: int = 10, num_runs: int = 100,
                                   img_size: int = None) -> Dict[str, float]:
        """
        Benchmark inference speed.

        Args:
            num_warmup: Number of warmup runs
            num_runs: Number of benchmark runs
            img_size: Image size for inference

        Returns:
            Dictionary with speed metrics
        """
        if img_size is None:
            img_size = self.config.get('data.img_size', 640)

        logger.info(f"Benchmarking inference speed ({num_runs} runs after {num_warmup} warmup)...")

        # Create dummy image
        dummy_image = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

        # Warmup
        for _ in range(num_warmup):
            _ = self.model(dummy_image, verbose=False)

        # Benchmark
        inference_times = []
        for _ in tqdm(range(num_runs), desc="Benchmarking"):
            start_time = time.time()
            _ = self.model(dummy_image, verbose=False)
            end_time = time.time()
            inference_times.append((end_time - start_time) * 1000)  # Convert to ms

        # Calculate statistics
        speed_metrics = {
            'mean_ms': np.mean(inference_times),
            'std_ms': np.std(inference_times),
            'min_ms': np.min(inference_times),
            'max_ms': np.max(inference_times),
            'median_ms': np.median(inference_times),
            'fps': 1000.0 / np.mean(inference_times),
        }

        self.inference_times = inference_times

        logger.info("Inference Speed Metrics:")
        logger.info(f"  Mean: {speed_metrics['mean_ms']:.2f} ms")
        logger.info(f"  Std: {speed_metrics['std_ms']:.2f} ms")
        logger.info(f"  FPS: {speed_metrics['fps']:.2f}")

        # Save metrics
        with open(self.output_dir / 'speed_metrics.json', 'w') as f:
            json.dump(speed_metrics, f, indent=2)

        return speed_metrics

    def plot_metrics(self):
        """Generate and save metric visualizations."""
        logger.info("Generating metric visualizations...")

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

        # 1. Inference time distribution
        if len(self.inference_times) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.hist(self.inference_times, bins=50, edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(self.inference_times), color='r', linestyle='--',
                      label=f'Mean: {np.mean(self.inference_times):.2f} ms')
            ax.set_xlabel('Inference Time (ms)')
            ax.set_ylabel('Frequency')
            ax.set_title('Inference Time Distribution')
            ax.legend()
            plt.tight_layout()
            plt.savefig(self.output_dir / 'inference_time_distribution.png', dpi=300)
            plt.close()

        # 2. Per-class AP comparison
        if 'per_class_ap50' in self.metrics and len(self.metrics['per_class_ap50']) > 0:
            # Load class names
            data_yaml = self.config.get('data.yaml_path')
            import yaml
            with open(data_yaml, 'r') as f:
                data_config = yaml.safe_load(f)
            class_names = data_config['names']

            # Create bar plot
            fig, ax = plt.subplots(1, 1, figsize=(14, 8))
            x = np.arange(len(self.metrics['per_class_ap50']))
            width = 0.35

            ax.bar(x - width/2, self.metrics['per_class_ap50'], width, label='AP@50', alpha=0.8)
            ax.bar(x + width/2, self.metrics['per_class_ap'][:len(x)], width,
                  label='AP@50-95', alpha=0.8)

            ax.set_xlabel('Class')
            ax.set_ylabel('Average Precision')
            ax.set_title('Per-Class Average Precision Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(class_names[:len(x)], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'per_class_ap_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 3. Summary metrics bar plot
        if self.metrics:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            metrics_to_plot = ['mAP50', 'mAP50-95', 'precision', 'recall']
            values = [self.metrics.get(m, 0) for m in metrics_to_plot]

            bars = ax.bar(metrics_to_plot, values, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Metrics')
            ax.set_ylim([0, 1])

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(self.output_dir / 'summary_metrics.png', dpi=300)
            plt.close()

        logger.info(f"Visualizations saved to {self.output_dir}")

    def visualize_predictions(self, image_dir: str, num_samples: int = 10,
                             conf_threshold: float = None):
        """
        Generate prediction visualizations on sample images.

        Args:
            image_dir: Directory containing test images
            num_samples: Number of samples to visualize
            conf_threshold: Confidence threshold for predictions
        """
        if conf_threshold is None:
            conf_threshold = self.config.get('inference.conf_thres', 0.25)

        logger.info(f"Generating prediction visualizations for {num_samples} samples...")

        # Create output directory
        vis_dir = self.output_dir / 'predictions'
        vis_dir.mkdir(exist_ok=True)

        # Get image files
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        image_files = image_files[:num_samples]

        # Generate predictions
        for img_path in tqdm(image_files, desc="Generating predictions"):
            try:
                # Run inference
                results = self.model(str(img_path), conf=conf_threshold, verbose=False)

                # Save visualization
                for i, result in enumerate(results):
                    # Plot and save
                    result_img = result.plot()
                    output_path = vis_dir / f"{img_path.stem}_pred.jpg"
                    cv2.imwrite(str(output_path), result_img)

            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")

        logger.info(f"Predictions saved to {vis_dir}")

    def save_evaluation_report(self, speed_metrics: Dict = None):
        """
        Save comprehensive evaluation report.

        Args:
            speed_metrics: Dictionary of speed metrics
        """
        report = {
            'model_path': str(self.model_path),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': self.metrics,
        }

        if speed_metrics:
            report['speed'] = speed_metrics

        # Save as JSON
        report_path = self.output_dir / 'evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Save as text
        text_report_path = self.output_dir / 'evaluation_report.txt'
        with open(text_report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("YOLO OBJECT DETECTION EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")

            f.write(f"Model: {self.model_path}\n")
            f.write(f"Date: {report['timestamp']}\n\n")

            f.write("-"*80 + "\n")
            f.write("DETECTION METRICS\n")
            f.write("-"*80 + "\n")
            f.write(f"mAP@50:     {self.metrics.get('mAP50', 0):.4f}\n")
            f.write(f"mAP@50-95:  {self.metrics.get('mAP50-95', 0):.4f}\n")
            f.write(f"Precision:  {self.metrics.get('precision', 0):.4f}\n")
            f.write(f"Recall:     {self.metrics.get('recall', 0):.4f}\n\n")

            if speed_metrics:
                f.write("-"*80 + "\n")
                f.write("INFERENCE SPEED\n")
                f.write("-"*80 + "\n")
                f.write(f"Mean time:   {speed_metrics['mean_ms']:.2f} ms\n")
                f.write(f"Std time:    {speed_metrics['std_ms']:.2f} ms\n")
                f.write(f"Min time:    {speed_metrics['min_ms']:.2f} ms\n")
                f.write(f"Max time:    {speed_metrics['max_ms']:.2f} ms\n")
                f.write(f"FPS:         {speed_metrics['fps']:.2f}\n\n")

            f.write("="*80 + "\n")

        logger.info(f"Evaluation report saved to {report_path}")
        logger.info(f"Text report saved to {text_report_path}")


def main():
    """Main evaluation function."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 model")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model weights'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='./configs/training_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Path to dataset YAML file'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to evaluate'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run inference speed benchmark'
    )
    parser.add_argument(
        '--visualize',
        type=str,
        default=None,
        help='Directory with images to visualize predictions'
    )
    parser.add_argument(
        '--num_vis',
        type=int,
        default=10,
        help='Number of images to visualize'
    )

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = YOLOEvaluator(args.model, args.config)

    # Evaluate
    evaluator.evaluate_model(data_yaml=args.data, split=args.split)

    # Calculate per-class metrics
    evaluator.calculate_per_class_metrics(data_yaml=args.data)

    # Benchmark if requested
    speed_metrics = None
    if args.benchmark:
        speed_metrics = evaluator.benchmark_inference_speed()

    # Generate plots
    evaluator.plot_metrics()

    # Visualize predictions if requested
    if args.visualize:
        evaluator.visualize_predictions(args.visualize, num_samples=args.num_vis)

    # Save report
    evaluator.save_evaluation_report(speed_metrics)

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
