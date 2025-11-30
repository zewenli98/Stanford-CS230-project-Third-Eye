"""
YOLOv8 Training Script for SUN RGB-D Object Detection
Implements training with early stopping, checkpointing, and comprehensive logging.
"""

import os
import sys
from pathlib import Path
import logging
import torch
import yaml
from datetime import datetime
import json
import shutil

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ultralytics import YOLO
from ultralytics.utils.callbacks import Callbacks
from configs.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./Agents/logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class YOLOTrainer:
    """Wrapper class for YOLOv8 training with custom callbacks and monitoring."""

    def __init__(self, config_path: str = None):
        """
        Initialize trainer.

        Args:
            config_path: Path to configuration file
        """
        self.config = Config(config_path)
        self.config.validate()

        # Create output directories
        self.save_dir = Path(self.config.get('logging.save_dir'))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Early stopping parameters
        self.best_fitness = 0.0
        self.patience_counter = 0
        self.best_epoch = 0

        # Initialize model
        self.model = None
        self.results = None

    def setup_model(self):
        """Initialize YOLOv8 model."""
        model_name = self.config.get('model.name')
        pretrained = self.config.get('model.pretrained')

        logger.info(f"Initializing {model_name} model (pretrained={pretrained})")

        if pretrained:
            # Load pretrained model
            self.model = YOLO(f"{model_name}.pt")
        else:
            # Load model architecture only
            self.model = YOLO(f"{model_name}.yaml")

        logger.info(f"Model initialized: {self.model.model}")

    def check_early_stopping(self, epoch: int, fitness: float) -> bool:
        """
        Check if training should stop early.

        Args:
            epoch: Current epoch number
            fitness: Current fitness value (mAP)

        Returns:
            True if should stop, False otherwise
        """
        if not self.config.get('early_stopping.enabled'):
            return False

        patience = self.config.get('early_stopping.patience')
        min_delta = self.config.get('early_stopping.min_delta')

        # Check if fitness improved
        if fitness > self.best_fitness + min_delta:
            self.best_fitness = fitness
            self.best_epoch = epoch
            self.patience_counter = 0
            logger.info(f"New best fitness: {fitness:.4f} at epoch {epoch}")
        else:
            self.patience_counter += 1
            logger.info(f"No improvement for {self.patience_counter} epochs (patience: {patience})")

        # Check if should stop
        if self.patience_counter >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            logger.info(f"Best fitness: {self.best_fitness:.4f} at epoch {self.best_epoch}")
            return True

        return False

    def train(self):
        """Run training."""
        if self.model is None:
            self.setup_model()

        # Get training parameters
        data_yaml = self.config.get('data.yaml_path')
        epochs = self.config.get('training.epochs')
        batch_size = self.config.get('training.batch_size')
        img_size = self.config.get('data.img_size')
        device = self.config.get('hardware.device')
        workers = self.config.get('hardware.workers')
        amp = self.config.get('hardware.amp')

        # Optimizer parameters
        optimizer = self.config.get('training.optimizer')
        lr0 = self.config.get('training.lr0')
        lrf = self.config.get('training.lrf')
        momentum = self.config.get('training.momentum')
        weight_decay = self.config.get('training.weight_decay')
        warmup_epochs = self.config.get('training.warmup_epochs')

        # Augmentation parameters
        hsv_h = self.config.get('augmentation.hsv_h')
        hsv_s = self.config.get('augmentation.hsv_s')
        hsv_v = self.config.get('augmentation.hsv_v')
        degrees = self.config.get('augmentation.degrees')
        translate = self.config.get('augmentation.translate')
        scale = self.config.get('augmentation.scale')
        shear = self.config.get('augmentation.shear')
        perspective = self.config.get('augmentation.perspective')
        flipud = self.config.get('augmentation.flipud')
        fliplr = self.config.get('augmentation.fliplr')
        mosaic = self.config.get('augmentation.mosaic')
        mixup = self.config.get('augmentation.mixup')

        # Logging parameters
        project = self.config.get('logging.project')
        name = self.config.get('logging.name')
        verbose = self.config.get('logging.verbose')
        plots = self.config.get('logging.plots')

        # Validation parameters
        val_period = self.config.get('validation.val_period')
        save_period = self.config.get('checkpointing.save_period')

        logger.info("Starting training with configuration:")
        logger.info(f"  Data: {data_yaml}")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Image size: {img_size}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Optimizer: {optimizer}")
        logger.info(f"  Learning rate: {lr0} -> {lrf}")

        try:
            # Train the model
            self.results = self.model.train(
                # Data
                data=data_yaml,
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,

                # Hardware
                device=device,
                workers=workers,
                amp=amp,

                # Optimizer
                optimizer=optimizer,
                lr0=lr0,
                lrf=lrf,
                momentum=momentum,
                weight_decay=weight_decay,
                warmup_epochs=warmup_epochs,

                # Augmentation
                hsv_h=hsv_h,
                hsv_s=hsv_s,
                hsv_v=hsv_v,
                degrees=degrees,
                translate=translate,
                scale=scale,
                shear=shear,
                perspective=perspective,
                flipud=flipud,
                fliplr=fliplr,
                mosaic=mosaic,
                mixup=mixup,

                # Logging
                project=project,
                name=name,
                exist_ok=True,
                verbose=verbose,
                plots=plots,

                # Checkpointing
                save=True,
                save_period=save_period,

                # Validation
                val=True,
                val_period=val_period,

                # Resume
                resume=False,
            )

            logger.info("Training completed successfully!")
            self.save_training_summary()

            return self.results

        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    def save_training_summary(self):
        """Save training summary and metrics."""
        if self.results is None:
            logger.warning("No training results to save")
            return

        # Create summary directory
        summary_dir = self.save_dir / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)

        # Save training summary
        summary = {
            'model': self.config.get('model.name'),
            'dataset': self.config.get('data.yaml_path'),
            'epochs': self.config.get('training.epochs'),
            'batch_size': self.config.get('training.batch_size'),
            'image_size': self.config.get('data.img_size'),
            'optimizer': self.config.get('training.optimizer'),
            'learning_rate': self.config.get('training.lr0'),
            'timestamp': datetime.now().isoformat(),
        }

        # Add results if available
        if hasattr(self.results, 'results_dict'):
            summary['metrics'] = self.results.results_dict

        summary_file = summary_dir / f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Training summary saved to {summary_file}")

    def validate(self, data_yaml: str = None):
        """
        Run validation on the trained model.

        Args:
            data_yaml: Path to data YAML file
        """
        if self.model is None:
            logger.error("Model not trained or loaded")
            return

        if data_yaml is None:
            data_yaml = self.config.get('data.yaml_path')

        logger.info(f"Running validation on {data_yaml}")

        try:
            metrics = self.model.val(
                data=data_yaml,
                batch=self.config.get('training.batch_size'),
                imgsz=self.config.get('data.img_size'),
                conf=self.config.get('validation.conf_thres'),
                iou=self.config.get('validation.iou_thres'),
                max_det=self.config.get('validation.max_det'),
                save_json=self.config.get('validation.save_json'),
                plots=self.config.get('logging.plots'),
            )

            logger.info("Validation metrics:")
            logger.info(f"  mAP50: {metrics.box.map50:.4f}")
            logger.info(f"  mAP50-95: {metrics.box.map:.4f}")
            logger.info(f"  Precision: {metrics.box.mp:.4f}")
            logger.info(f"  Recall: {metrics.box.mr:.4f}")

            return metrics

        except Exception as e:
            logger.error(f"Error during validation: {e}")
            raise

    def export_model(self, format: str = 'onnx'):
        """
        Export model to different formats.

        Args:
            format: Export format (onnx, torchscript, coreml, etc.)
        """
        if self.model is None:
            logger.error("Model not trained or loaded")
            return

        logger.info(f"Exporting model to {format} format")

        try:
            export_path = self.model.export(format=format)
            logger.info(f"Model exported to {export_path}")
            return export_path
        except Exception as e:
            logger.error(f"Error exporting model: {e}")
            raise


def main():
    """Main training function."""
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLOv8 on SUN RGB-D dataset")
    parser.add_argument(
        '--config',
        type=str,
        default='./Agents/configs/training_config.yaml',
        help='Path to training configuration file'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run validation after training'
    )
    parser.add_argument(
        '--export',
        type=str,
        default=None,
        help='Export model format (onnx, torchscript, etc.)'
    )

    args = parser.parse_args()

    # Initialize trainer
    trainer = YOLOTrainer(config_path=args.config)

    # Train
    trainer.train()

    # Validate if requested
    if args.validate:
        trainer.validate()

    # Export if requested
    if args.export:
        trainer.export_model(format=args.export)


if __name__ == "__main__":
    main()
