"""
Main Pipeline Runner
Orchestrates the complete workflow: data preparation -> training -> evaluation
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./Agents/logs/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Pipeline:
    """Main pipeline orchestrator."""

    def __init__(self, config_path: str = None):
        """
        Initialize pipeline.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or './Agents/configs/training_config.yaml'
        self.base_dir = Path(__file__).parent

    def run_command(self, command: list, description: str):
        """
        Run a command and log output.

        Args:
            command: Command list
            description: Description of the command
        """
        logger.info(f"Starting: {description}")
        logger.info(f"Command: {' '.join(command)}")

        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"Completed: {description}")
            if result.stdout:
                logger.info(f"Output:\n{result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed: {description}")
            logger.error(f"Error:\n{e.stderr}")
            return False

    def prepare_data(self, sunrgbd_path: str = './SUNRGBD',
                     output_path: str = './Agents/data_prep/sunrgbd_yolo',
                     min_samples: int = 10):
        """
        Run data preparation step.

        Args:
            sunrgbd_path: Path to SUN RGB-D dataset
            output_path: Output path for processed data
            min_samples: Minimum samples per class
        """
        command = [
            'python',
            str(self.base_dir / 'data_prep' / 'parse_sunrgbd.py'),
            '--sunrgbd_path', sunrgbd_path,
            '--output_path', output_path,
            '--min_samples', str(min_samples),
            '--train_ratio', '0.8',
            '--val_ratio', '0.1',
            '--test_ratio', '0.1',
        ]

        return self.run_command(command, "Data Preparation")

    def train_model(self, validate: bool = True):
        """
        Run training step.

        Args:
            validate: Whether to validate after training
        """
        command = [
            'python',
            str(self.base_dir / 'training' / 'train.py'),
            '--config', self.config_path,
        ]

        if validate:
            command.append('--validate')

        return self.run_command(command, "Model Training")

    def evaluate_model(self, model_path: str, benchmark: bool = True,
                      visualize: bool = True, num_vis: int = 20):
        """
        Run evaluation step.

        Args:
            model_path: Path to trained model
            benchmark: Whether to benchmark speed
            visualize: Whether to visualize predictions
            num_vis: Number of visualizations
        """
        command = [
            'python',
            str(self.base_dir / 'evaluation' / 'evaluate.py'),
            '--model', model_path,
            '--config', self.config_path,
            '--split', 'test',
        ]

        if benchmark:
            command.append('--benchmark')

        if visualize:
            command.extend([
                '--visualize',
                './Agents/data_prep/sunrgbd_yolo/images/test',
                '--num_vis',
                str(num_vis)
            ])

        return self.run_command(command, "Model Evaluation")

    def run_full_pipeline(self, sunrgbd_path: str = './SUNRGBD',
                         skip_data_prep: bool = False,
                         skip_training: bool = False,
                         model_path: str = None):
        """
        Run the complete pipeline.

        Args:
            sunrgbd_path: Path to SUN RGB-D dataset
            skip_data_prep: Skip data preparation
            skip_training: Skip training
            model_path: Path to model (if skipping training)
        """
        logger.info("="*80)
        logger.info("STARTING FULL PIPELINE")
        logger.info("="*80)

        # Step 1: Data Preparation
        if not skip_data_prep:
            logger.info("\n[1/3] Data Preparation")
            if not self.prepare_data(sunrgbd_path):
                logger.error("Data preparation failed. Aborting pipeline.")
                return False
        else:
            logger.info("\n[1/3] Data Preparation (SKIPPED)")

        # Step 2: Training
        if not skip_training:
            logger.info("\n[2/3] Model Training")
            if not self.train_model(validate=True):
                logger.error("Training failed. Aborting pipeline.")
                return False

            # Find best model
            models_dir = Path('./Agents/models')
            if not model_path:
                # Try to find the latest training run
                best_models = list(models_dir.rglob('best.pt'))
                if best_models:
                    model_path = str(sorted(best_models, key=lambda x: x.stat().st_mtime)[-1])
                    logger.info(f"Found model: {model_path}")
        else:
            logger.info("\n[2/3] Model Training (SKIPPED)")

        # Step 3: Evaluation
        if model_path and Path(model_path).exists():
            logger.info("\n[3/3] Model Evaluation")
            if not self.evaluate_model(model_path, benchmark=True, visualize=True):
                logger.error("Evaluation failed.")
                return False
        else:
            logger.error(f"Model not found: {model_path}")
            return False

        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="SUN RGB-D Object Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python main.py --mode full --sunrgbd_path ./SUNRGBD

  # Only prepare data
  python main.py --mode prepare --sunrgbd_path ./SUNRGBD

  # Only train (assumes data is prepared)
  python main.py --mode train

  # Only evaluate (requires model path)
  python main.py --mode evaluate --model ./Agents/models/best.pt

  # Skip data preparation in full pipeline
  python main.py --mode full --skip_data_prep
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['full', 'prepare', 'train', 'evaluate'],
        default='full',
        help='Pipeline mode'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='./Agents/configs/training_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--sunrgbd_path',
        type=str,
        default='./SUNRGBD',
        help='Path to SUN RGB-D dataset'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='./Agents/data_prep/sunrgbd_yolo',
        help='Output path for processed dataset'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to model weights (for evaluation)'
    )
    parser.add_argument(
        '--min_samples',
        type=int,
        default=10,
        help='Minimum samples per class'
    )
    parser.add_argument(
        '--skip_data_prep',
        action='store_true',
        help='Skip data preparation (full mode only)'
    )
    parser.add_argument(
        '--skip_training',
        action='store_true',
        help='Skip training (full mode only)'
    )

    args = parser.parse_args()

    # Create pipeline
    pipeline = Pipeline(config_path=args.config)

    # Create necessary directories
    Path('./Agents/logs').mkdir(parents=True, exist_ok=True)

    # Run based on mode
    if args.mode == 'full':
        success = pipeline.run_full_pipeline(
            sunrgbd_path=args.sunrgbd_path,
            skip_data_prep=args.skip_data_prep,
            skip_training=args.skip_training,
            model_path=args.model
        )
    elif args.mode == 'prepare':
        success = pipeline.prepare_data(
            sunrgbd_path=args.sunrgbd_path,
            output_path=args.output_path,
            min_samples=args.min_samples
        )
    elif args.mode == 'train':
        success = pipeline.train_model(validate=True)
    elif args.mode == 'evaluate':
        if not args.model:
            logger.error("Model path required for evaluation mode")
            sys.exit(1)
        success = pipeline.evaluate_model(args.model)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
