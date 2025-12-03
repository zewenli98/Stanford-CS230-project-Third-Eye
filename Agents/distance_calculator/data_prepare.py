"""
Data Preparation for Depth Estimation
Reads SUN RGB-D dataset and prepares training data for depth prediction.
"""

import os
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import json
import logging
from tqdm import tqdm
import shutil
from typing import Tuple, List, Dict
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DepthDatasetPreparer:
    """Prepares SUN RGB-D dataset for depth estimation training."""

    def __init__(self, sunrgbd_path: str, output_path: str, img_size: Tuple[int, int] = (480, 640)):
        """
        Initialize dataset preparer.

        Args:
            sunrgbd_path: Path to SUNRGBD dataset
            output_path: Output path for prepared dataset
            img_size: Target image size (height, width)
        """
        self.sunrgbd_path = Path(sunrgbd_path)
        self.output_path = Path(output_path)
        self.img_size = img_size

        # Create output directories
        for split in ['train', 'val', 'test']:
            (self.output_path / split / 'rgb').mkdir(parents=True, exist_ok=True)
            (self.output_path / split / 'depth').mkdir(parents=True, exist_ok=True)

        self.dataset_stats = {
            'samples_found': 0,
            'samples_processed': 0,
            'samples_failed': 0,
            'depth_min': float('inf'),
            'depth_max': float('-inf'),
            'depth_mean': 0.0,
        }

    def read_depth_image(self, depth_path: Path) -> np.ndarray:
        """
        Read depth image from SUN RGB-D dataset.

        Args:
            depth_path: Path to depth image

        Returns:
            Depth array in meters
        """
        # SUN RGB-D depth images are stored as 16-bit PNG
        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)

        if depth_img is None:
            raise ValueError(f"Could not read depth image: {depth_path}")

        # Convert to meters (SUN RGB-D stores in mm or needs conversion)
        # The actual conversion depends on the sensor type
        depth_meters = depth_img.astype(np.float32) / 1000.0

        # Handle invalid depth values
        depth_meters[depth_meters <= 0] = 0
        depth_meters[depth_meters > 10] = 10  # Clamp to 10 meters

        return depth_meters

    def normalize_depth(self, depth: np.ndarray, min_depth: float = 0.1,
                       max_depth: float = 10.0) -> np.ndarray:
        """
        Normalize depth values to [0, 1] range.

        Args:
            depth: Depth array in meters
            min_depth: Minimum depth value
            max_depth: Maximum depth value

        Returns:
            Normalized depth array
        """
        # Clip values
        depth = np.clip(depth, min_depth, max_depth)

        # Normalize to [0, 1]
        depth_normalized = (depth - min_depth) / (max_depth - min_depth)

        return depth_normalized

    def find_rgb_depth_pairs(self) -> List[Dict]:
        """
        Find all RGB-Depth image pairs in the dataset.

        Returns:
            List of dictionaries containing RGB and depth paths
        """
        pairs = []

        logger.info("Scanning SUN RGB-D dataset for RGB-Depth pairs...")

        # Traverse dataset directory
        for root, dirs, files in os.walk(self.sunrgbd_path):
            root_path = Path(root)

            # Check for image directory
            image_dir = root_path / "image"
            depth_dir = root_path / "depth"

            if image_dir.exists() and depth_dir.exists():
                # Get RGB images
                rgb_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

                # Get depth images
                depth_files = list(depth_dir.glob("*.png")) + list(depth_dir.glob("*.jpg"))

                # Try to match RGB and depth files
                if rgb_files and depth_files:
                    # Usually there's one RGB and one depth per directory
                    rgb_file = rgb_files[0]
                    depth_file = depth_files[0]

                    pairs.append({
                        'rgb_path': rgb_file,
                        'depth_path': depth_file,
                        'sample_id': root_path.name
                    })

        self.dataset_stats['samples_found'] = len(pairs)
        logger.info(f"Found {len(pairs)} RGB-Depth pairs")

        return pairs

    def process_sample(self, sample: Dict, split: str, index: int) -> bool:
        """
        Process a single RGB-Depth pair.

        Args:
            sample: Sample dictionary with paths
            split: Dataset split (train/val/test)
            index: Sample index for naming

        Returns:
            True if successful, False otherwise
        """
        try:
            # Read RGB image
            rgb_img = cv2.imread(str(sample['rgb_path']))
            if rgb_img is None:
                logger.warning(f"Could not read RGB image: {sample['rgb_path']}")
                return False

            # Read depth image
            try:
                depth_img = self.read_depth_image(sample['depth_path'])
            except Exception as e:
                logger.warning(f"Could not read depth image {sample['depth_path']}: {e}")
                return False

            # Resize images
            rgb_resized = cv2.resize(rgb_img, (self.img_size[1], self.img_size[0]))
            depth_resized = cv2.resize(depth_img, (self.img_size[1], self.img_size[0]))

            # Update statistics
            valid_depth = depth_resized[depth_resized > 0]
            if len(valid_depth) > 0:
                self.dataset_stats['depth_min'] = min(self.dataset_stats['depth_min'], valid_depth.min())
                self.dataset_stats['depth_max'] = max(self.dataset_stats['depth_max'], valid_depth.max())

            # Normalize depth
            depth_normalized = self.normalize_depth(depth_resized)

            # Create unique filename
            filename = f"{split}_{index:05d}"

            # Save RGB image
            rgb_output = self.output_path / split / 'rgb' / f"{filename}.png"
            cv2.imwrite(str(rgb_output), rgb_resized)

            # Save depth as 16-bit PNG for precision
            depth_uint16 = (depth_normalized * 65535).astype(np.uint16)
            depth_output = self.output_path / split / 'depth' / f"{filename}.png"
            cv2.imwrite(str(depth_output), depth_uint16)

            return True

        except Exception as e:
            logger.error(f"Error processing sample {sample['sample_id']}: {e}")
            return False

    def split_dataset(self, pairs: List[Dict], train_ratio: float = 0.8,
                     val_ratio: float = 0.1, test_ratio: float = 0.1,
                     seed: int = 42):
        """
        Split dataset into train/val/test sets and process.

        Args:
            pairs: List of RGB-Depth pairs
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            seed: Random seed
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

        # Shuffle pairs
        random.seed(seed)
        random.shuffle(pairs)

        # Calculate split indices
        n_samples = len(pairs)
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)

        splits = {
            'train': pairs[:train_end],
            'val': pairs[train_end:val_end],
            'test': pairs[val_end:]
        }

        logger.info(f"Dataset split - Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

        # Process each split
        for split_name, split_pairs in splits.items():
            logger.info(f"Processing {split_name} set...")
            success_count = 0
            fail_count = 0

            for idx, sample in enumerate(tqdm(split_pairs, desc=f"Processing {split_name}")):
                if self.process_sample(sample, split_name, idx):
                    success_count += 1
                else:
                    fail_count += 1

            logger.info(f"{split_name} set: {success_count} successful, {fail_count} failed")
            self.dataset_stats['samples_processed'] += success_count
            self.dataset_stats['samples_failed'] += fail_count

    def save_dataset_info(self):
        """Save dataset information and statistics."""
        info = {
            'dataset': 'SUN RGB-D Depth Estimation',
            'image_size': self.img_size,
            'statistics': self.dataset_stats,
            'splits': {
                'train': len(list((self.output_path / 'train' / 'rgb').glob('*.png'))),
                'val': len(list((self.output_path / 'val' / 'rgb').glob('*.png'))),
                'test': len(list((self.output_path / 'test' / 'rgb').glob('*.png')))
            },
            'depth_range': {
                'min': float(self.dataset_stats['depth_min']),
                'max': float(self.dataset_stats['depth_max'])
            }
        }

        info_path = self.output_path / 'dataset_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))

        logger.info(f"Dataset info saved to {info_path}")

        # Also save as text
        text_path = self.output_path / 'dataset_info.txt'
        with open(text_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SUN RGB-D DEPTH ESTIMATION DATASET\n")
            f.write("="*80 + "\n\n")
            f.write(f"Image Size: {self.img_size[1]}x{self.img_size[0]}\n\n")
            f.write("Dataset Statistics:\n")
            f.write(f"  Total samples found: {self.dataset_stats['samples_found']}\n")
            f.write(f"  Successfully processed: {self.dataset_stats['samples_processed']}\n")
            f.write(f"  Failed: {self.dataset_stats['samples_failed']}\n\n")
            f.write("Dataset Splits:\n")
            f.write(f"  Train: {info['splits']['train']} samples\n")
            f.write(f"  Val:   {info['splits']['val']} samples\n")
            f.write(f"  Test:  {info['splits']['test']} samples\n\n")
            f.write("Depth Range:\n")
            f.write(f"  Min: {self.dataset_stats['depth_min']:.2f} m\n")
            f.write(f"  Max: {self.dataset_stats['depth_max']:.2f} m\n")

        logger.info(f"Dataset summary saved to {text_path}")

    def prepare_dataset(self):
        """Main function to prepare the dataset."""
        logger.info("Starting dataset preparation...")

        # Find RGB-Depth pairs
        pairs = self.find_rgb_depth_pairs()

        if len(pairs) == 0:
            logger.error("No RGB-Depth pairs found!")
            return

        # Split and process dataset
        self.split_dataset(pairs)

        # Save dataset information
        self.save_dataset_info()

        logger.info("Dataset preparation completed!")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Prepare SUN RGB-D dataset for depth estimation")
    parser.add_argument(
        '--sunrgbd_path',
        type=str,
        default='../../SUNRGBD',
        help='Path to SUN RGB-D dataset'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='./data_prepare',
        help='Output path for prepared dataset'
    )
    parser.add_argument(
        '--img_height',
        type=int,
        default=240,
        help='Target image height'
    )
    parser.add_argument(
        '--img_width',
        type=int,
        default=320,
        help='Target image width'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='Training set ratio'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.1,
        help='Validation set ratio'
    )
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.1,
        help='Test set ratio'
    )

    args = parser.parse_args()

    # Create preparer
    preparer = DepthDatasetPreparer(
        sunrgbd_path=args.sunrgbd_path,
        output_path=args.output_path,
        img_size=(args.img_height, args.img_width)
    )

    # Prepare dataset
    preparer.prepare_dataset()


if __name__ == "__main__":
    main()
