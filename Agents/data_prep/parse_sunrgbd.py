"""
SUN RGB-D Dataset Parser
Converts SUN RGB-D annotations to YOLO format for object detection training.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm
import shutil
import scipy.io as sio
from PIL import Image
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SUNRGBDParser:
    """Parser for SUN RGB-D dataset to YOLO format."""

    def __init__(self, sunrgbd_path: str, output_path: str):
        """
        Initialize the parser.

        Args:
            sunrgbd_path: Path to SUNRGBD dataset root
            output_path: Path to output YOLO format dataset
        """
        self.sunrgbd_path = Path(sunrgbd_path)
        self.output_path = Path(output_path)
        self.class_names = set()
        self.class_to_idx = {}

        # Create output directories
        self.images_path = self.output_path / "images"
        self.labels_path = self.output_path / "labels"

        for split in ['train', 'val', 'test']:
            (self.images_path / split).mkdir(parents=True, exist_ok=True)
            (self.labels_path / split).mkdir(parents=True, exist_ok=True)

    def polygon_to_bbox(self, x_coords: List[int], y_coords: List[int]) -> Tuple[int, int, int, int]:
        """
        Convert polygon coordinates to bounding box.

        Args:
            x_coords: List of x coordinates
            y_coords: List of y coordinates

        Returns:
            Tuple of (x_min, y_min, x_max, y_max)
        """
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)
        return x_min, y_min, x_max, y_max

    def bbox_to_yolo(self, bbox: Tuple[int, int, int, int], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """
        Convert bounding box to YOLO format (normalized center x, center y, width, height).

        Args:
            bbox: Tuple of (x_min, y_min, x_max, y_max)
            img_width: Image width
            img_height: Image height

        Returns:
            Tuple of (x_center, y_center, width, height) normalized to [0, 1]
        """
        x_min, y_min, x_max, y_max = bbox

        # Calculate center and dimensions
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min

        # Normalize to [0, 1]
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height

        # Clip to valid range
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))

        return x_center, y_center, width, height

    def parse_annotation_json(self, json_path: Path) -> List[Dict]:
        """
        Parse annotation JSON file.

        Args:
            json_path: Path to annotation JSON file

        Returns:
            List of annotation dictionaries
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            annotations = []
            if 'frames' in data and len(data['frames']) > 0:
                frame = data['frames'][0]
                if 'polygon' in frame:
                    for poly in frame['polygon']:
                        if 'object' in poly:
                            obj_idx = poly['object']
                            if obj_idx < len(data['objects']):
                                obj_name = data['objects'][obj_idx]['name']
                                x_coords = poly['x']
                                y_coords = poly['y']

                                # Convert polygon to bbox
                                bbox = self.polygon_to_bbox(x_coords, y_coords)

                                annotations.append({
                                    'class_name': obj_name,
                                    'bbox': bbox
                                })

                                # Track all class names
                                self.class_names.add(obj_name)

            return annotations
        except Exception as e:
            logger.error(f"Error parsing {json_path}: {e}")
            return []

    def find_all_samples(self) -> List[Dict]:
        """
        Find all samples in the dataset.

        Returns:
            List of sample dictionaries with paths
        """
        samples = []

        # Traverse the dataset directory
        for root, dirs, files in os.walk(self.sunrgbd_path):
            root_path = Path(root)

            # Check if this directory has annotations
            annotation_json = root_path / "annotation2Dfinal" / "index.json"

            if annotation_json.exists():
                # Find corresponding image
                image_dir = root_path / "image"
                if image_dir.exists():
                    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

                    if image_files:
                        samples.append({
                            'image_path': image_files[0],
                            'annotation_path': annotation_json,
                            'sample_id': root_path.name
                        })

        logger.info(f"Found {len(samples)} samples in dataset")
        return samples

    def create_class_mapping(self, min_samples: int = 10):
        """
        Create class name to index mapping, filtering rare classes.

        Args:
            min_samples: Minimum number of samples for a class to be included
        """
        # Count class occurrences
        class_counts = {name: 0 for name in self.class_names}

        samples = self.find_all_samples()
        for sample in samples:
            annotations = self.parse_annotation_json(sample['annotation_path'])
            for ann in annotations:
                if ann['class_name'] in class_counts:
                    class_counts[ann['class_name']] += 1

        # Filter classes by minimum samples
        filtered_classes = sorted([
            name for name, count in class_counts.items()
            if count >= min_samples
        ])

        self.class_to_idx = {name: idx for idx, name in enumerate(filtered_classes)}

        logger.info(f"Total classes found: {len(self.class_names)}")
        logger.info(f"Classes after filtering (min {min_samples} samples): {len(filtered_classes)}")

        # Save class names
        class_file = self.output_path / "classes.txt"
        with open(class_file, 'w') as f:
            for class_name in filtered_classes:
                f.write(f"{class_name}\n")

        logger.info(f"Class names saved to {class_file}")

    def process_sample(self, sample: Dict, split: str) -> bool:
        """
        Process a single sample and save in YOLO format.

        Args:
            sample: Sample dictionary
            split: Data split (train/val/test)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Read image to get dimensions
            img = Image.open(sample['image_path'])
            img_width, img_height = img.size

            # Parse annotations
            annotations = self.parse_annotation_json(sample['annotation_path'])

            # Filter annotations to only include known classes
            valid_annotations = []
            for ann in annotations:
                if ann['class_name'] in self.class_to_idx:
                    valid_annotations.append(ann)

            # Skip samples with no valid annotations
            if len(valid_annotations) == 0:
                return False

            # Create unique sample name
            sample_name = f"{sample['sample_id']}"

            # Copy image
            image_output = self.images_path / split / f"{sample_name}.jpg"
            shutil.copy(sample['image_path'], image_output)

            # Create YOLO format label file
            label_output = self.labels_path / split / f"{sample_name}.txt"
            with open(label_output, 'w') as f:
                for ann in valid_annotations:
                    class_idx = self.class_to_idx[ann['class_name']]
                    yolo_bbox = self.bbox_to_yolo(ann['bbox'], img_width, img_height)

                    # YOLO format: class_id x_center y_center width height
                    f.write(f"{class_idx} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")

            return True

        except Exception as e:
            logger.error(f"Error processing sample {sample['sample_id']}: {e}")
            return False

    def split_dataset(self, train_ratio: float = 0.8, val_ratio: float = 0.1,
                      test_ratio: float = 0.1, seed: int = 42):
        """
        Split dataset into train/val/test sets.

        Args:
            train_ratio: Ratio of training samples
            val_ratio: Ratio of validation samples
            test_ratio: Ratio of test samples
            seed: Random seed for reproducibility
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

        # Get all samples
        samples = self.find_all_samples()

        # Shuffle samples
        np.random.seed(seed)
        np.random.shuffle(samples)

        # Calculate split indices
        n_samples = len(samples)
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)

        splits = {
            'train': samples[:train_end],
            'val': samples[train_end:val_end],
            'test': samples[val_end:]
        }

        logger.info(f"Dataset split - Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

        # Process each split
        for split_name, split_samples in splits.items():
            logger.info(f"Processing {split_name} set...")
            success_count = 0

            for sample in tqdm(split_samples, desc=f"Processing {split_name}"):
                if self.process_sample(sample, split_name):
                    success_count += 1

            logger.info(f"{split_name} set: {success_count}/{len(split_samples)} samples processed successfully")

    def create_yaml_config(self):
        """Create YOLO dataset configuration YAML file."""
        yaml_content = f"""# SUN RGB-D Dataset Configuration for YOLOv8
path: {self.output_path.absolute()}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
test: images/test  # test images (relative to 'path')

# Classes
nc: {len(self.class_to_idx)}  # number of classes
names: {list(self.class_to_idx.keys())}  # class names
"""

        yaml_path = self.output_path / "sunrgbd.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)

        logger.info(f"Dataset configuration saved to {yaml_path}")

    def convert(self, min_samples_per_class: int = 10):
        """
        Main conversion pipeline.

        Args:
            min_samples_per_class: Minimum samples required per class
        """
        logger.info("Starting SUN RGB-D to YOLO conversion...")

        # Find all samples first to collect class names
        logger.info("Scanning dataset for class names...")
        all_samples = self.find_all_samples()
        for sample in all_samples:
            self.parse_annotation_json(sample['annotation_path'])

        # Create class mapping
        logger.info("Creating class mapping...")
        self.create_class_mapping(min_samples=min_samples_per_class)

        # Split and process dataset
        logger.info("Splitting and processing dataset...")
        self.split_dataset()

        # Create YAML config
        logger.info("Creating YAML configuration...")
        self.create_yaml_config()

        logger.info("Conversion completed successfully!")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Convert SUN RGB-D dataset to YOLO format")
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
        help='Output path for YOLO format dataset'
    )
    parser.add_argument(
        '--min_samples',
        type=int,
        default=10,
        help='Minimum samples per class to include'
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
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Create parser
    parser_obj = SUNRGBDParser(args.sunrgbd_path, args.output_path)

    # Run conversion
    parser_obj.convert(min_samples_per_class=args.min_samples)


if __name__ == "__main__":
    main()
