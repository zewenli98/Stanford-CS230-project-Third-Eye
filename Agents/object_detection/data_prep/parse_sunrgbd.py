"""
SUN RGB-D Dataset Parser - Filtered for 6 Target Classes
Converts SUN RGB-D annotations to YOLO format for object detection training.
Only includes: Mug, Book, Cabinet, Door, Chair, Lamp
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Set
import logging
from tqdm import tqdm
import shutil
import scipy.io as sio
from PIL import Image
import cv2
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Target classes - case insensitive matching
TARGET_CLASSES = {
    'mug': 'mug',
    'book': 'book',
    'cabinet': 'cabinet',
    'door': 'door',
    'chair': 'chair',
    'lamp': 'lamp'
}

# Variations to normalize (lowercase keys)
CLASS_VARIATIONS = {
    # Mug variations
    'mug': 'mug',
    'Mug': 'mug',
    'MUG': 'mug',
    'cup': 'mug',  # Cup can be considered as mug
    'Cup': 'mug',
    'CUP': 'mug',

    # Book variations
    'book': 'book',
    'Book': 'book',
    'BOOK': 'book',
    'books': 'book',
    'Books': 'book',
    'BOOKS': 'book',

    # Cabinet variations
    'cabinet': 'cabinet',
    'Cabinet': 'cabinet',
    'CABINET': 'cabinet',
    'cabinets': 'cabinet',
    'Cabinets': 'cabinet',
    'cupboard': 'cabinet',
    'Cupboard': 'cabinet',
    'cubboard': 'cabinet',

    # Door variations
    'door': 'door',
    'Door': 'door',
    'DOOR': 'door',
    'glassdoor': 'door',
    'cabinetdoor': 'door',

    # Chair variations
    'chair': 'chair',
    'Chair': 'chair',
    'CHAIR': 'chair',
    'chairs': 'chair',
    'Chairs': 'chair',
    'officechair': 'chair',
    'armchair': 'chair',
    'Armchair': 'chair',

    # Lamp variations
    'lamp': 'lamp',
    'Lamp': 'lamp',
    'LAMP': 'lamp',
    'tablelamp': 'lamp',
}


class SUNRGBDParser:
    """Parser for SUN RGB-D dataset to YOLO format - Filtered for 6 target classes."""

    def __init__(self, sunrgbd_path: str, output_path: str):
        """
        Initialize the parser.

        Args:
            sunrgbd_path: Path to SUNRGBD dataset root
            output_path: Path to output YOLO format dataset
        """
        self.sunrgbd_path = Path(sunrgbd_path)
        self.output_path = Path(output_path)

        # Only use target classes
        self.target_classes = TARGET_CLASSES
        self.class_variations = CLASS_VARIATIONS

        # Class to index mapping (0-indexed for YOLO)
        self.class_to_idx = {
            'mug': 0,
            'book': 1,
            'cabinet': 2,
            'door': 3,
            'chair': 4,
            'lamp': 5
        }

        # Track statistics
        self.class_counts = Counter()
        self.skipped_classes = Counter()

        # Create output directories
        self.images_path = self.output_path / "images"
        self.labels_path = self.output_path / "labels"

        for split in ['train', 'val', 'test']:
            (self.images_path / split).mkdir(parents=True, exist_ok=True)
            (self.labels_path / split).mkdir(parents=True, exist_ok=True)

    def normalize_class_name(self, class_name: str) -> str:
        """
        Normalize class name to one of the 6 target classes.

        Args:
            class_name: Original class name from annotation

        Returns:
            Normalized class name or None if not a target class
        """
        # Exact match in variations dictionary
        if class_name in self.class_variations:
            return self.class_variations[class_name]

        # Special case: ANY object containing "chair" (case insensitive) maps to 'chair'
        class_lower = class_name.lower()
        if 'chair' in class_lower:
            return 'chair'

        # Case-insensitive partial matching for other target classes
        for target in self.target_classes.keys():
            if target in class_lower:
                return target

        # Not a target class
        self.skipped_classes[class_name] += 1
        return None

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
        Parse annotation JSON file and filter for target classes.

        Args:
            json_path: Path to annotation JSON file

        Returns:
            List of annotation dictionaries (only target classes)
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
                                original_name = data['objects'][obj_idx]['name']

                                # Normalize to target class
                                normalized_name = self.normalize_class_name(original_name)

                                if normalized_name:
                                    x_coords = poly['x']
                                    y_coords = poly['y']

                                    # Convert polygon to bbox
                                    bbox = self.polygon_to_bbox(x_coords, y_coords)

                                    # Validate bbox
                                    if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                                        annotations.append({
                                            'class_name': normalized_name,
                                            'bbox': bbox,
                                            'original_name': original_name
                                        })

                                        self.class_counts[normalized_name] += 1

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
        logger.info("Scanning dataset for samples...")
        for root, dirs, files in tqdm(list(os.walk(self.sunrgbd_path))):
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

        logger.info(f"Found {len(samples)} total samples in dataset")
        return samples

    def filter_samples_with_target_classes(self, samples: List[Dict]) -> List[Dict]:
        """
        Include ALL samples - both with target objects and pure background images.
        Background images help the model learn what is NOT a target object.

        Args:
            samples: List of all samples

        Returns:
            List of all valid samples (including background images)
        """
        filtered_samples = []
        background_count = 0

        logger.info("Processing samples (including background images)...")
        for sample in tqdm(samples, desc="Filtering samples"):
            annotations = self.parse_annotation_json(sample['annotation_path'])

            # Include ALL samples - both with and without target objects
            filtered_samples.append(sample)

            # Track background images (no target objects)
            if len(annotations) == 0:
                background_count += 1

        logger.info(f"Total samples: {len(filtered_samples)}")
        logger.info(f"  - With target objects: {len(filtered_samples) - background_count}")
        logger.info(f"  - Background images (no targets): {background_count}")
        return filtered_samples

    def process_sample(self, sample: Dict, split: str) -> bool:
        """
        Process a single sample and save in YOLO format.
        Includes background images (empty label files) for better training.

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

            # Parse annotations (only target classes)
            annotations = self.parse_annotation_json(sample['annotation_path'])

            # Create unique sample name
            sample_name = f"{sample['sample_id']}"

            # Copy image
            image_output = self.images_path / split / f"{sample_name}.jpg"

            # Convert to JPG if needed
            if sample['image_path'].suffix.lower() == '.png':
                img_rgb = img.convert('RGB')
                img_rgb.save(image_output, 'JPEG')
            else:
                shutil.copy(sample['image_path'], image_output)

            # Create YOLO format label file
            # Empty file for background images (no annotations)
            label_output = self.labels_path / split / f"{sample_name}.txt"
            with open(label_output, 'w') as f:
                for ann in annotations:
                    class_idx = self.class_to_idx[ann['class_name']]
                    yolo_bbox = self.bbox_to_yolo(ann['bbox'], img_width, img_height)

                    # YOLO format: class_id x_center y_center width height
                    f.write(f"{class_idx} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")

            return True

        except Exception as e:
            logger.error(f"Error processing sample {sample['sample_id']}: {e}")
            return False

    def split_dataset(self, samples: List[Dict], train_ratio: float = 0.8,
                      val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42):
        """
        Split dataset into train/val/test sets.

        Args:
            samples: List of filtered samples
            train_ratio: Ratio of training samples
            val_ratio: Ratio of validation samples
            test_ratio: Ratio of test samples
            seed: Random seed for reproducibility
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

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
        yaml_content = f"""# SUN RGB-D Dataset Configuration for YOLOv8 - 6 Target Classes
# Only includes: Mug, Book, Cabinet, Door, Chair, Lamp

path: {self.output_path.absolute()}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
test: images/test  # test images (relative to 'path')

# Classes (case-normalized)
nc: 6  # number of classes
names:
  0: mug
  1: book
  2: cabinet
  3: door
  4: chair
  5: lamp
"""

        yaml_path = self.output_path / "sunrgbd_6classes.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)

        logger.info(f"Dataset configuration saved to {yaml_path}")

    def save_class_statistics(self):
        """Save statistics about class distribution."""
        stats_file = self.output_path / "class_statistics.txt"

        with open(stats_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SUN RGB-D FILTERED DATASET - CLASS STATISTICS\n")
            f.write("=" * 80 + "\n\n")

            f.write("TARGET CLASSES (6 classes):\n")
            f.write("-" * 80 + "\n")
            total_instances = sum(self.class_counts.values())

            for class_name in sorted(self.class_to_idx.keys()):
                count = self.class_counts[class_name]
                percentage = (count / total_instances * 100) if total_instances > 0 else 0
                f.write(f"{class_name:<15} {count:>6} instances ({percentage:>5.2f}%)\n")

            f.write("-" * 80 + "\n")
            f.write(f"{'TOTAL':<15} {total_instances:>6} instances\n\n")

            # Top skipped classes
            f.write("=" * 80 + "\n")
            f.write("TOP 50 SKIPPED CLASSES (not in target list):\n")
            f.write("=" * 80 + "\n")
            for class_name, count in self.skipped_classes.most_common(50):
                f.write(f"{class_name:<40} {count:>6} occurrences\n")

        logger.info(f"Class statistics saved to {stats_file}")

    def convert(self):
        """
        Main conversion pipeline for 6 target classes.
        Includes background images (images without target objects) for better training.
        """
        logger.info("=" * 80)
        logger.info("Starting SUN RGB-D to YOLO conversion (6 Target Classes)")
        logger.info("Target classes: Mug, Book, Cabinet, Door, Chair, Lamp")
        logger.info("Includes background images for robust training")
        logger.info("=" * 80)

        # Find all samples
        all_samples = self.find_all_samples()

        # Process all samples (including background images)
        filtered_samples = self.filter_samples_with_target_classes(all_samples)

        if len(filtered_samples) == 0:
            logger.error("No samples found!")
            return

        # Print class distribution
        logger.info("\nClass distribution (target objects only):")
        for class_name in sorted(self.class_to_idx.keys()):
            count = self.class_counts[class_name]
            logger.info(f"  {class_name}: {count} instances")

        total_target_instances = sum(self.class_counts.values())
        logger.info(f"\nTotal target object instances: {total_target_instances}")

        # Split and process dataset
        logger.info("\nSplitting and processing dataset...")
        self.split_dataset(filtered_samples)

        # Create YAML config
        logger.info("\nCreating YAML configuration...")
        self.create_yaml_config()

        # Save statistics
        logger.info("\nSaving class statistics...")
        self.save_class_statistics()

        logger.info("\n" + "=" * 80)
        logger.info("Conversion completed successfully!")
        logger.info(f"Total samples: {len(filtered_samples)}")
        logger.info(f"Total target object instances: {total_target_instances}")
        logger.info("=" * 80)


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert SUN RGB-D dataset to YOLO format (6 target classes only)"
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
        default='./Agents/object_detection/data/sunrgbd_6classes',
        help='Output path for YOLO format dataset'
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
    parser_obj.convert()


if __name__ == "__main__":
    main()
