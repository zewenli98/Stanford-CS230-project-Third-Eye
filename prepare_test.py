import os
import sys
import json
import random
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import csv
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# HYPERPARAMETERS - Configure these settings
# ============================================================================
NUM_SAMPLES = 100  # Number of test samples to generate (excludes negative samples)
NEGATIVE_SAMPLE_RATIO = 0.2  # Ratio of negative samples (20% = true negatives without target objects)
ENTRY_FOLDER = None  # Specific folder to look into (e.g., "kv2/kinect2data") or None for random selection across all folders
SUNRGBD_ROOT = "./SUNRGBD"  # Root directory of SUNRGBD dataset
OUTPUT_DIR = "./queries"  # Output directory for generated data
OUTPUT_IMAGES_DIR = "./queries/images"  # Output directory for images
OUTPUT_CSV = "./queries/prompts.csv"  # Output CSV file path

# Goal objects to search for (only objects from this list will be selected as goals)
GOAL_OBJECTS = ["mug", "chair", "book", "cabinet", "door", "lamp"]

# ============================================================================


class SUNRGBDTestDataPreparer:
    """
    Prepare test data from SUNRGBD dataset.

    The SUNRGBD dataset structure:
    - Root directory contains entry-level folders: kv1, kv2, realsense, xtion
    - Each entry folder contains subdirectories with scene data
    - Each scene folder contains:
        - image/*.jpg: RGB image
        - depth/*.png: Depth image
        - scene.txt: Scene type
        - annotation/index.json: Object annotations
        - intrinsics.txt: Camera intrinsics (3x3 matrix)

    Object Matching Logic:
    - Uses case-insensitive partial matching for all goal objects
    - Examples:
        - "chair" matches: chair, Chair, CHAIR, woodchair, armchair, officechair, chairs
        - "mug" matches: mug, Mug, MUG, coffeemug
        - "book" matches: book, Book, BOOK, books, notebook, textbook
        - "cabinet" matches: cabinet, Cabinet, filecabinet, medicalcabinet
        - "door" matches: door, Door, glassdoor, wooddoor
        - "lamp" matches: lamp, Lamp, tablelamp, floorlamp
    - Preserves original name (e.g., "woodchair") while tracking matched category (e.g., "chair")
    """

    def __init__(self, sunrgbd_root: str = "./SUNRGBD"):
        """
        Initialize the test data preparer.

        Args:
            sunrgbd_root: Root directory of SUNRGBD dataset
        """
        self.sunrgbd_root = Path(sunrgbd_root)
        if not self.sunrgbd_root.exists():
            raise FileNotFoundError(f"SUNRGBD root directory not found: {sunrgbd_root}")

        # Entry-level folders in SUNRGBD
        self.entry_folders = ['kv1', 'kv2', 'realsense', 'xtion']
        logger.info(f"Initialized SUNRGBD test data preparer at {sunrgbd_root}")

    def get_all_scene_folders(self, entry_folder: Optional[str] = None) -> List[Path]:
        """
        Get all scene folders from specified entry folder or all entry folders.

        Args:
            entry_folder: Specific entry folder (e.g., "kv2/kinect2data"),
                         or None to search across all entry folders

        Returns:
            List of paths to scene folders
        """
        scene_folders = []

        if entry_folder:
            # Use specific entry folder
            folder_path = self.sunrgbd_root / entry_folder
            if not folder_path.exists():
                logger.warning(f"Entry folder not found: {entry_folder}")
                return []

            # Find all subdirectories that contain required files
            for scene_dir in folder_path.rglob("*"):
                if scene_dir.is_dir() and self._is_valid_scene_folder(scene_dir):
                    scene_folders.append(scene_dir)
        else:
            # Search across all entry folders
            for entry in self.entry_folders:
                entry_path = self.sunrgbd_root / entry
                if entry_path.exists():
                    for scene_dir in entry_path.rglob("*"):
                        if scene_dir.is_dir() and self._is_valid_scene_folder(scene_dir):
                            scene_folders.append(scene_dir)

        logger.info(f"Found {len(scene_folders)} valid scene folders")
        return scene_folders

    def _is_valid_scene_folder(self, scene_dir: Path) -> bool:
        """
        Check if a directory is a valid scene folder with required files.

        Args:
            scene_dir: Path to potential scene directory

        Returns:
            True if directory contains required files
        """
        required_items = [
            scene_dir / "scene.txt",
            scene_dir / "annotation" / "index.json",
            scene_dir / "image",
            scene_dir / "depth"
        ]
        return all(item.exists() for item in required_items)

    def load_scene_data(self, scene_folder: Path) -> Dict:
        """
        Load all data from a scene folder.

        Args:
            scene_folder: Path to scene folder

        Returns:
            Dictionary containing scene data:
                - scene_type: str
                - image_path: Path
                - image: PIL.Image
                - depth_path: Path
                - depth: numpy array
                - annotations: dict from index.json
                - intrinsics: 3x3 camera intrinsics matrix
        """
        # Read scene type
        scene_txt = scene_folder / "scene.txt"
        with open(scene_txt, 'r') as f:
            scene_type = f.read().strip()

        # Find image file
        image_dir = scene_folder / "image"
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        if not image_files:
            raise FileNotFoundError(f"No image found in {image_dir}")
        image_path = image_files[0]
        image = Image.open(image_path).convert('RGB')

        # Find depth file
        depth_dir = scene_folder / "depth"
        depth_files = list(depth_dir.glob("*.png"))
        if not depth_files:
            raise FileNotFoundError(f"No depth image found in {depth_dir}")
        depth_path = depth_files[0]
        depth = np.array(Image.open(depth_path))

        # Read annotations
        annotation_file = scene_folder / "annotation" / "index.json"
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)

        # Read camera intrinsics
        intrinsics_file = scene_folder / "intrinsics.txt"
        intrinsics = self._load_intrinsics(intrinsics_file)

        return {
            'scene_type': scene_type,
            'image_path': image_path,
            'image': image,
            'depth_path': depth_path,
            'depth': depth,
            'annotations': annotations,
            'intrinsics': intrinsics,
            'scene_folder': scene_folder
        }

    def _load_intrinsics(self, intrinsics_file: Path) -> np.ndarray:
        """
        Load camera intrinsics matrix from file.

        Args:
            intrinsics_file: Path to intrinsics.txt

        Returns:
            3x3 numpy array with camera intrinsics
        """
        intrinsics = []
        with open(intrinsics_file, 'r') as f:
            for line in f:
                values = [float(x) for x in line.strip().split()]
                intrinsics.append(values)
        return np.array(intrinsics)

    def select_closest_goal_object(self, scene_data: Dict, goal_objects: List[str] = None) -> Optional[Dict]:
        """
        Select the CLOSEST goal object from scene annotations.
        Only selects objects that match the goal_objects list (case-insensitive, partial matching).

        Args:
            scene_data: Scene data dictionary from load_scene_data
            goal_objects: List of object names to search for (case-insensitive, partial match)

        Returns:
            Dictionary containing:
                - name: object name (original, e.g., "woodchair")
                - matched_category: normalized category (e.g., "chair")
                - bbox: [x_min, y_min, x_max, y_max] bounding box
                - polygon: full polygon data
                - object_index: index in objects list
                - distance: distance in meters
            Or None if no valid objects found
        """
        annotations = scene_data['annotations']

        if 'objects' not in annotations or 'frames' not in annotations:
            logger.warning("Invalid annotation structure")
            return None

        objects = annotations['objects']
        if not objects:
            logger.warning("No objects found in annotations")
            return None

        # Get polygons from first frame
        if not annotations['frames'] or 'polygon' not in annotations['frames'][0]:
            logger.warning("No polygons found in annotations")
            return None

        polygons = annotations['frames'][0]['polygon']

        # Convert goal objects to lowercase for case-insensitive matching
        if goal_objects:
            goal_objects_lower = [obj.lower() for obj in goal_objects]
        else:
            goal_objects_lower = None

        # Filter valid polygons (with non-zero XYZ coordinates and matching goal objects)
        valid_objects = []
        for poly in polygons:
            if 'object' in poly and poly['object'] < len(objects):
                object_idx = poly['object']
                object_name = objects[object_idx].get('name', '').strip()

                # Skip if object name is empty
                if not object_name:
                    continue

                # Check if object is in goal objects list (case-insensitive partial matching)
                if goal_objects_lower:
                    # Partial matching: "chair" matches "chair", "Chair", "CHAIR", "woodchair", "armchair", etc.
                    # This handles:
                    # - Case variations: Chair, CHAIR, chair
                    # - Compound names: woodchair, officechair, armchair
                    # - Plural forms: chairs, Chairs, CHAIRS
                    object_name_lower = object_name.lower()
                    matched_category = None

                    for goal_obj in goal_objects_lower:
                        if goal_obj in object_name_lower:
                            matched_category = goal_obj
                            break

                    if not matched_category:
                        continue

                    # Check if polygon has valid XYZ data
                    if 'XYZ' in poly and poly['XYZ']:
                        # Check if XYZ has non-zero values
                        xyz_array = np.array(poly['XYZ'])
                        if xyz_array.size > 0 and not np.allclose(xyz_array, 0):
                            # Calculate distance (average of all XYZ points)
                            xyz_mean = xyz_array.mean(axis=0)
                            x_cam, y_cam, z_cam = xyz_mean
                            distance = np.sqrt(x_cam**2 + y_cam**2 + z_cam**2)

                            # Calculate bounding box from polygon coordinates
                            x_coords = poly['x']
                            y_coords = poly['y']
                            bbox = [
                                min(x_coords),  # x_min
                                min(y_coords),  # y_min
                                max(x_coords),  # x_max
                                max(y_coords)   # y_max
                            ]

                            valid_objects.append({
                                'name': object_name,  # Original name (e.g., "woodchair")
                                'matched_category': matched_category,  # Normalized (e.g., "chair")
                                'bbox': bbox,
                                'polygon': poly,
                                'object_index': object_idx,
                                'distance': distance
                            })

        if not valid_objects:
            if goal_objects:
                logger.warning(f"No valid objects found matching goal list: {goal_objects}")
            else:
                logger.warning("No valid polygons with XYZ data found")
            return None

        # Find the closest object
        closest_object = min(valid_objects, key=lambda obj: obj['distance'])

        logger.info(f"Found closest target object: {closest_object['name']} at {closest_object['distance']:.2f}m")
        if closest_object['name'].lower() != closest_object['matched_category']:
            logger.debug(f"  '{closest_object['name']}' → matched as '{closest_object['matched_category']}'")

        return closest_object

    def calculate_clock_direction(self, x_cam: float, z_cam: float) -> str:
        """
        Calculate clock-wise direction based on object's position relative to camera.
        Uses 2D plane (x-z) to determine angle.

        12 o'clock = straight ahead (z+)
        3 o'clock = right (x+)
        6 o'clock = behind (z-)
        9 o'clock = left (x-)

        Args:
            x_cam: X coordinate in camera frame (positive = right)
            z_cam: Z coordinate in camera frame (positive = forward)

        Returns:
            Clock direction string (e.g., "2 o'clock", "11 o'clock")
        """
        # Calculate angle from camera forward direction
        # atan2(x, z) gives angle from forward axis (z+)
        angle_rad = np.arctan2(x_cam, z_cam)
        angle_deg = np.degrees(angle_rad)

        # Convert to 0-360 range (0° = forward/12 o'clock, clockwise positive)
        angle_deg = (90 - angle_deg) % 360

        # Convert to clock hours (30° per hour)
        clock_hour = int(round(angle_deg / 30)) % 12
        if clock_hour == 0:
            clock_hour = 12

        return f"{clock_hour} o'clock"

    def calculate_object_position(self,
                                  goal_object: Dict,
                                  scene_data: Dict) -> Dict:
        """
        Calculate distance and clock-wise direction from camera to goal object.

        Uses the XYZ coordinates from annotations.

        Args:
            goal_object: Goal object dictionary from select_all_goal_objects
            scene_data: Scene data dictionary from load_scene_data

        Returns:
            Dictionary containing:
                - distance_meters: Average distance in meters
                - distance_feet: Distance in feet
                - direction: Clock-wise direction (e.g., "2 o'clock", "11 o'clock")
                - center_3d: [x, y, z] center position in camera coordinates
                - bbox_center_2d: [u, v] center in image coordinates
        """
        polygon = goal_object['polygon']
        xyz_coords = np.array(polygon['XYZ'])  # List of [x, y, z] points
        image_width, image_height = scene_data['image'].size

        # Calculate 3D center position
        center_3d = np.mean(xyz_coords, axis=0)  # [x, y, z]
        x_cam, y_cam, z_cam = center_3d

        # Calculate distance (depth is z-coordinate in camera frame)
        distance_meters = float(z_cam)
        distance_feet = distance_meters * 3.28084  # Convert to feet

        # Calculate 2D bbox center
        bbox = goal_object['bbox']
        bbox_center_u = (bbox[0] + bbox[2]) / 2
        bbox_center_v = (bbox[1] + bbox[3]) / 2

        # Calculate clock-wise direction
        direction = self.calculate_clock_direction(x_cam, z_cam)

        return {
            'distance_meters': distance_meters,
            'distance_feet': distance_feet,
            'direction': direction,
            'center_3d': center_3d.tolist(),
            'bbox_center_2d': [bbox_center_u, bbox_center_v]
        }

    def prepare_test_sample(self, entry_folder: Optional[str] = None, goal_objects: List[str] = None,
                           target_category: str = None) -> Optional[Dict]:
        """
        Prepare a complete test sample from SUNRGBD dataset with the CLOSEST target object.

        Args:
            entry_folder: Specific entry folder (e.g., "kv2/kinect2data"),
                         or None to randomly select across all folders
            goal_objects: List of object names to search for (case-insensitive)
            target_category: Preferred category to balance sampling (optional)

        Returns:
            Dictionary containing complete test sample:
                - scene_folder: Path to scene folder
                - scene_type: Scene description
                - image: PIL Image
                - image_path: Path to image file
                - object_name: Object name (original, e.g., "woodchair")
                - object_category: Object category (normalized, e.g., "chair")
                - object_bbox: Bounding box [x_min, y_min, x_max, y_max]
                - distance_meters: Distance in meters
                - distance_feet: Distance in feet
                - direction: Clock direction (e.g., "2 o'clock")
            Or None if no valid sample could be prepared
        """
        # Get all scene folders
        scene_folders = self.get_all_scene_folders(entry_folder)

        if not scene_folders:
            logger.error("No valid scene folders found")
            return None

        # Try up to 100 random scenes to find one with valid objects
        # Increased from 20 to handle category balancing better
        max_attempts = 100
        category_fallback_threshold = 50  # After 50 attempts, relax category requirement

        for attempt in range(max_attempts):
            # Randomly select a scene folder
            scene_folder = random.choice(scene_folders)
            logger.debug(f"Attempt {attempt + 1}: Selected scene folder: {scene_folder.name}")

            try:
                # Load scene data
                scene_data = self.load_scene_data(scene_folder)

                # Select the CLOSEST goal object
                closest_object = self.select_closest_goal_object(scene_data, goal_objects)

                if closest_object is None:
                    logger.debug(f"No valid objects in scene, trying another...")
                    continue

                # If target_category specified, prefer scenes with that category
                # But after many attempts, accept any valid scene
                if target_category and attempt < category_fallback_threshold:
                    if closest_object.get('matched_category', '').lower() != target_category.lower():
                        logger.debug(f"Scene doesn't contain preferred category '{target_category}', trying another...")
                        continue
                elif target_category and attempt >= category_fallback_threshold:
                    logger.info(f"Relaxing category requirement after {category_fallback_threshold} attempts")

                # Calculate detailed position for the closest object
                position = self.calculate_object_position(closest_object, scene_data)

                # Prepare final test sample
                test_sample = {
                    'scene_folder': str(scene_folder),
                    'scene_type': scene_data['scene_type'],
                    'image': scene_data['image'],
                    'image_path': str(scene_data['image_path']),
                    'object_name': closest_object['name'],  # Original name (e.g., "woodchair")
                    'object_category': closest_object.get('matched_category', closest_object['name'].lower()),  # Normalized (e.g., "chair")
                    'object_bbox': closest_object['bbox'],
                    'object_index': closest_object['object_index'],
                    'distance_meters': position['distance_meters'],
                    'distance_feet': position['distance_feet'],
                    'direction': position['direction'],
                    'center_3d': position['center_3d'],
                    'bbox_center_2d': position['bbox_center_2d'],
                    'full_annotation': scene_data['annotations']  # Include full annotation
                }

                logger.info(f"✓ Sample prepared: {closest_object['name']} at {position['distance_meters']:.2f}m, {position['direction']} in {scene_data['scene_type']}")
                if closest_object['name'].lower() != closest_object.get('matched_category', ''):
                    logger.debug(f"  '{closest_object['name']}' → matched as '{closest_object['matched_category']}'")

                return test_sample

            except Exception as e:
                logger.debug(f"Error processing scene {scene_folder.name}: {e}")
                continue

        logger.error(f"✗ Failed to prepare test sample after {max_attempts} attempts")
        logger.error(f"  Target category: {target_category}")
        logger.error(f"  Try: 1) Check if SUNRGBD dataset has these objects, 2) Reduce NUM_SAMPLES")
        return None

    def prepare_negative_sample(self, entry_folder: Optional[str] = None, goal_objects: List[str] = None) -> Optional[Dict]:
        """
        Prepare a negative test sample (image without target objects).

        Args:
            entry_folder: Specific entry folder or None for random selection
            goal_objects: List of object names (pick one as fake target)

        Returns:
            Dictionary with empty object list
        """
        scene_folders = self.get_all_scene_folders(entry_folder)
        if not scene_folders:
            return None

        max_attempts = 20
        for attempt in range(max_attempts):
            scene_folder = random.choice(scene_folders)

            try:
                scene_data = self.load_scene_data(scene_folder)

                # Check that scene has NO target objects
                closest_object = self.select_closest_goal_object(scene_data, goal_objects)

                # Only use scenes with NO target objects
                if closest_object is not None:
                    continue

                # Pick a random goal object name (fake target)
                fake_target = random.choice(goal_objects) if goal_objects else "object"

                # Create negative sample
                test_sample = {
                    'scene_folder': str(scene_folder),
                    'scene_type': scene_data['scene_type'],
                    'image': scene_data['image'],
                    'image_path': str(scene_data['image_path']),
                    'goal_objects': [],  # Empty - no objects found
                    'primary_object': fake_target,  # Fake target
                    'full_annotation': scene_data['annotations'],
                    'is_negative': True  # Flag as negative sample
                }

                logger.info(f"Successfully prepared negative sample (no {fake_target} found)")
                return test_sample

            except Exception as e:
                logger.warning(f"Error processing negative sample: {e}")
                continue

        logger.warning("Failed to prepare negative sample")
        return None

    def prepare_multiple_test_samples(self,
                                     n_samples: int = 10,
                                     entry_folder: Optional[str] = None,
                                     goal_objects: List[str] = None,
                                     negative_ratio: float = 0.2) -> List[Dict]:
        """
        Prepare multiple test samples with balanced categories and negative samples.

        Args:
            n_samples: Number of positive test samples to prepare
            entry_folder: Specific entry folder or None for random selection
            goal_objects: List of object names to search for (case-insensitive)
            negative_ratio: Ratio of negative samples (default 0.2 = 20%)

        Returns:
            List of test sample dictionaries (positive + negative)
        """
        # Calculate number of negative samples
        num_negatives = int(n_samples * negative_ratio)
        num_positives = n_samples

        logger.info(f"Target: {num_positives} positive samples + {num_negatives} negative samples")

        # Track category counts for balancing
        category_counts = {obj: 0 for obj in (goal_objects or [])}

        samples = []

        # Generate positive samples with balanced categories
        for i in range(num_positives):
            logger.info(f"\nPreparing positive sample {i + 1}/{num_positives}")

            # Find category with minimum count (for balancing)
            if goal_objects:
                target_category = min(category_counts, key=category_counts.get)
                logger.info(f"  Targeting category: {target_category} (current count: {category_counts[target_category]})")
            else:
                target_category = None

            sample = self.prepare_test_sample(entry_folder, goal_objects, target_category)

            if sample:
                samples.append(sample)
                # Update category count using normalized category
                object_category = sample.get('object_category', sample['object_name'].lower())
                if object_category in category_counts:
                    category_counts[object_category] += 1

        logger.info(f"\nSuccessfully prepared {len(samples)}/{num_positives} positive samples")
        logger.info(f"Category distribution: {category_counts}")

        # Generate negative samples
        for i in range(num_negatives):
            logger.info(f"\nPreparing negative sample {i + 1}/{num_negatives}")
            neg_sample = self.prepare_negative_sample(entry_folder, goal_objects)
            if neg_sample:
                samples.append(neg_sample)

        logger.info(f"\nTotal samples prepared: {len(samples)} ({len(samples)-num_negatives} positive + {len([s for s in samples if s.get('is_negative', False)])} negative)")
        return samples

    def save_test_sample(self, test_sample: Dict, output_dir: str = "./test_samples"):
        """
        Save a test sample to disk.

        Args:
            test_sample: Test sample dictionary
            output_dir: Output directory for saving files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate sample ID from scene folder
        scene_name = Path(test_sample['scene_folder']).name

        # Save image
        image_output = output_path / f"{scene_name}_image.jpg"
        test_sample['image'].save(image_output)

        # Save metadata
        metadata = {
            'scene_folder': test_sample['scene_folder'],
            'scene_type': test_sample['scene_type'],
            'image_path': test_sample['image_path'],
            'saved_image_path': str(image_output),
            'goal_object': test_sample['goal_object'],
            'position': test_sample['position']
        }

        metadata_output = output_path / f"{scene_name}_metadata.json"
        with open(metadata_output, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved test sample to {output_path}")
        logger.info(f"  Image: {image_output}")
        logger.info(f"  Metadata: {metadata_output}")

    def generate_queries_csv(self,
                            samples: List[Dict],
                            output_csv: str = "./queries/prompts.csv",
                            output_images_dir: str = "./queries/images"):
        """
        Generate queries in CSV format with images.

        Args:
            samples: List of test sample dictionaries
            output_csv: Path to output CSV file
            output_images_dir: Directory to save images

        CSV columns:
            - prompt_id: Unique identifier for each prompt
            - prompt_text: Query text in format "where is {object name}"
            - image_name: Filename of the RGB image
            - object: Object name (query target)
            - object_bbox: Bounding box [x_min, y_min, x_max, y_max] (JSON array) - empty for negatives
            - object_distance: Distance in meters (string) - empty for negatives
            - object_direction: Direction (e.g., "2 o'clock") - empty for negatives
            - scene: Scene type
            - annotation: Full annotation JSON as string
            - depth_image: Filename of the depth image
            - is_negative: Whether this is a negative sample (true/false)
        """
        # Create output directories
        output_csv_path = Path(output_csv)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        output_images_path = Path(output_images_dir)

        # Clear existing images in the directory
        if output_images_path.exists():
            logger.info(f"Clearing existing images from {output_images_dir}")
            for file in output_images_path.glob("*"):
                if file.is_file():
                    file.unlink()

        output_images_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating queries CSV with {len(samples)} samples")
        logger.info(f"  CSV output: {output_csv}")
        logger.info(f"  Images output: {output_images_dir}")

        # Prepare CSV data
        csv_rows = []

        for idx, sample in enumerate(samples, start=1):
            prompt_id = idx
            object_name = sample.get('object_name', sample.get('primary_object', 'object'))
            prompt_text = f"where is {object_name}"

            # Generate unique image filenames
            scene_name = Path(sample['scene_folder']).name
            image_filename = f"{prompt_id:04d}_{scene_name}.jpg"
            depth_image_filename = f"{prompt_id:04d}_{scene_name}_depth.png"

            # Copy RGB image
            rgb_image_path = output_images_path / image_filename
            sample['image'].save(rgb_image_path)

            # Copy depth image
            scene_folder = Path(sample['scene_folder'])
            depth_dir = scene_folder / "depth"
            depth_files = list(depth_dir.glob("*.png"))
            if depth_files:
                depth_image_path = output_images_path / depth_image_filename
                shutil.copy2(depth_files[0], depth_image_path)
            else:
                logger.warning(f"No depth image found for sample {prompt_id}")
                depth_image_filename = ""

            # Check if negative sample
            is_negative = sample.get('is_negative', False)

            # Prepare object data (empty for negative samples)
            if is_negative:
                bbox_json = "[]"
                distance_str = ""
                direction_str = ""
            else:
                bbox_json = json.dumps(sample.get('object_bbox', []))
                distance_str = f"{sample.get('distance_meters', 0):.2f}"
                direction_str = sample.get('direction', '')

            # Prepare row data
            row = {
                'prompt_id': prompt_id,
                'prompt_text': prompt_text,
                'image_name': image_filename,
                'object': object_name,
                'object_bbox': bbox_json,
                'object_distance': distance_str,
                'object_direction': direction_str,
                'scene': sample['scene_type'],
                'annotation': json.dumps(sample.get('full_annotation', {})),
                'depth_image': depth_image_filename,
                'is_negative': str(is_negative).lower()
            }
            csv_rows.append(row)

            if is_negative:
                logger.info(f"  [{prompt_id:04d}] {prompt_text} - NEGATIVE - {sample['scene_type']}")
            else:
                logger.info(f"  [{prompt_id:04d}] {prompt_text} - {distance_str}m, {direction_str} - {sample['scene_type']}")

        # Write CSV file
        fieldnames = [
            'prompt_id',
            'prompt_text',
            'image_name',
            'object',
            'object_bbox',
            'object_distance',
            'object_direction',
            'scene',
            'annotation',
            'depth_image',
            'is_negative'
        ]

        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        logger.info(f"\nSuccessfully generated queries CSV:")
        logger.info(f"  CSV file: {output_csv}")
        logger.info(f"  Images directory: {output_images_dir}")
        logger.info(f"  Total samples: {len(csv_rows)}")

    def append_negative_samples(self,
                                n_negatives: int = 10,
                                entry_folder: Optional[str] = None,
                                goal_objects: List[str] = None,
                                output_csv: str = "./queries/prompts.csv",
                                output_images_dir: str = "./queries/images"):
        """
        Generate negative samples and append them to existing CSV file.
        Does NOT regenerate positive samples - only adds new negatives.

        Args:
            n_negatives: Number of negative samples to generate
            entry_folder: Specific entry folder or None for random selection
            goal_objects: List of object names (pick one as fake target)
            output_csv: Path to existing CSV file
            output_images_dir: Directory to save images

        Returns:
            Number of negative samples successfully added
        """
        output_csv_path = Path(output_csv)
        output_images_path = Path(output_images_dir)

        # Check if CSV exists
        if not output_csv_path.exists():
            logger.error(f"CSV file not found: {output_csv}")
            logger.error("Please run the main script first to generate positive samples")
            return 0

        # Read existing CSV to get max prompt_id
        existing_rows = []
        max_prompt_id = 0

        with open(output_csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            fieldnames = reader.fieldnames
            for row in reader:
                existing_rows.append(row)
                max_prompt_id = max(max_prompt_id, int(row['prompt_id']))

        logger.info(f"Found {len(existing_rows)} existing samples in CSV")
        logger.info(f"Max prompt_id: {max_prompt_id}")
        logger.info(f"Generating {n_negatives} negative samples...")

        # Generate negative samples
        negative_samples = []
        for i in range(n_negatives):
            logger.info(f"\nPreparing negative sample {i + 1}/{n_negatives}")
            neg_sample = self.prepare_negative_sample(entry_folder, goal_objects)
            if neg_sample:
                negative_samples.append(neg_sample)

        if not negative_samples:
            logger.warning("No negative samples were generated")
            return 0

        logger.info(f"\nSuccessfully generated {len(negative_samples)} negative samples")

        # Prepare new CSV rows
        new_rows = []
        for idx, sample in enumerate(negative_samples, start=1):
            prompt_id = max_prompt_id + idx
            object_name = sample.get('primary_object', 'object')
            prompt_text = f"where is {object_name}"

            # Generate unique image filenames
            scene_name = Path(sample['scene_folder']).name
            image_filename = f"{prompt_id:04d}_{scene_name}.jpg"
            depth_image_filename = f"{prompt_id:04d}_{scene_name}_depth.png"

            # Copy RGB image
            rgb_image_path = output_images_path / image_filename
            sample['image'].save(rgb_image_path)

            # Copy depth image
            scene_folder = Path(sample['scene_folder'])
            depth_dir = scene_folder / "depth"
            depth_files = list(depth_dir.glob("*.png"))
            if depth_files:
                depth_image_path = output_images_path / depth_image_filename
                shutil.copy2(depth_files[0], depth_image_path)
            else:
                logger.warning(f"No depth image found for sample {prompt_id}")
                depth_image_filename = ""

            # Negative sample row (empty bbox, distance, direction)
            row = {
                'prompt_id': prompt_id,
                'prompt_text': prompt_text,
                'image_name': image_filename,
                'object': object_name,
                'object_bbox': "[]",
                'object_distance': "",
                'object_direction': "",
                'scene': sample['scene_type'],
                'annotation': json.dumps(sample.get('full_annotation', {})),
                'depth_image': depth_image_filename,
                'is_negative': 'true'
            }
            new_rows.append(row)
            logger.info(f"  [{prompt_id:04d}] {prompt_text} - NEGATIVE - {sample['scene_type']}")

        # Append to existing CSV
        with open(output_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerows(new_rows)

        logger.info(f"\n{'='*70}")
        logger.info(f"Successfully appended {len(new_rows)} negative samples to CSV")
        logger.info(f"  CSV file: {output_csv}")
        logger.info(f"  Images directory: {output_images_dir}")
        logger.info(f"  Total samples now: {len(existing_rows) + len(new_rows)}")
        logger.info(f"{'='*70}")

        return len(new_rows)


def main():
    """
    Main function to generate test queries using hyperparameters defined at the top of the file.

    Command-line usage:
        python prepare_test.py                  # Generate positive and negative samples (default)
        python prepare_test.py --add-negatives N  # Append N negative samples to existing CSV

    Uses the following hyperparameters:
        - NUM_SAMPLES: Number of test samples to generate
        - ENTRY_FOLDER: Specific folder to use (or None for random)
        - SUNRGBD_ROOT: Root directory of SUNRGBD dataset
        - OUTPUT_DIR: Output directory for queries
        - OUTPUT_IMAGES_DIR: Output directory for images
        - OUTPUT_CSV: Output CSV file path
    """
    # Check for command-line arguments
    if len(sys.argv) > 1 and sys.argv[1] == '--add-negatives':
        # Mode: Append negative samples only
        if len(sys.argv) < 3:
            logger.error("Usage: python prepare_test.py --add-negatives N")
            logger.error("  N = number of negative samples to add")
            return

        try:
            n_negatives = int(sys.argv[2])
        except ValueError:
            logger.error(f"Invalid number: {sys.argv[2]}")
            logger.error("Usage: python prepare_test.py --add-negatives N")
            return

        logger.info("="*70)
        logger.info("SUNRGBD Test Data Preparation - Append Negative Samples Only")
        logger.info("="*70)
        logger.info(f"Adding {n_negatives} negative samples to existing CSV")
        logger.info(f"  CSV file: {OUTPUT_CSV}")
        logger.info(f"  Images directory: {OUTPUT_IMAGES_DIR}")
        logger.info("="*70)

        # Initialize preparer
        preparer = SUNRGBDTestDataPreparer(sunrgbd_root=SUNRGBD_ROOT)

        # Append negative samples
        num_added = preparer.append_negative_samples(
            n_negatives=n_negatives,
            entry_folder=ENTRY_FOLDER,
            goal_objects=GOAL_OBJECTS,
            output_csv=OUTPUT_CSV,
            output_images_dir=OUTPUT_IMAGES_DIR
        )

        if num_added > 0:
            logger.info(f"\n✅ Successfully added {num_added} negative samples")
        else:
            logger.error(f"\n❌ Failed to add negative samples")

        return

    # Default mode: Generate positive and negative samples
    logger.info("="*70)
    logger.info("SUNRGBD Test Data Preparation")
    logger.info("="*70)
    logger.info(f"Hyperparameters:")
    logger.info(f"  NUM_SAMPLES: {NUM_SAMPLES}")
    logger.info(f"  ENTRY_FOLDER: {ENTRY_FOLDER or 'Random (across all folders)'}")
    logger.info(f"  GOAL_OBJECTS: {GOAL_OBJECTS}")
    logger.info(f"  SUNRGBD_ROOT: {SUNRGBD_ROOT}")
    logger.info(f"  OUTPUT_DIR: {OUTPUT_DIR}")
    logger.info(f"  OUTPUT_IMAGES_DIR: {OUTPUT_IMAGES_DIR}")
    logger.info(f"  OUTPUT_CSV: {OUTPUT_CSV}")
    logger.info("="*70)

    # Initialize preparer
    preparer = SUNRGBDTestDataPreparer(sunrgbd_root=SUNRGBD_ROOT)

    # Prepare test samples based on hyperparameters
    logger.info(f"\nPreparing {NUM_SAMPLES} test samples...")
    samples = preparer.prepare_multiple_test_samples(
        n_samples=NUM_SAMPLES,
        entry_folder=ENTRY_FOLDER,
        goal_objects=GOAL_OBJECTS,
        negative_ratio=NEGATIVE_SAMPLE_RATIO
    )

    if not samples:
        logger.error("No samples were generated. Please check your dataset and try again.")
        return

    # Generate queries CSV with all data
    logger.info(f"\nGenerating queries CSV...")
    preparer.generate_queries_csv(
        samples=samples,
        output_csv=OUTPUT_CSV,
        output_images_dir=OUTPUT_IMAGES_DIR
    )

    logger.info("\n" + "="*70)
    logger.info("Done! Output files:")
    logger.info(f"  CSV: {OUTPUT_CSV}")
    logger.info(f"  Images: {OUTPUT_IMAGES_DIR}")
    logger.info(f"  Total samples generated: {len(samples)}/{NUM_SAMPLES}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
