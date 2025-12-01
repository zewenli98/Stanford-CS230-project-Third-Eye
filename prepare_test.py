import os
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
NUM_SAMPLES = 10  # Number of test samples to generate
ENTRY_FOLDER = "kv2/kinect2data"  # Specific folder to look into (e.g., "kv2/kinect2data") or None for random selection across all folders
SUNRGBD_ROOT = "./SUNRGBD"  # Root directory of SUNRGBD dataset
OUTPUT_DIR = "./queries"  # Output directory for generated data
OUTPUT_IMAGES_DIR = "./queries/images"  # Output directory for images
OUTPUT_CSV = "./queries/prompts.csv"  # Output CSV file path

# Goal objects to search for (only objects from this list will be selected as goals)
GOAL_OBJECTS = ["cup", "chair", "microwave", "bookcase"]
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

    def select_random_goal_object(self, scene_data: Dict, goal_objects: List[str] = None) -> Optional[Dict]:
        """
        Randomly select a goal object from scene annotations.
        Only selects objects that match the goal_objects list.

        Args:
            scene_data: Scene data dictionary from load_scene_data
            goal_objects: List of object names to search for (case-insensitive)

        Returns:
            Dictionary containing:
                - name: object name
                - bbox: [x_min, y_min, x_max, y_max] bounding box
                - polygon: full polygon data
                - object_index: index in objects list
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
        valid_polygons = []
        for poly in polygons:
            if 'object' in poly and poly['object'] < len(objects):
                object_idx = poly['object']
                object_name = objects[object_idx].get('name', '').strip()

                # Skip if object name is empty
                if not object_name:
                    continue

                # Check if object is in goal objects list (case-insensitive)
                if goal_objects_lower:
                    # Check for partial matches (e.g., "chair" matches "armchair", "officechair")
                    object_name_lower = object_name.lower()
                    is_goal_object = any(goal_obj in object_name_lower for goal_obj in goal_objects_lower)

                    if not is_goal_object:
                        continue

                # Check if polygon has valid XYZ data
                if 'XYZ' in poly and poly['XYZ']:
                    # Check if XYZ has non-zero values
                    xyz_array = np.array(poly['XYZ'])
                    if xyz_array.size > 0 and not np.allclose(xyz_array, 0):
                        valid_polygons.append(poly)

        if not valid_polygons:
            if goal_objects:
                logger.warning(f"No valid objects found matching goal list: {goal_objects}")
            else:
                logger.warning("No valid polygons with XYZ data found")
            return None

        # Randomly select one polygon
        selected_polygon = random.choice(valid_polygons)
        object_idx = selected_polygon['object']
        object_name = objects[object_idx]['name']

        # Calculate bounding box from polygon coordinates
        x_coords = selected_polygon['x']
        y_coords = selected_polygon['y']
        bbox = [
            min(x_coords),  # x_min
            min(y_coords),  # y_min
            max(x_coords),  # x_max
            max(y_coords)   # y_max
        ]

        return {
            'name': object_name,
            'bbox': bbox,
            'polygon': selected_polygon,
            'object_index': object_idx
        }

    def calculate_object_position(self,
                                  goal_object: Dict,
                                  scene_data: Dict) -> Dict:
        """
        Calculate distance and direction from camera to goal object.

        Uses the XYZ coordinates from annotations and depth image for verification.

        Args:
            goal_object: Goal object dictionary from select_random_goal_object
            scene_data: Scene data dictionary from load_scene_data

        Returns:
            Dictionary containing:
                - distance_meters: Average distance in meters
                - distance_feet: Distance in feet
                - direction: Direction description (left/right/center, up/down/middle)
                - center_3d: [x, y, z] center position in camera coordinates
                - bbox_center_2d: [u, v] center in image coordinates
        """
        polygon = goal_object['polygon']
        xyz_coords = np.array(polygon['XYZ'])  # List of [x, y, z] points
        depth = scene_data['depth']
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

        # Determine horizontal direction (left/right/center)
        horizontal_ratio = bbox_center_u / image_width
        if horizontal_ratio < 0.33:
            horizontal_dir = "left"
        elif horizontal_ratio > 0.67:
            horizontal_dir = "right"
        else:
            horizontal_dir = "center"

        # Determine vertical direction (up/down/middle)
        vertical_ratio = bbox_center_v / image_height
        if vertical_ratio < 0.33:
            vertical_dir = "above"
        elif vertical_ratio > 0.67:
            vertical_dir = "below"
        else:
            vertical_dir = "middle"

        # Combine direction
        if vertical_dir == "middle":
            direction = horizontal_dir
        else:
            direction = f"{vertical_dir} and {horizontal_dir}"

        # Also use x_cam for more precise left/right determination
        if x_cam < -0.5:
            direction += " (camera-left)"
        elif x_cam > 0.5:
            direction += " (camera-right)"

        return {
            'distance_meters': distance_meters,
            'distance_feet': distance_feet,
            'direction': direction,
            'center_3d': center_3d.tolist(),
            'bbox_center_2d': [bbox_center_u, bbox_center_v],
            'horizontal_direction': horizontal_dir,
            'vertical_direction': vertical_dir
        }

    def prepare_test_sample(self, entry_folder: Optional[str] = None, goal_objects: List[str] = None) -> Optional[Dict]:
        """
        Prepare a complete test sample from SUNRGBD dataset.

        Args:
            entry_folder: Specific entry folder (e.g., "kv2/kinect2data"),
                         or None to randomly select across all folders
            goal_objects: List of object names to search for (case-insensitive)

        Returns:
            Dictionary containing complete test sample:
                - scene_folder: Path to scene folder
                - scene_type: Scene description
                - image: PIL Image
                - image_path: Path to image file
                - goal_object: Object information (name, bbox)
                - position: Distance and direction information
            Or None if no valid sample could be prepared
        """
        # Get all scene folders
        scene_folders = self.get_all_scene_folders(entry_folder)

        if not scene_folders:
            logger.error("No valid scene folders found")
            return None

        # Try up to 10 random scenes to find one with valid objects
        max_attempts = 10
        for attempt in range(max_attempts):
            # Randomly select a scene folder
            scene_folder = random.choice(scene_folders)
            logger.info(f"Attempt {attempt + 1}: Selected scene folder: {scene_folder}")

            try:
                # Load scene data
                scene_data = self.load_scene_data(scene_folder)

                # Select random goal object
                goal_object = self.select_random_goal_object(scene_data, goal_objects)

                if goal_object is None:
                    logger.warning(f"No valid objects in scene, trying another...")
                    continue

                # Calculate object position
                position = self.calculate_object_position(goal_object, scene_data)

                # Prepare final test sample
                test_sample = {
                    'scene_folder': str(scene_folder),
                    'scene_type': scene_data['scene_type'],
                    'image': scene_data['image'],
                    'image_path': str(scene_data['image_path']),
                    'goal_object': {
                        'name': goal_object['name'],
                        'bbox': goal_object['bbox'],
                        'object_index': goal_object['object_index']
                    },
                    'position': position,
                    'full_annotation': scene_data['annotations']  # Include full annotation
                }

                logger.info(f"Successfully prepared test sample:")
                logger.info(f"  Scene: {scene_data['scene_type']}")
                logger.info(f"  Goal object: {goal_object['name']}")
                logger.info(f"  Distance: {position['distance_meters']:.2f}m ({position['distance_feet']:.2f}ft)")
                logger.info(f"  Direction: {position['direction']}")

                return test_sample

            except Exception as e:
                logger.warning(f"Error processing scene {scene_folder}: {e}")
                continue

        logger.error(f"Failed to prepare test sample after {max_attempts} attempts")
        return None

    def prepare_multiple_test_samples(self,
                                     n_samples: int = 10,
                                     entry_folder: Optional[str] = None,
                                     goal_objects: List[str] = None) -> List[Dict]:
        """
        Prepare multiple test samples.

        Args:
            n_samples: Number of test samples to prepare
            entry_folder: Specific entry folder or None for random selection
            goal_objects: List of object names to search for (case-insensitive)

        Returns:
            List of test sample dictionaries
        """
        samples = []
        for i in range(n_samples):
            logger.info(f"\nPreparing test sample {i + 1}/{n_samples}")
            sample = self.prepare_test_sample(entry_folder, goal_objects)
            if sample:
                samples.append(sample)

        logger.info(f"\nSuccessfully prepared {len(samples)}/{n_samples} test samples")
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
            - object: Object name
            - object_distance: Distance of object to camera (in meters)
            - object_direction: Direction of object relative to camera
            - scene: Scene type
            - object_bbox: Bounding box in format [x_min, y_min, x_max, y_max]
            - annotation: Full annotation JSON as string
            - depth_image: Filename of the depth image
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
            object_name = sample['goal_object']['name']
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

            # Prepare row data
            row = {
                'prompt_id': prompt_id,
                'prompt_text': prompt_text,
                'image_name': image_filename,
                'object': object_name,
                'object_distance': f"{sample['position']['distance_meters']:.2f}",
                'object_direction': sample['position']['direction'],
                'scene': sample['scene_type'],
                'object_bbox': json.dumps(sample['goal_object']['bbox']),
                'annotation': json.dumps(sample.get('full_annotation', {})),
                'depth_image': depth_image_filename
            }
            csv_rows.append(row)

            logger.info(f"  [{prompt_id:04d}] {prompt_text} - {sample['scene_type']}")

        # Write CSV file
        fieldnames = [
            'prompt_id',
            'prompt_text',
            'image_name',
            'object',
            'object_distance',
            'object_direction',
            'scene',
            'object_bbox',
            'annotation',
            'depth_image'
        ]

        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        logger.info(f"\nSuccessfully generated queries CSV:")
        logger.info(f"  CSV file: {output_csv}")
        logger.info(f"  Images directory: {output_images_dir}")
        logger.info(f"  Total samples: {len(csv_rows)}")


def main():
    """
    Main function to generate test queries using hyperparameters defined at the top of the file.

    Uses the following hyperparameters:
        - NUM_SAMPLES: Number of test samples to generate
        - ENTRY_FOLDER: Specific folder to use (or None for random)
        - SUNRGBD_ROOT: Root directory of SUNRGBD dataset
        - OUTPUT_DIR: Output directory for queries
        - OUTPUT_IMAGES_DIR: Output directory for images
        - OUTPUT_CSV: Output CSV file path
    """
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
        goal_objects=GOAL_OBJECTS
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
