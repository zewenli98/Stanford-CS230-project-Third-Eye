"""
Locate Target Object Using Object Detection Model

This script uses a trained object detection model to automatically detect
target objects in images and generate bounding boxes for pathfinding.

It replaces manual bbox annotation with automatic detection.
"""

import os
import sys
import csv
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import cv2
import torch
from PIL import Image

# =============================================================================
# CONFIGURATION - UPDATE THESE BEFORE RUNNING
# =============================================================================

# Path to queries folder
QUERY_PATH = "../queries-sample10/"  # Change to "../queries/" or other path as needed

# Path to object detection model
MODEL_PATH = "models/best.pth"  # Path to trained object detection model

# Confidence threshold for object detection
CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence to consider a detection valid

# IoU threshold for NMS (Non-Maximum Suppression)
IOU_THRESHOLD = 0.45

# =============================================================================
# END CONFIGURATION
# =============================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ObjectDetector:
    """
    Object detection wrapper for YOLOv5/YOLOv8 or similar models.
    """

    def __init__(self, model_path: str, device: str = None):
        """
        Initialize object detector.

        Args:
            model_path: Path to trained model (.pth file)
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.model_path = model_path

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        # Load model
        self.model = self._load_model()
        logger.info(f"Model loaded from {model_path}")

    def _normalize_object_name(self, name: str) -> str:
        """
        Normalize object name for matching.

        Args:
            name: Object name (e.g., "coffeemug", "Chair2", "chairs")

        Returns:
            Normalized name (e.g., "coffeemug", "chair", "chair")
        """
        import re

        # Convert to lowercase
        normalized = name.lower()

        # Remove numbers (e.g., "chair2" -> "chair")
        normalized = re.sub(r'\d+', '', normalized)

        # Remove special characters except letters
        normalized = re.sub(r'[^a-z]', '', normalized)

        # Handle common plural forms (optional - remove trailing 's' if it makes sense)
        # This is a simple heuristic; more sophisticated stemming could be used
        if normalized.endswith('s') and len(normalized) > 3:
            # Try singular form (e.g., "chairs" -> "chair")
            # But keep words that naturally end in 's' (e.g., "glass")
            singular = normalized[:-1]
            # Simple check: if removing 's' gives a valid-looking word, use it
            if not singular.endswith('s'):  # Avoid double-s words like "glass"
                normalized = singular

        return normalized

    def _load_model(self):
        """Load the object detection model."""
        try:
            # Try loading as YOLOv5/YOLOv8 model (most common for object detection)
            # First check if it's a YOLOv5 model
            if os.path.exists(self.model_path):
                logger.info("Attempting to load model...")

                # Try loading with torch.load first
                try:
                    model = torch.load(self.model_path, map_location=self.device)

                    # If it's a dict with 'model' key (common format)
                    if isinstance(model, dict):
                        if 'model' in model:
                            model = model['model']
                        elif 'state_dict' in model:
                            # This is just state dict, need to reconstruct model
                            logger.warning("Model is state_dict only. Trying to load with ultralytics...")
                            from ultralytics import YOLO
                            model = YOLO(self.model_path)
                            return model

                    # Set to eval mode
                    if hasattr(model, 'eval'):
                        model.eval()
                    if hasattr(model, 'to'):
                        model = model.to(self.device)

                    return model

                except Exception as e:
                    logger.warning(f"Failed to load with torch.load: {e}")
                    # Try loading with ultralytics YOLO
                    try:
                        from ultralytics import YOLO
                        model = YOLO(self.model_path)
                        logger.info("Model loaded with ultralytics YOLO")
                        return model
                    except Exception as e2:
                        logger.error(f"Failed to load with ultralytics: {e2}")
                        raise

            else:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def detect(self, image_path: str, target_object: str,
               confidence_threshold: float = CONFIDENCE_THRESHOLD,
               iou_threshold: float = IOU_THRESHOLD) -> Optional[List[int]]:
        """
        Detect target object in image and return bounding box.

        Args:
            image_path: Path to image file
            target_object: Name of object to detect (e.g., "chair", "bottle")
            confidence_threshold: Minimum confidence threshold
            iou_threshold: IoU threshold for NMS

        Returns:
            Bounding box [x1, y1, x2, y2] of largest detection, or None if not found
        """
        logger.info(f"Detecting '{target_object}' in {os.path.basename(image_path)}")

        # Load image
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return None

        # Run inference based on model type
        try:
            # Check if model has predict method (ultralytics YOLO)
            if hasattr(self.model, 'predict'):
                results = self.model.predict(
                    image_path,
                    conf=confidence_threshold,
                    iou=iou_threshold,
                    verbose=False
                )

                # Parse results
                if results and len(results) > 0:
                    result = results[0]  # First image result

                    # Get detections for target object
                    target_detections = []

                    # Get class names
                    class_names = result.names  # Dict mapping class_id to name

                    # Find target class id with intelligent mapping
                    target_class_id = None
                    matched_class_name = None

                    # Normalize target object name
                    target_normalized = self._normalize_object_name(target_object)

                    # Try different matching strategies
                    for class_id, class_name in class_names.items():
                        class_normalized = self._normalize_object_name(class_name)

                        # Strategy 1: Exact match (case insensitive)
                        if class_normalized == target_normalized:
                            target_class_id = class_id
                            matched_class_name = class_name
                            break

                        # Strategy 2: Class name contained in target (e.g., "mug" in "coffeemug")
                        if class_normalized in target_normalized:
                            target_class_id = class_id
                            matched_class_name = class_name
                            break

                        # Strategy 3: Target contained in class name (e.g., "chair" in "armchair")
                        if target_normalized in class_normalized:
                            target_class_id = class_id
                            matched_class_name = class_name
                            break

                    if target_class_id is None:
                        logger.warning(f"Object '{target_object}' not in model classes: {list(class_names.values())}")
                        logger.warning(f"Normalized target: '{target_normalized}'")
                        return None

                    logger.info(f"Mapped '{target_object}' to model class '{matched_class_name}'")

                    # Filter detections by target class
                    if result.boxes is not None and len(result.boxes) > 0:
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])

                            if class_id == target_class_id and confidence >= confidence_threshold:
                                # Get bbox in xyxy format
                                bbox = box.xyxy[0].cpu().numpy()
                                x1, y1, x2, y2 = bbox
                                area = (x2 - x1) * (y2 - y1)

                                target_detections.append({
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'confidence': confidence,
                                    'area': area
                                })

                    if not target_detections:
                        logger.warning(f"No '{target_object}' detected in image")
                        return None

                    # Select largest detection (by area)
                    largest_detection = max(target_detections, key=lambda x: x['area'])
                    logger.info(f"Found {len(target_detections)} '{target_object}' detection(s), "
                              f"selected largest (confidence: {largest_detection['confidence']:.2f})")

                    return largest_detection['bbox']

            else:
                # Fallback for other model types
                logger.error("Model type not supported. Please use ultralytics YOLO model.")
                return None

        except Exception as e:
            logger.error(f"Error during detection: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None


def load_prompts_info(query_path: str) -> List[Dict]:
    """
    Load prompt information from prompts.csv.

    Args:
        query_path: Path to queries folder

    Returns:
        List of dictionaries with prompt_id, image_name, object
    """
    csv_path = os.path.join(query_path, "prompts.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"prompts.csv not found at {csv_path}")

    prompts = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append({
                'prompt_id': int(row['prompt_id']),
                'image_name': row.get('image_name', ''),
                'object': row.get('object', '')
            })

    logger.info(f"Loaded {len(prompts)} prompts from CSV")
    return prompts


def detect_and_generate_targets(
    query_path: str,
    model_path: str,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    iou_threshold: float = IOU_THRESHOLD
):
    """
    Detect target objects in images and generate target_prediction.txt.

    Args:
        query_path: Path to queries folder
        model_path: Path to object detection model
        confidence_threshold: Minimum confidence threshold
        iou_threshold: IoU threshold for NMS
    """
    logger.info("="*80)
    logger.info("AUTOMATIC TARGET OBJECT DETECTION")
    logger.info("="*80)
    logger.info(f"Query Path: {query_path}")
    logger.info(f"Model Path: {model_path}")
    logger.info(f"Confidence Threshold: {confidence_threshold}")
    logger.info("="*80)

    # Load prompts
    prompts = load_prompts_info(query_path)

    # Initialize detector
    detector = ObjectDetector(model_path)

    # Detect objects
    results = []
    images_folder = os.path.join(query_path, "images")

    for prompt in prompts:
        prompt_id = prompt['prompt_id']
        image_name = prompt['image_name']
        target_object = prompt['object']

        # Skip if no object specified
        if not target_object or target_object.strip() == '':
            logger.warning(f"Prompt {prompt_id}: No object specified, skipping")
            continue

        # Construct image path
        image_path = os.path.join(images_folder, image_name)

        if not os.path.exists(image_path):
            logger.warning(f"Prompt {prompt_id}: Image not found: {image_path}, skipping")
            continue

        # Detect object
        bbox = detector.detect(
            image_path,
            target_object,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold
        )

        if bbox is not None:
            # Format bbox as string [x1, y1, x2, y2]
            bbox_str = f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"

            results.append({
                'prompt_id': prompt_id,
                'object': target_object,
                'bbox': bbox_str
            })

            logger.info(f"  ✓ Prompt {prompt_id}: {target_object} detected at {bbox_str}")
        else:
            logger.warning(f"  ✗ Prompt {prompt_id}: {target_object} not detected")

    # Save to target_prediction.txt
    output_path = os.path.join(query_path, "target_prediction.txt")

    if results:
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                line = f"{result['prompt_id']},{result['object']},{result['bbox']}\n"
                f.write(line)

        logger.info("="*80)
        logger.info(f"✓ Successfully detected {len(results)} objects out of {len(prompts)} queries")
        logger.info(f"✓ Results saved to: {output_path}")
        logger.info(f"✓ Format: prompt_id,object,bbox")
        logger.info("="*80)
    else:
        logger.warning("="*80)
        logger.warning("✗ No objects were detected")
        logger.warning("Possible reasons:")
        logger.warning("  - Model not trained on these object classes")
        logger.warning("  - Confidence threshold too high")
        logger.warning("  - Images don't contain target objects")
        logger.warning("="*80)


def main():
    """Main function."""

    # Validate configuration
    if QUERY_PATH == "place holder":
        logger.error("="*80)
        logger.error("ERROR: QUERY_PATH is not configured!")
        logger.error("="*80)
        logger.error("Please edit locate_target.py and set QUERY_PATH.")
        logger.error("Example: QUERY_PATH = '../queries/' or '../queries-sample10/'")
        logger.error("="*80)
        return

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    query_path = os.path.join(script_dir, QUERY_PATH)
    model_path = os.path.join(script_dir, MODEL_PATH)

    # Validate paths
    if not os.path.exists(query_path):
        logger.error(f"ERROR: QUERY_PATH does not exist: {query_path}")
        return

    if not os.path.exists(model_path):
        logger.error(f"ERROR: MODEL_PATH does not exist: {model_path}")
        logger.error(f"Please place your trained model at: {model_path}")
        return

    # Run detection
    try:
        detect_and_generate_targets(
            query_path=query_path,
            model_path=model_path,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            iou_threshold=IOU_THRESHOLD
        )
    except Exception as e:
        logger.error(f"Error during detection: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
