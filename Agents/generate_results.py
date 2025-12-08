"""
Complete pipeline for generating navigation results for blind users.
Processes queries from CSV, generates depth images, detects objects, finds targets,
generates path instructions, and produces LLM guidance.
"""

import csv
import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import cv2
import torch
from PIL import Image
from tqdm import tqdm

# Add parent directories to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

# Import local modules
from pathfinder import PathFinder
from object_detection.evaluation.inference import ObjectDetector
from distance_calculator.train import UNet

# Import LLM processor from get_results
sys.path.append(str(Path(__file__).parent.parent))
from get_results import LocalLLMProcessor, Query

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
TEST_FILE_PATH = "../queries/"
TEST_PROMPT_CSV = "prompts.csv"
TEST_IMAGE_PATH = TEST_FILE_PATH + "images/"
DEPTH_MODEL_PATH = "./distance_calculator/models/best.pth"
YOLO_MODEL_PATH = "./object_detection/models/best.pt"
TEMP_DIR = "./temp/"
DEPTH_OUTPUT_DIR = TEMP_DIR + "depth_images/"
OBJECT_OUTPUT_DIR = TEMP_DIR + "object_images/"
RESULTS_DIR = "./results/"

# Ensure directories exist
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(DEPTH_OUTPUT_DIR, exist_ok=True)
os.makedirs(OBJECT_OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================================
# READ QUERIES FROM CSV
# ============================================================================

def read_from_queries():
    """
    Read images, prompts, and ground truth from queries CSV.

    Returns:
        List of dictionaries containing query data
    """
    data_list = []
    prompt_filename = TEST_FILE_PATH + TEST_PROMPT_CSV

    # Check if file exists
    if not os.path.exists(prompt_filename):
        logger.error(f"The file '{prompt_filename}' does not exist.")
        return []

    try:
        with open(prompt_filename, mode='r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            data_list = list(csv_reader)
        logger.info(f"Loaded {len(data_list)} queries from {prompt_filename}")
    except Exception as e:
        logger.error(f"An error occurred while reading the file: {e}")

    return data_list


# ============================================================================
# DEPTH IMAGE GENERATION
# ============================================================================

def generate_depth_images(prompts: List[Dict]) -> Dict[int, str]:
    """
    Generate depth images from RGB images using pretrained U-Net model.

    Args:
        prompts: List of prompt dictionaries

    Returns:
        Dictionary mapping prompt_id to depth image path
    """
    logger.info("=" * 80)
    logger.info("GENERATING DEPTH IMAGES")
    logger.info("=" * 80)

    # Check if model exists
    if not os.path.exists(DEPTH_MODEL_PATH):
        logger.error(f"Depth model not found at {DEPTH_MODEL_PATH}")
        logger.info("Using existing depth images from CSV if available")
        # Return existing depth images from CSV
        depth_paths = {}
        for prompt in prompts:
            prompt_id = int(prompt['prompt_id'])
            if 'depth_image' in prompt and prompt['depth_image']:
                depth_paths[prompt_id] = os.path.join(TEST_IMAGE_PATH, prompt['depth_image'])
        return depth_paths

    # Load depth estimation model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Loading depth estimation model on {device}")

    model = UNet().to(device)
    checkpoint = torch.load(DEPTH_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info("Depth model loaded successfully")

    depth_paths = {}

    # Process each prompt
    for prompt in tqdm(prompts, desc="Generating depth images"):
        prompt_id = int(prompt['prompt_id'])
        image_name = prompt['image_name']
        rgb_path = os.path.join(TEST_IMAGE_PATH, image_name)

        if not os.path.exists(rgb_path):
            logger.warning(f"RGB image not found: {rgb_path}")
            continue

        try:
            # Load and preprocess RGB image
            rgb_image = cv2.imread(rgb_path)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            rgb_resized = cv2.resize(rgb_image, (640, 480))

            # Normalize and convert to tensor
            rgb_tensor = torch.from_numpy(rgb_resized).permute(2, 0, 1).float() / 255.0
            rgb_tensor = rgb_tensor.unsqueeze(0).to(device)

            # Generate depth prediction
            with torch.no_grad():
                depth_pred = model(rgb_tensor)

            # Convert to numpy and denormalize
            depth_map = depth_pred[0, 0].cpu().numpy()
            depth_map = (depth_map * 10000).astype(np.uint16)  # Scale to SUN RGB-D format

            # Save depth image
            depth_filename = f"{prompt_id:04d}_depth.png"
            depth_path = os.path.join(DEPTH_OUTPUT_DIR, depth_filename)
            cv2.imwrite(depth_path, depth_map)

            depth_paths[prompt_id] = depth_path

        except Exception as e:
            logger.error(f"Error generating depth for prompt {prompt_id}: {e}")

    logger.info(f"Generated {len(depth_paths)} depth images")
    return depth_paths


# ============================================================================
# OBJECT DETECTION
# ============================================================================

def object_location_detection(prompts: List[Dict]) -> Tuple[Dict[int, List], Dict[int, List]]:
    """
    Use pretrained YOLO model to detect objects and bounding boxes from RGB images.
    Generate cropped object images and save to temp directory.

    Args:
        prompts: List of prompt dictionaries

    Returns:
        Tuple of:
        - object_images_dict: Dict mapping prompt_id to list of object image paths
        - object_bboxes_dict: Dict mapping prompt_id to list of bounding boxes
    """
    logger.info("=" * 80)
    logger.info("DETECTING OBJECTS")
    logger.info("=" * 80)

    # Check if YOLO model exists
    if not os.path.exists(YOLO_MODEL_PATH):
        logger.error(f"YOLO model not found at {YOLO_MODEL_PATH}")
        return {}, {}

    # Initialize YOLO detector
    logger.info("Loading YOLO model...")
    detector = ObjectDetector(YOLO_MODEL_PATH, device='cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("YOLO model loaded successfully")

    object_images_dict = {}
    object_bboxes_dict = {}

    # Process each prompt
    for prompt in tqdm(prompts, desc="Detecting objects"):
        prompt_id = int(prompt['prompt_id'])
        image_name = prompt['image_name']
        rgb_path = os.path.join(TEST_IMAGE_PATH, image_name)

        if not os.path.exists(rgb_path):
            logger.warning(f"RGB image not found: {rgb_path}")
            continue

        try:
            # Run YOLO detection
            result = detector.predict_image(rgb_path, save=False, show=False)
            predictions = result['predictions']

            # Load original image for cropping
            image = cv2.imread(rgb_path)

            # Create output directory for this prompt
            prompt_output_dir = os.path.join(OBJECT_OUTPUT_DIR, f"{prompt_id:04d}")
            os.makedirs(prompt_output_dir, exist_ok=True)

            object_images = []
            object_bboxes = []

            # Process each detection
            for i, detection in enumerate(predictions['detections']):
                bbox = detection['bbox']
                class_name = detection['class_name']
                confidence = detection['confidence']

                # Extract bounding box coordinates
                x1 = int(bbox['x1'])
                y1 = int(bbox['y1'])
                x2 = int(bbox['x2'])
                y2 = int(bbox['y2'])

                # Crop object from image
                cropped_object = image[y1:y2, x1:x2]

                # Save cropped object
                object_filename = f"{class_name}_{i:03d}_conf{confidence:.2f}.jpg"
                object_path = os.path.join(prompt_output_dir, object_filename)
                cv2.imwrite(object_path, cropped_object)

                object_images.append(object_path)
                object_bboxes.append([x1, y1, x2, y2])

            object_images_dict[prompt_id] = object_images
            object_bboxes_dict[prompt_id] = object_bboxes

        except Exception as e:
            logger.error(f"Error detecting objects for prompt {prompt_id}: {e}")

    logger.info(f"Detected objects for {len(object_images_dict)} prompts")
    return object_images_dict, object_bboxes_dict


# ============================================================================
# TARGET OBJECT FINDING
# ============================================================================

def find_target_object(object_images_dict: Dict[int, List],
                       object_bboxes_dict: Dict[int, List],
                       prompts: List[Dict]) -> Dict[int, Dict]:
    """
    Use visual LLM to find the target object from detected objects.

    Args:
        object_images_dict: Dict mapping prompt_id to list of object image paths
        object_bboxes_dict: Dict mapping prompt_id to list of bounding boxes
        prompts: List of prompt dictionaries

    Returns:
        Dict mapping prompt_id to target object info (image_path, bbox, class_name)
    """
    logger.info("=" * 80)
    logger.info("FINDING TARGET OBJECTS")
    logger.info("=" * 80)

    target_objects = {}

    # For now, use a simpler approach: match by object name from CSV
    # TODO: Replace with LLM-based matching if needed

    for prompt in tqdm(prompts, desc="Finding target objects"):
        prompt_id = int(prompt['prompt_id'])
        target_name = prompt.get('object', '').lower().strip()

        if prompt_id not in object_images_dict:
            logger.warning(f"No detections for prompt {prompt_id}")
            continue

        # Get object images and bboxes for this prompt
        object_images = object_images_dict[prompt_id]
        object_bboxes = object_bboxes_dict[prompt_id]

        # Find matching object by name
        target_found = False
        for img_path, bbox in zip(object_images, object_bboxes):
            # Extract class name from filename
            filename = os.path.basename(img_path)
            class_name = filename.split('_')[0].lower()

            if class_name == target_name or target_name in class_name:
                target_objects[prompt_id] = {
                    'image_path': img_path,
                    'bbox': bbox,
                    'class_name': class_name
                }
                target_found = True
                break

        if not target_found:
            logger.warning(f"Target object '{target_name}' not found for prompt {prompt_id}")

    logger.info(f"Found {len(target_objects)} target objects")
    return target_objects


# ============================================================================
# PATH INSTRUCTION GENERATION
# ============================================================================

def generate_path_instructions(target_objects: Dict[int, Dict],
                               depth_paths: Dict[int, str],
                               prompts: List[Dict]) -> Dict[int, Any]:
    """
    Use pathfinder.py to generate navigation instructions.

    Args:
        target_objects: Dict mapping prompt_id to target object info
        depth_paths: Dict mapping prompt_id to depth image path
        prompts: List of prompt dictionaries

    Returns:
        Dict mapping prompt_id to NavigationInstruction objects
    """
    logger.info("=" * 80)
    logger.info("GENERATING PATH INSTRUCTIONS")
    logger.info("=" * 80)

    # Initialize PathFinder
    pathfinder = PathFinder(output_dir=os.path.join(RESULTS_DIR, "pathfinder_outputs"))

    path_instructions = {}

    # Process each prompt that has a target object
    for prompt in tqdm(prompts, desc="Generating paths"):
        prompt_id = int(prompt['prompt_id'])

        if prompt_id not in target_objects:
            continue

        # Get target object info
        target = target_objects[prompt_id]
        bbox = target['bbox']
        bbox_str = str(bbox)
        object_name = target['class_name']

        # Get RGB and depth paths
        rgb_path = os.path.join(TEST_IMAGE_PATH, prompt['image_name'])
        depth_path = depth_paths.get(prompt_id)

        # Use existing depth image from CSV if generated one not available
        if depth_path is None and 'depth_image' in prompt:
            depth_path = os.path.join(TEST_IMAGE_PATH, prompt['depth_image'])

        if not depth_path or not os.path.exists(depth_path):
            logger.warning(f"Depth image not found for prompt {prompt_id}")
            continue

        try:
            # Get annotation data if available
            annotation_json = prompt.get('annotation', None)

            # Process query with pathfinder
            instruction = pathfinder.process_query(
                rgb_path=rgb_path,
                depth_path=depth_path,
                object_name=object_name,
                bbox_str=bbox_str,
                annotation_json=annotation_json
            )

            path_instructions[prompt_id] = instruction

        except Exception as e:
            logger.error(f"Error generating path for prompt {prompt_id}: {e}")

    logger.info(f"Generated path instructions for {len(path_instructions)} prompts")
    return path_instructions


# ============================================================================
# LLM GUIDANCE GENERATION
# ============================================================================

def generate_llm_guidance(path_instructions: Dict[int, Any],
                         target_objects: Dict[int, Dict],
                         prompts: List[Dict],
                         model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct") -> Dict[int, str]:
    """
    Generate detailed LLM guidance by passing path instructions and target object information.

    Args:
        path_instructions: Dict mapping prompt_id to NavigationInstruction objects
        target_objects: Dict mapping prompt_id to target object info
        prompts: List of prompt dictionaries
        model_name: LLM model to use

    Returns:
        Dict mapping prompt_id to LLM guidance text
    """
    logger.info("=" * 80)
    logger.info("GENERATING LLM GUIDANCE")
    logger.info("=" * 80)

    # Initialize LLM processor
    processor = LocalLLMProcessor(device='cuda' if torch.cuda.is_available() else 'cpu')

    llm_guidance = {}

    # Prepare queries for LLM
    llm_queries = []
    query_prompt_ids = []

    for prompt in prompts:
        prompt_id = int(prompt['prompt_id'])

        if prompt_id not in path_instructions or prompt_id not in target_objects:
            continue

        # Get path instruction and target object
        instruction = path_instructions[prompt_id]
        target = target_objects[prompt_id]

        # Load RGB image
        rgb_path = os.path.join(TEST_IMAGE_PATH, prompt['image_name'])
        try:
            image = Image.open(rgb_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image for prompt {prompt_id}: {e}")
            continue

        # Build enhanced prompt with navigation details
        enhanced_prompt = build_enhanced_prompt(
            original_prompt=prompt['prompt_text'],
            instruction=instruction,
            target=target,
            prompt_data=prompt
        )

        # Create Query object
        query = Query(
            prompt_id=prompt_id,
            prompt_text=enhanced_prompt,
            image_name=prompt['image_name'],
            image=image
        )

        llm_queries.append(query)
        query_prompt_ids.append(prompt_id)

    if not llm_queries:
        logger.warning("No queries to process for LLM guidance")
        return llm_guidance

    # Call LLM in batch
    logger.info(f"Processing {len(llm_queries)} queries with {model_name}")
    responses = processor.call_llm_batch(model_name, model_name, llm_queries)

    # Map responses back to prompt IDs
    for prompt_id, response in zip(query_prompt_ids, responses):
        llm_guidance[prompt_id] = response

    logger.info(f"Generated LLM guidance for {len(llm_guidance)} prompts")
    return llm_guidance


def build_enhanced_prompt(original_prompt: str,
                         instruction: Any,
                         target: Dict,
                         prompt_data: Dict) -> str:
    """
    Build enhanced prompt with path instructions and object details.

    Args:
        original_prompt: Original user prompt
        instruction: NavigationInstruction object
        target: Target object info
        prompt_data: Original prompt data from CSV

    Returns:
        Enhanced prompt string
    """
    context = """
**Role:**
You are a visual-navigation assistant AI designed to help a blind user locate and retrieve objects.

**High-Level Task:**
The user has uploaded an image and asked you to find a specific object. You have been provided with:
1. Object detection results (bounding box location)
2. Distance and direction calculations
3. Navigation waypoints

Your job is to generate clear, detailed navigation instructions in JSON format.

---

## **RESPONSE FORMAT (Mandatory Structured JSON)**

You MUST return a JSON object with the following fields:

```json
{
  "found": true/false,
  "object_location_in_image": {
    "description": "Describe where the object appears in the image",
    "detailed_area_description": "Split image into 4 parts: left top, right top, left bottom, right bottom. Describe which quadrant contains the object"
  },
  "distance_and_direction_from_camera": {
    "distance_meters": float,
    "distance_inches": float,
    "direction": "description of direction (e.g., 'slightly to your right', 'straight ahead')"
  },
  "navigation_instructions": [
    "Step-by-step instructions from the user's current facing direction to approach the object.",
    "Only reference stable, touchable landmarks (table, chair, sofa, wall, counter, etc.).",
    "Flag obstacles in the path if any."
  ],
  "hand_guidance": "Describe how to position and move the user's hand to grab the object.",
  "fallback": "If object not found or image unclear, ask user to take another photo and suggest how to reposition."
}
```

---

## **PROVIDED INFORMATION:**

**Target Object:** {object_name}
**Object Bounding Box:** {bbox}
**Distance:** {distance}m ({distance_inches} inches)
**Direction:** {direction} ({direction_degrees}°)
**Is Fetchable (within arm's reach):** {is_fetchable}

"""

    # Add waypoint information if available
    if instruction.waypoints:
        waypoints_text = "**Navigation Waypoints:**\n"
        for i, wp in enumerate(instruction.waypoints, 1):
            waypoints_text += f"{i}. Walk toward {wp['direction']} for {wp['distance']} meters\n"
        context += waypoints_text + "\n"
    else:
        context += "**Navigation:** Object is within arm's reach, no navigation needed.\n\n"

    # Add warnings if any
    if instruction.warnings:
        context += "**Warnings:**\n"
        for warning in instruction.warnings:
            context += f"- {warning}\n"
        context += "\n"

    # Format the context with actual values
    distance_inches = instruction.distance_meters * 39.3701  # meters to inches

    formatted_context = context.format(
        object_name=target['class_name'],
        bbox=target['bbox'],
        distance=instruction.distance_meters,
        distance_inches=round(distance_inches, 1),
        direction=instruction.direction_clock,
        direction_degrees=instruction.direction_degrees,
        is_fetchable=instruction.is_fetchable
    )

    # Extract user question from original prompt
    user_question = original_prompt.split('\n')[-1] if '\n' in original_prompt else original_prompt

    return formatted_context + f"\n**User's Request:** {user_question}\n\nProvide your response as a JSON object only."


# ============================================================================
# RESULTS SAVING
# ============================================================================

def save_results(path_instructions: Dict[int, Any],
                llm_guidance: Dict[int, str],
                prompts: List[Dict],
                target_objects: Dict[int, Dict]) -> str:
    """
    Save results to CSV file in the results folder.

    Args:
        path_instructions: Dict mapping prompt_id to NavigationInstruction objects
        llm_guidance: Dict mapping prompt_id to LLM guidance text
        prompts: List of prompt dictionaries
        target_objects: Dict mapping prompt_id to target object info

    Returns:
        Path to saved CSV file
    """
    logger.info("=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)

    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(RESULTS_DIR, f"navigation_results_{timestamp}.csv")

    # Define CSV headers
    fieldnames = [
        'prompt_id', 'prompt_text', 'image_name', 'object',
        'ground_truth_distance', 'ground_truth_direction',
        'detected_object', 'detected_bbox',
        'calculated_distance', 'calculated_direction', 'direction_degrees',
        'is_fetchable', 'waypoints', 'warnings',
        'llm_guidance'
    ]

    # Write results to CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for prompt in prompts:
            prompt_id = int(prompt['prompt_id'])

            # Get instruction if available
            instruction = path_instructions.get(prompt_id)
            target = target_objects.get(prompt_id)
            guidance = llm_guidance.get(prompt_id, '')

            # Build row
            row = {
                'prompt_id': prompt_id,
                'prompt_text': prompt.get('prompt_text', ''),
                'image_name': prompt.get('image_name', ''),
                'object': prompt.get('object', ''),
                'ground_truth_distance': prompt.get('object_distance', ''),
                'ground_truth_direction': prompt.get('object_direction', ''),
            }

            # Add detection results
            if target:
                row['detected_object'] = target['class_name']
                row['detected_bbox'] = str(target['bbox'])
            else:
                row['detected_object'] = 'NOT_FOUND'
                row['detected_bbox'] = ''

            # Add path instruction results
            if instruction:
                row['calculated_distance'] = instruction.distance_meters
                row['calculated_direction'] = instruction.direction_clock
                row['direction_degrees'] = instruction.direction_degrees
                row['is_fetchable'] = instruction.is_fetchable
                row['waypoints'] = json.dumps(instruction.waypoints) if instruction.waypoints else ''
                row['warnings'] = json.dumps(instruction.warnings) if instruction.warnings else ''
            else:
                row['calculated_distance'] = ''
                row['calculated_direction'] = ''
                row['direction_degrees'] = ''
                row['is_fetchable'] = ''
                row['waypoints'] = ''
                row['warnings'] = ''

            # Add LLM guidance
            row['llm_guidance'] = guidance

            writer.writerow(row)

    logger.info(f"Results saved to {output_path}")
    return output_path


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main pipeline execution."""
    logger.info("\n" + "=" * 80)
    logger.info("STARTING NAVIGATION RESULTS GENERATION PIPELINE")
    logger.info("=" * 80 + "\n")

    # Read prompts from CSV
    prompts = read_from_queries()
    if not prompts:
        logger.error("No prompts loaded. Exiting.")
        return

    logger.info(f"Loaded {len(prompts)} prompts")

    # Step 1: Generate depth images
    depth_paths = generate_depth_images(prompts)
    logger.info(f"✓ Depth images: {len(depth_paths)} generated/loaded\n")

    # Step 2: Object location detection
    object_images_dict, object_bboxes_dict = object_location_detection(prompts)
    logger.info(f"✓ Object detection: {len(object_images_dict)} images processed\n")

    # Step 3: Find target objects
    target_objects = find_target_object(object_images_dict, object_bboxes_dict, prompts)
    logger.info(f"✓ Target objects found: {len(target_objects)}\n")

    # Step 4: Generate path instructions
    path_instructions = generate_path_instructions(target_objects, depth_paths, prompts)
    logger.info(f"✓ Path instructions generated: {len(path_instructions)}\n")

    # Step 5: Generate LLM guidance
    llm_guidance = generate_llm_guidance(path_instructions, target_objects, prompts)
    logger.info(f"✓ LLM guidance generated: {len(llm_guidance)}\n")

    # Step 6: Save results
    output_path = save_results(path_instructions, llm_guidance, prompts, target_objects)
    logger.info(f"✓ Results saved to: {output_path}\n")

    # Summary
    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total prompts processed: {len(prompts)}")
    logger.info(f"Depth images generated: {len(depth_paths)}")
    logger.info(f"Objects detected: {len(object_images_dict)}")
    logger.info(f"Target objects found: {len(target_objects)}")
    logger.info(f"Path instructions: {len(path_instructions)}")
    logger.info(f"LLM guidance: {len(llm_guidance)}")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
