"""
Agent-Enhanced Results Generator

This script integrates pathfinding spatial information with LLM vision models
to provide enhanced navigation instructions for blind users.

It reads:
- prompts.csv: Query prompts and images
- pathfinder_output.csv: Pathfinding results (distance, direction, waypoints)

Then calls LLM with enhanced prompts containing spatial information.
"""

import os
import csv
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import logging
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import json
from tqdm import tqdm
import gc
from peft import PeftModel

# =============================================================================
# CONFIGURATION - UPDATE THESE BEFORE RUNNING
# =============================================================================

# Path to queries folder (contains prompts.csv and pathfinder_output.csv)
QUERY_PATH = "./queries-sample10/"  # Change to "./queries-sample10/" if needed

# Model and processor paths
MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"  # Change to your model path
PROCESSOR_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"  # Change to your processor path

# Output folder for results
OUTPUT_FOLDER = "./agent_results/"

# Base model for LoRA (if using LoRA models)
BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

# =============================================================================
# END CONFIGURATION
# =============================================================================

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Data classes
@dataclass
class PathfinderResult:
    """Pathfinding result from pathfinder_output.csv"""
    prompt_id: int
    object_name: str
    distance_meters: float
    direction_clock: str
    direction_degrees: float
    is_fetchable: bool
    waypoints: str  # Semicolon-separated waypoints
    warnings: str


@dataclass
class Query:
    """Single query object containing image and prompt"""
    prompt_id: int
    prompt_text: str
    image_name: str
    image_path: str = None
    image: Image.Image = None
    # Additional fields from prompts.csv
    object: str = None
    object_distance: str = None
    object_direction: str = None
    scene: str = None
    object_bbox: str = None
    annotation: str = None
    depth_image: str = None
    # Pathfinding info
    pathfinder_info: Optional[PathfinderResult] = None


@dataclass
class Result:
    """Result object containing query and response"""
    prompt_id: int
    prompt_text: str
    image_name: str
    model_name: str
    response: str
    # Additional fields from prompts.csv
    object: str = None
    object_distance: str = None
    object_direction: str = None
    scene: str = None
    object_bbox: str = None
    annotation: str = None
    depth_image: str = None
    # Pathfinding fields
    pathfinder_distance: float = None
    pathfinder_direction: str = None
    pathfinder_is_fetchable: bool = None


class LocalLLMProcessor:
    """Handle batch processing for local Qwen models"""

    def __init__(self, device: str = None):
        """
        Initialize the local LLM processor

        Args:
            device: Device to run the model on ('cuda', 'cpu', or auto-detect)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_model_name = None

    def load_model(self, model_name: str, processor_name: str):
        """
        Load a local Qwen model

        Args:
            model_name: Path or name of the model
            processor_name: Path or name of the processor
        """
        if self.current_model_name == model_name:
            logger.info(f"Model {model_name} already loaded")
            return

        logger.info(f"Loading model: {model_name}")

        # Clear previous model from memory
        if self.model is not None:
            del self.model
            del self.processor
            if self.tokenizer is not None:
                del self.tokenizer
            torch.cuda.empty_cache()
            gc.collect()

        try:
            # For Qwen-VL models (vision-language models)
            if "VL" in model_name or "vl" in model_name:
                if "lora" in model_name.lower():
                    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        BASE_MODEL, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
                    ).cuda()
                    self.model = PeftModel.from_pretrained(base_model, model_name).cuda().eval()
                else:
                    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        model_name, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
                    )
                self.processor = AutoProcessor.from_pretrained(processor_name, trust_remote_code=True)

            if self.device == "cpu":
                self.model = self.model.to(self.device)

            self.model.eval()
            self.current_model_name = model_name
            logger.info(f"Model {model_name} loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise

    def call_llm_batch(self, model_name: str, processor_name: str, query_list: List[Query]) -> List[str]:
        """
        Call local LLM model in batch with images and prompts

        Args:
            model_name: Name/path of the model
            processor_name: Name/path of the processor
            query_list: List of Query objects containing images and prompts

        Returns:
            List of response texts from the LLM
        """
        # Load model if not already loaded
        self.load_model(model_name, processor_name)

        responses = []
        logger.info(f"Processing {len(query_list)} queries with model: {model_name}")

        # Process with progress bar
        for query in tqdm(query_list, desc="Processing queries with agent enhancement"):
            try:
                response = self._process_single_query(query)
                responses.append(response)
            except Exception as e:
                logger.error(f"Error processing query {query.prompt_id}: {e}")
                responses.append(f"Error: {str(e)}")

        return responses

    def _process_single_query(self, query: Query) -> str:
        """
        Process a single query with image and text

        Args:
            query: Query object with image and prompt

        Returns:
            Response text from the model
        """
        return self._process_vision_query(query)

    def _process_vision_query(self, query: Query) -> str:
        """Process query with Qwen-VL model"""

        # Prepare the messages in Qwen-VL format
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": query.image,
                    },
                    {"type": "text", "text": query.prompt_text},
                ],
            }
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Process inputs
        image_inputs, video_inputs = self._process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Move to device
        inputs = inputs.to(self.device)

        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
            )

        # Decode only the generated tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return response

    def _process_vision_info(self, messages):
        """Extract images and videos from messages"""
        image_inputs = []
        video_inputs = []

        for message in messages:
            if isinstance(message["content"], list):
                for item in message["content"]:
                    if item.get("type") == "image":
                        image_inputs.append(item["image"])
                    elif item.get("type") == "video":
                        video_inputs.append(item["video"])

        return image_inputs, video_inputs


def load_pathfinder_results(query_path: str) -> Dict[int, PathfinderResult]:
    """
    Load pathfinding results from pathfinder_output.csv

    Args:
        query_path: Path to queries folder

    Returns:
        Dictionary mapping prompt_id to PathfinderResult
    """
    csv_path = os.path.join(query_path, "pathfinder_output.csv")

    if not os.path.exists(csv_path):
        logger.warning(f"pathfinder_output.csv not found at {csv_path}")
        logger.warning("Proceeding without pathfinding information")
        return {}

    pathfinder_results = {}

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt_id = int(row['prompt_id'])

            # Parse is_fetchable
            is_fetchable_str = row.get('is_fetchable', 'False')
            is_fetchable = is_fetchable_str.lower() in ('true', '1', 'yes')

            result = PathfinderResult(
                prompt_id=prompt_id,
                object_name=row.get('object_name', ''),
                distance_meters=float(row.get('distance_meters', 0)),
                direction_clock=row.get('direction_clock', ''),
                direction_degrees=float(row.get('direction_degrees', 0)),
                is_fetchable=is_fetchable,
                waypoints=row.get('waypoints', ''),
                warnings=row.get('warnings', '')
            )
            pathfinder_results[prompt_id] = result

    logger.info(f"Loaded {len(pathfinder_results)} pathfinding results")
    return pathfinder_results


def update_prompt_context(prompt: str, pathfinder_info: Optional[PathfinderResult] = None) -> str:
    """
    Update prompt with context and pathfinding information

    Args:
        prompt: Original user prompt
        pathfinder_info: Optional pathfinding spatial information

    Returns:
        Enhanced prompt with context
    """
    context_prompt = """
        **Role:**
        You are a visual-navigation assistant AI designed to help a blind user locate and retrieve objects using a single image.

        **High-Level Task:**
        The user will upload an image of their surroundings and ask you to find a specific object.
        Your job is to:
        1. Decide whether the target object is visible in the image.
        2. If visible, describe its position in the image and its approximate direction and distance from the camera.
        3. Provide clear, discrete movement instructions the user can follow without seeing anything.
        4. Explain how to position and move the user's hand to grab the object once they are close.
        5. If the object is not visible or the scene is too ambiguous for safe guidance, say so clearly and propose how to retake the photo.

        **You must:**
        - Be honest about uncertainty.
        - Never invent objects that are not clearly visible.
        - Prefer safety over brevity; avoid suggesting fast or risky movements.
        - Assume the user is standing where the photo was taken and facing the same direction as the camera.
    """

    # Add pathfinding spatial information if available
    if pathfinder_info is not None:
        pathfinding_context = f"""

        **PATHFINDING SPATIAL INFORMATION (from depth sensor and navigation system):**

        You have access to accurate spatial information about the target object from our depth sensor and pathfinding system:

        - **Target Object:** {pathfinder_info.object_name}
        - **Distance from Camera:** {pathfinder_info.distance_meters} meters
        - **Direction:** {pathfinder_info.direction_clock} ({pathfinder_info.direction_degrees}° from center)
        - **Fetchable Status:** {"YES - Within arm's reach (< 0.5m). User can reach directly without walking." if pathfinder_info.is_fetchable else "NO - Requires walking to reach the object."}
        """

        # Add waypoints if available and object is not fetchable
        if not pathfinder_info.is_fetchable and pathfinder_info.waypoints:
            waypoint_list = pathfinder_info.waypoints.split(';')
            if waypoint_list and waypoint_list[0].strip():
                pathfinding_context += f"""
        - **Navigation Waypoints:** Step-by-step walking directions to reach the object:
"""
                for i, waypoint in enumerate(waypoint_list, 1):
                    if waypoint.strip():
                        pathfinding_context += f"          {i}. {waypoint.strip()}\n"

        # Add warnings if any
        if pathfinder_info.warnings:
            pathfinding_context += f"""
        - **Warnings:** {pathfinder_info.warnings}
        """

        pathfinding_context += """

        **IMPORTANT:** Use this spatial information to provide accurate, safe navigation instructions.
        The distance and direction values are measured from the depth sensor and are reliable.
        Cross-reference this data with what you see in the image to provide the best guidance.
        """

        context_prompt += pathfinding_context

    context_prompt += """

        ---

        ## **RESPONSE FORMAT (Mandatory Structured JSON)**

        You MUST always return a JSON object with the following fields:

        ```json
        {
        "found": true/false,
        "object_location_in_image": {
            "description": "Describe where the object appears in the image, or null if not present",
            "Detailed area description": "split image into 4 parts: left top, right top, left bottom, right bottom. Describe where the object appears in each part, or null if not present",
        },
        "distance_and_direction_from_camera": {
            "distance_meters": "float (approximate distance in meters from the camera to the object) or null if not present",
            "direction_o_clock": "integer from 1 to 12 (approximate direction in 12 o'clock format from the camera to the object) or null if not present"
        },
        "navigation_instructions": [
            "Tell the user in the second person perspective how to approach the object in the image, step-by-step instructions from the user's current facing direction to approach the object.",
            "Only reference stable, touchable landmarks (table, chair, sofa, wall, counter, etc.).",
            "Flag obstacles in the path."
        ],
        "hand_guidance": "Tell the user in the second person perspective how to position and move the user's hand to grab the object.",
        "fallback": "If object is not found or image is unclear, ask the user in the second person perspective to take another photo and suggest how to reposition."
        }
        ```

        **User's Request:**
    """

    return context_prompt + prompt


def load_queries_from_local(
    csv_path: str,
    images_folder: str,
    pathfinder_results: Dict[int, PathfinderResult]
) -> List[Query]:
    """
    Load queries from local files and merge with pathfinding results

    Args:
        csv_path: Path to CSV file containing prompts
        images_folder: Path to folder containing images
        pathfinder_results: Dictionary of pathfinding results

    Returns:
        List of Query objects with pathfinding info
    """
    queries = []

    # Check if paths exist
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not os.path.exists(images_folder):
        raise FileNotFoundError(f"Images folder not found: {images_folder}")

    logger.info(f"Loading queries from {csv_path}")

    # Read CSV file
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            prompt_id = int(row.get('prompt_id', 0))
            prompt_text = row.get('prompt_text', '').strip()
            image_name = row.get('image_name', '').strip()

            # Get pathfinding info for this query
            pathfinder_info = pathfinder_results.get(prompt_id)

            # Update prompt with pathfinding context
            enhanced_prompt = update_prompt_context(prompt_text, pathfinder_info)

            # Load image
            image_path = os.path.join(images_folder, image_name)
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}, skipping query {prompt_id}")
                continue

            try:
                # Load image with PIL
                image = Image.open(image_path).convert('RGB')

                # Create query with pathfinding info
                query = Query(
                    prompt_id=prompt_id,
                    prompt_text=enhanced_prompt,  # Use enhanced prompt
                    image_name=image_name,
                    image_path=image_path,
                    image=image,
                    object=row.get('object', ''),
                    object_distance=row.get('object_distance', ''),
                    object_direction=row.get('object_direction', ''),
                    scene=row.get('scene', ''),
                    object_bbox=row.get('object_bbox', ''),
                    annotation=row.get('annotation', ''),
                    depth_image=row.get('depth_image', ''),
                    pathfinder_info=pathfinder_info
                )
                queries.append(query)

            except Exception as e:
                logger.error(f"Error loading image {image_path}: {e}")
                continue

    logger.info(f"Loaded {len(queries)} queries successfully")
    return queries


def save_results_to_csv(
    results: List[Result],
    output_path: str = None
) -> str:
    """
    Save results to CSV file

    Args:
        results: List of Result objects
        output_path: Path for output CSV

    Returns:
        Path to the saved CSV file
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"agent_results_{timestamp}.csv"

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Saving {len(results)} results to {output_path}")

    # Write results to CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'prompt_id', 'prompt_text', 'image_name', 'object',
            'object_distance', 'object_direction', 'scene', 'object_bbox',
            'annotation', 'depth_image',
            'pathfinder_distance', 'pathfinder_direction', 'pathfinder_is_fetchable',
            'model_name', 'llm_response'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow({
                'prompt_id': result.prompt_id,
                'prompt_text': result.prompt_text,
                'image_name': result.image_name,
                'object': result.object,
                'object_distance': result.object_distance,
                'object_direction': result.object_direction,
                'scene': result.scene,
                'object_bbox': result.object_bbox,
                'annotation': result.annotation,
                'depth_image': result.depth_image,
                'pathfinder_distance': result.pathfinder_distance,
                'pathfinder_direction': result.pathfinder_direction,
                'pathfinder_is_fetchable': result.pathfinder_is_fetchable,
                'model_name': result.model_name,
                'llm_response': result.response
            })

    logger.info(f"Results saved successfully to {output_path}")
    return output_path


def eval_with_agent(
    model_name: str,
    processor_name: str,
    query_path: str = QUERY_PATH,
    output_folder: str = OUTPUT_FOLDER,
    device: str = None
) -> str:
    """
    Evaluate model with agent enhancement (pathfinding integration)

    Args:
        model_name: Path to model
        processor_name: Path to processor
        query_path: Path to queries folder
        output_folder: Folder to save results
        device: Device to run on ('cuda', 'cpu', or None for auto)

    Returns:
        Path to output CSV file
    """
    logger.info("="*80)
    logger.info("AGENT-ENHANCED EVALUATION")
    logger.info("="*80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Query Path: {query_path}")
    logger.info(f"Output Folder: {output_folder}")
    logger.info("="*80)

    # Construct paths
    csv_path = os.path.join(query_path, "prompts.csv")
    images_folder = os.path.join(query_path, "images")

    # Load pathfinding results
    pathfinder_results = load_pathfinder_results(query_path)

    # Load queries with pathfinding enhancement
    queries = load_queries_from_local(csv_path, images_folder, pathfinder_results)

    if not queries:
        logger.error("No queries loaded, exiting evaluation")
        return None

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Initialize processor
    processor = LocalLLMProcessor(device=device)

    try:
        # Call LLM in batch
        responses = processor.call_llm_batch(model_name, processor_name, queries)

        # Create Result objects
        results = []
        for query, response in zip(queries, responses):
            result = Result(
                prompt_id=query.prompt_id,
                prompt_text=query.prompt_text,
                image_name=query.image_name,
                model_name=model_name,
                response=response,
                # Include all additional fields from query
                object=query.object,
                object_distance=query.object_distance,
                object_direction=query.object_direction,
                scene=query.scene,
                object_bbox=query.object_bbox,
                annotation=query.annotation,
                depth_image=query.depth_image,
                # Add pathfinding fields
                pathfinder_distance=query.pathfinder_info.distance_meters if query.pathfinder_info else None,
                pathfinder_direction=query.pathfinder_info.direction_clock if query.pathfinder_info else None,
                pathfinder_is_fetchable=query.pathfinder_info.is_fetchable if query.pathfinder_info else None
            )
            results.append(result)

        # Save results to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = model_name.replace("/", "_").replace(":", "_").replace("\\", "_")
        output_path = os.path.join(
            output_folder,
            f"agent_results_{safe_model_name}_{timestamp}.csv"
        )

        saved_path = save_results_to_csv(results, output_path)

        logger.info("="*80)
        logger.info(f"✓ Evaluation completed successfully")
        logger.info(f"✓ Results saved to: {saved_path}")
        logger.info("="*80)

        return saved_path

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def main():
    """Main function"""

    # Validate configuration
    if QUERY_PATH == "place holder":
        logger.error("="*80)
        logger.error("ERROR: QUERY_PATH is not configured!")
        logger.error("="*80)
        logger.error("Please edit get_agents_results.py and set QUERY_PATH at the top.")
        logger.error("Example: QUERY_PATH = './queries/' or './queries-sample10/'")
        logger.error("="*80)
        return

    if not os.path.exists(QUERY_PATH):
        logger.error(f"ERROR: QUERY_PATH does not exist: {QUERY_PATH}")
        return

    # Run evaluation
    logger.info(f"Starting agent-enhanced evaluation")
    logger.info(f"Model: {MODEL_PATH}")
    logger.info(f"Processor: {PROCESSOR_PATH}")

    result_path = eval_with_agent(
        model_name=MODEL_PATH,
        processor_name=PROCESSOR_PATH,
        query_path=QUERY_PATH,
        output_folder=OUTPUT_FOLDER,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    if result_path:
        print("\n" + "="*80)
        print("✓ Agent-enhanced evaluation complete!")
        print(f"✓ Results saved to: {result_path}")
        print("="*80)
    else:
        print("\n✗ Evaluation failed. Check logs for details.")


if __name__ == "__main__":
    main()
