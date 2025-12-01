import os
import csv
import json
import re
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import logging
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
import base64
from io import BytesIO

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# ============================================================================
# CONFIGURATION - Configure these settings
# ============================================================================
RESULTS_CSV_FILES = [
    "results_._finetuned_models_LoRA-SpaceThinker-Qwen2.5-VL-7B-Instruct_epoch-1_20251201_095202.csv",
    "results_._finetuned_models_LoRA-OpenSpaces_MC_R1-Qwen2.5-VL-7B-Instruct_epoch-3_20251201_095053.csv",
    "results_._finetuned_models_LoRA-OpenSpaces_MC_R1-Qwen2.5-VL-7B-Instruct_epoch-2_20251201_094943.csv",
    "results_._finetuned_models_LoRA-OpenSpaces_MC_R1-Qwen2.5-VL-7B-Instruct_epoch-1_20251201_094839.csv",
    "results_._finetuned_models_full_params-SpaceThinker-Qwen2.5-VL-7B-Instruct_epoch-3_20251201_094729.csv",
    "results_._finetuned_models_full_params-SpaceThinker-Qwen2.5-VL-7B-Instruct_epoch-2_20251201_094500.csv",
    "results_._finetuned_models_full_params-SpaceThinker-Qwen2.5-VL-7B-Instruct_epoch-1_20251201_094229.csv",
    "results_._finetuned_models_full_params-OpenSpaces_MC_R1-Qwen2.5-VL-7B-Instruct_epoch-3_20251201_094000.csv",
    "results_._finetuned_models_full_params-OpenSpaces_MC_R1-Qwen2.5-VL-7B-Instruct_epoch-2_20251201_093404.csv",
    "results_._finetuned_models_full_params-OpenSpaces_MC_R1-Qwen2.5-VL-7B-Instruct_epoch-1_20251201_093035.csv",
    ] # Name of the results CSV file to evaluate
RESULTS_FOLDER = "./results_1130"  # Folder containing results files
# ============================================================================

EVALUATION_METRICS = {
    'found': 5,
    'object_location_description': 5,
    'object_location_detailed_area': 5,
    'distance': 5,
    'direction': 5,
    'navigation_instructions': 5,
    'hand_guidance': 5,
    'fallback': 5
}

@dataclass
class ExtractedResponse:
    """Extracted information from LLM response"""
    # Original fields
    prompt_id: int
    prompt_text: str
    image_name: str
    model_name: str
    raw_response: str

    # Ground truth from prompts.csv
    gt_object: str
    gt_object_distance: float  # in meters
    gt_object_direction: str
    gt_object_bbox: List[float]

    gt_scene: str
    annotation: Dict[str, Any]
    depth_image: str

    # Parsed LLM response fields
    found: Optional[bool] = None
    predicted_bbox: Optional[List[float]] = None
    predicted_object_location: Optional[str] = None
    predicted_object_location_details: Optional[str] = None

    predicted_distance_inches: Optional[float] = None
    predicted_direction: Optional[str] = None
    navigation_instructions: Optional[List[str]] = None
    hand_guidance: Optional[str] = None
    fallback: Optional[str] = None

    # Parse status
    parse_success: bool = False
    parse_error: Optional[str] = None


def load_results_from_csv(csv_path: str) -> List[Dict[str, Any]]:
    """
    Load results from a CSV file.

    Args:
        csv_path: Path to the results CSV file

    Returns:
        List of dictionaries containing all columns from the CSV
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results CSV file not found: {csv_path}")

    results = []
    logger.info(f"Loading results from {csv_path}")

    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            results.append(row)

    logger.info(f"Loaded {len(results)} results from CSV")
    return results


def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON content from an LLM response containing ```json ... ``` or plain JSON.
    Returns a Python dict.
    """

    # 1. Remove code fences like ```json ... ```
    cleaned = re.sub(r"```json|```", "", response, flags=re.IGNORECASE).strip()

    # 2. Sometimes model outputs extra characters before/after JSON. 
    #    Try to extract the largest JSON object using regex.
    json_match = re.search(r"\{[\s\S]*\}", cleaned)
    if not json_match:
        raise ValueError("No JSON object found in the input text")

    json_str = json_match.group(0)

    # 3. Parse JSON safely
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON extracted: {e}\nJSON string was:\n{json_str}")


def parse_bbox(bbox_str: str) -> Optional[List[float]]:
    """
    Parse bounding box from string format.

    Args:
        bbox_str: String representation of bbox (e.g., "[100, 200, 300, 400]")

    Returns:
        List of 4 floats [x_min, y_min, x_max, y_max] or None
    """
    try:
        bbox = json.loads(bbox_str)
        if isinstance(bbox, list) and len(bbox) == 4:
            return [float(x) for x in bbox]
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return None


def parse_llm_response(row: Dict[str, Any]) -> ExtractedResponse:
    """
    Parse a single result row and extract information from LLM response.

    Args:
        row: Dictionary containing a single row from results CSV

    Returns:
        ExtractedResponse object with parsed information
    """
    required_fields = ['prompt_id', 'annotation', 'prompt_text', 'image_name', 'model_name', 'llm_response', 'object', 'object_distance', 'object_direction', 'scene', 'object_bbox']
    for field in required_fields:
        if not row.get(field):
            raise ValueError(f"Required field {field} is not present in row: {row}")

    # Extract basic fields
    prompt_id = int(row.get('prompt_id', 0))
    prompt_text = row.get('prompt_text', '')
    image_name = row.get('image_name', '')
    model_name = row.get('model_name', '')
    raw_response = row.get('llm_response', '')
    gt_object = row.get('object', '')
    gt_object_distance = float(row.get('object_distance', 0))
    gt_object_direction = row.get('object_direction', '')
    gt_scene = row.get('scene', '')
    gt_object_bbox = row.get('object_bbox', '[]')
    annotation = row.get('annotation', '[]')
    depth_image = row.get('depth_image', '')

    # Create response object
    extracted = ExtractedResponse(
        prompt_id=prompt_id,
        prompt_text=prompt_text,
        image_name=image_name,
        model_name=model_name,
        raw_response=raw_response,
        gt_object=gt_object,
        gt_object_distance=gt_object_distance,
        gt_object_direction=gt_object_direction,
        gt_scene=gt_scene,
        gt_object_bbox=gt_object_bbox if gt_object_bbox else [],
        annotation=annotation,
        depth_image=depth_image
    )

    # Try to parse JSON response
    try:
        json_data = extract_json_from_response(raw_response)

        if json_data is None:
            extracted.parse_success = False
            extracted.parse_error = "Could not extract JSON from response"
            return extracted

        # Extract fields from JSON
        extracted.found = json_data.get('found')

        # Extract object location
        obj_location = json_data.get('object_location_in_image', {})
        if obj_location:
            extracted.predicted_object_location = obj_location.get('description')
            extracted.predicted_object_location_details = obj_location.get('Detailed area description') 
        
        # Extract distance and direction
        distance_direction = json_data.get('distance_and_direction_from_camera', {})
        if distance_direction:
            extracted.predicted_distance_inches = distance_direction.get('distance_inches')
            extracted.predicted_direction = distance_direction.get('direction')

        # Extract navigation instructions
        nav_instructions = json_data.get('navigation_instructions')
        if nav_instructions and isinstance(nav_instructions, list):
            extracted.navigation_instructions = nav_instructions

        # Extract hand guidance and fallback
        extracted.hand_guidance = json_data.get('hand_guidance')
        extracted.fallback = json_data.get('fallback')

        extracted.parse_success = True

    except Exception as e:
        extracted.parse_success = False
        extracted.parse_error = str(e)
        logger.warning(f"Error parsing response for prompt_id {prompt_id}: {e}")

    return extracted


def extract_all_responses(csv_path: str) -> List[ExtractedResponse]:
    """
    Load results CSV and extract all LLM responses.

    Args:
        csv_path: Path to results CSV file

    Returns:
        List of ExtractedResponse objects
    """
    results = load_results_from_csv(csv_path)
    extracted_responses = []

    logger.info("Parsing LLM responses...")
    for row in results:
        extracted = parse_llm_response(row)
        extracted_responses.append(extracted)

    # Log parsing statistics
    successful_parses = sum(1 for r in extracted_responses if r.parse_success)
    logger.info(f"Successfully parsed {successful_parses}/{len(extracted_responses)} responses")

    return extracted_responses


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64 string.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def distance_evaluation(predicted_distance_inches: Optional[float],
                       gt_distance_meters: float,
                       tolerance_percent: float = 20.0) -> bool:
    """
    Evaluate if the predicted distance is within acceptable tolerance of ground truth.

    Args:
        predicted_distance_inches: Predicted distance in inches
        gt_distance_meters: Ground truth distance in meters
        tolerance_percent: Acceptable error percentage (default 20%)

    Returns:
        True if prediction is within tolerance, False otherwise
    """
    if predicted_distance_inches is None or gt_distance_meters is None:
        return False

    # Convert predicted inches to meters (1 inch = 0.0254 meters)
    predicted_meters = predicted_distance_inches * 0.0254

    # Calculate percentage error
    error_percent = abs(predicted_meters - gt_distance_meters) / gt_distance_meters * 100

    return error_percent <= tolerance_percent


def evaluate_response_with_llm(response: ExtractedResponse, image_path: str) -> Dict[str, int]:
    """
    Use GPT-4 Vision to evaluate all aspects of the response in a single call.

    Args:
        response: ExtractedResponse object containing predictions and ground truth
        image_path: Path to the image file

    Returns:
        Dictionary with scores for each evaluation metric
    """
    # Initialize scores (all start at 0)
    scores = {
        'object_location_description': 0,
        'object_location_detailed_area': 0,
        'direction': 0,
        'navigation_instructions': 0,
        'hand_guidance': 0,
        'fallback': 0
    }

    # If parsing failed or object not found when it should be, return zero scores
    if not response.parse_success:
        logger.warning(f"Skipping LLM evaluation for prompt_id {response.prompt_id}: parse failed")
        return scores

    # Check if image exists
    if not os.path.exists(image_path):
        logger.warning(f"Image not found: {image_path}. Skipping LLM evaluation.")
        return scores

    # Encode image to base64
    try:
        base64_image = encode_image_to_base64(image_path)
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return scores

    # Construct the evaluation prompt
    prompt = f"""You are an expert evaluator for a visual navigation assistance system for blind and low-vision individuals.
    Your task is to evaluate the quality of the system's response based on the image and ground truth information.

    **Ground Truth Information:**
    - Object: {response.gt_object}
    - Scene: {response.gt_scene}
    - Actual Distance: {response.gt_object_distance:.2f} meters
    - Actual Direction: {response.gt_object_direction}

    **System's Response:**
    - Found: {response.found}
    - Object Location Description: {response.predicted_object_location or 'Not provided'}
    - Detailed Area Description: {response.predicted_object_location_details or 'Not provided'}
    - Predicted Direction: {response.predicted_direction or 'Not provided'}
    - Navigation Instructions: {response.navigation_instructions or 'Not provided'}
    - Hand Guidance: {response.hand_guidance or 'Not provided'}
    - Fallback: {response.fallback or 'Not provided'}

    **Evaluation Criteria:**
    Please evaluate each aspect on a scale and return ONLY a JSON object with the following structure:

    {{
        "object_location_description": <0-5 points>,  // How accurate is the general location description of the object?
        "object_location_detailed_area": <0-5 points>,  // How detailed and accurate is the spatial area description?
        "direction": <0-5 points>,  // How accurate is the direction relative to camera (left/right/center/front)?
        "navigation_instructions": <0-5 points>,  // Are the navigation instructions clear, actionable, and safe?
        "hand_guidance": <0-5 points>,  // Is the hand guidance instruction helpful and appropriate?
        "fallback": <0-5 points>  // Is the fallback instruction reasonable and helpful?
    }}

    **Scoring Guidelines:**
    - 5: Excellent - Perfect or near-perfect accuracy and helpfulness
    - 4: Good - Accurate with minor issues
    - 3: Acceptable - Mostly accurate but with some notable issues
    - 2: Poor - Significant inaccuracies or unhelpful
    - 1: Very Poor - Highly inaccurate or misleading
    - 0: Not provided or completely wrong

    Return ONLY the JSON object, no additional text."""

    try:
        # Call GPT-4 Vision API
        logger.debug(f"Calling OpenAI API for prompt_id {response.prompt_id}")
        response_llm = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500,
            temperature=0.0,  # Use temperature 0 for more consistent evaluation
            response_format={"type": "json_object"}  # Request JSON mode
        )

        # Extract the response text
        llm_response_text = response_llm.choices[0].message.content.strip()

        # Log the raw response for debugging
        logger.debug(f"Raw LLM response: {llm_response_text}")

        # Parse the JSON response using our robust extraction function
        llm_scores = extract_json_from_response(llm_response_text)

        if llm_scores is None:
            logger.error(f"Failed to extract JSON from LLM response for prompt_id {response.prompt_id}")
            logger.error(f"Raw response was: {llm_response_text[:200]}...")
            return scores

        # Update scores with LLM evaluation
        for key in scores.keys():
            if key in llm_scores:
                scores[key] = llm_scores[key]

        logger.info(f"LLM evaluation completed for prompt_id {response.prompt_id}")

    except Exception as e:
        logger.error(f"Error in LLM evaluation for prompt_id {response.prompt_id}: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        # Return zero scores on error

    return scores


def evaluate(extracted_responses: List[ExtractedResponse], test_mode: bool = True) -> List[Dict[str, Any]]:
    """
    Evaluate the extracted responses and calculate metrics.

    Args:
        extracted_responses: List of ExtractedResponse objects
        test_mode: If True, only evaluate the first response (for testing)

    Returns:
        List of dictionaries, each containing evaluation metrics for one response
    """
    metrics = []
    images_folder = "./queries/images"

    logger.info("="*70)
    if test_mode:
        logger.info("Starting Evaluation (TEST MODE - 1 sample only)")
    else:
        logger.info("Starting Evaluation")
    logger.info("="*70)

    for idx, response in enumerate(extracted_responses):
        logger.info(f"Evaluating response {idx + 1}/{len(extracted_responses)} (prompt_id: {response.prompt_id})")

        # Initialize metric dictionary for this response
        metric = {
            'prompt_id': response.prompt_id,
            'image_name': response.image_name,
            'found': 0,
            'object_location_description': 0,
            'object_location_detailed_area': 0,
            'distance': 0,
            'direction': 0,
            'navigation_instructions': 0,
            'hand_guidance': 0,
            'fallback': 0,
            'total_score': 0,
            'max_possible_score': sum(EVALUATION_METRICS.values())
        }

        # Check if object detection is correct
        # Award points if: (1) found=True and object exists, OR (2) found=False and object doesn't exist
        if response.found and response.gt_object_distance is not None:
            metric['found'] = EVALUATION_METRICS['found']
        elif not response.found and response.gt_object_distance is None:
            metric['found'] = EVALUATION_METRICS['found']
        else:
            metric['found'] = 0
            logger.info(f"  - Object detection: INCORRECT (found={response.found}, gt_exists={response.gt_object_distance is not None})")

        # Only evaluate further if object was correctly identified as present
        if metric['found'] > 0 and response.found:
            # Evaluate distance
            if distance_evaluation(response.predicted_distance_inches, response.gt_object_distance):
                metric['distance'] = EVALUATION_METRICS['distance']
                logger.info(f"  - Distance: CORRECT (within tolerance)")
            else:
                metric['distance'] = 0
                logger.info(f"  - Distance: INCORRECT")

            # Construct image path
            image_path = os.path.join(images_folder, response.image_name)

            # Call LLM once to evaluate all remaining aspects
            logger.info(f"  - Calling LLM for evaluation...")
            llm_scores = evaluate_response_with_llm(response, image_path)

            # Update metric with LLM scores
            metric['object_location_description'] = llm_scores.get('object_location_description', 0)
            metric['object_location_detailed_area'] = llm_scores.get('object_location_detailed_area', 0)
            metric['direction'] = llm_scores.get('direction', 0)
            metric['navigation_instructions'] = llm_scores.get('navigation_instructions', 0)
            metric['hand_guidance'] = llm_scores.get('hand_guidance', 0)
            metric['fallback'] = llm_scores.get('fallback', 0)

            logger.info(f"  - LLM Scores: location_desc={llm_scores.get('object_location_description', 0)}, "
                       f"detailed_area={llm_scores.get('object_location_detailed_area', 0)}, "
                       f"direction={llm_scores.get('direction', 0)}, "
                       f"nav={llm_scores.get('navigation_instructions', 0)}, "
                       f"hand={llm_scores.get('hand_guidance', 0)}, "
                       f"fallback={llm_scores.get('fallback', 0)}")

        # Calculate total score for this response
        metric['total_score'] = (
            metric['found'] +
            metric['object_location_description'] +
            metric['object_location_detailed_area'] +
            metric['distance'] +
            metric['direction'] +
            metric['navigation_instructions'] +
            metric['hand_guidance'] +
            metric['fallback']
        )

        logger.info(f"  - Total Score: {metric['total_score']}/{metric['max_possible_score']}")

        metrics.append(metric)

        # Break after first sample if in test mode
        if test_mode:
            logger.info("Test mode: Stopping after 1 sample")
            break

    # Calculate aggregate statistics
    logger.info("="*70)
    logger.info("Evaluation Summary")
    logger.info("="*70)
    total_samples = len(metrics)
    avg_score = sum(m['total_score'] for m in metrics) / total_samples if total_samples > 0 else 0
    avg_possible = sum(m['max_possible_score'] for m in metrics) / total_samples if total_samples > 0 else 0

    logger.info(f"Total samples evaluated: {total_samples}")
    logger.info(f"Average score: {avg_score:.2f}/{avg_possible:.2f} ({(avg_score/avg_possible*100) if avg_possible > 0 else 0:.1f}%)")
    logger.info("="*70)

    return metrics


def save_evaluation_report(extracted_responses: List[ExtractedResponse],
                          metrics: List[Dict[str, Any]],
                          output_path: str):
    """
    Save detailed evaluation report to CSV.

    Args:
        extracted_responses: List of ExtractedResponse objects
        metrics: List of evaluation metrics dictionaries (one per response)
        output_path: Path to save the report CSV
    """
    logger.info(f"Saving evaluation report to {output_path}")

    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'prompt_id', 'image_name', 'model_name',
            'gt_object', 'gt_distance_m', 'gt_direction', 'gt_scene',
            'parse_success', 'found_predicted', 'found_score',
            'predicted_distance_inches', 'predicted_direction',
            'score_object_location_desc', 'score_object_location_detailed',
            'score_distance', 'score_direction',
            'score_navigation', 'score_hand_guidance', 'score_fallback',
            'total_score', 'max_score', 'percentage',
            'parse_error'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        # Create a mapping from prompt_id to metric for easy lookup
        metric_map = {m['prompt_id']: m for m in metrics}

        for response in extracted_responses:
            # Get the corresponding metric
            metric = metric_map.get(response.prompt_id, {})

            # Calculate percentage
            total = metric.get('total_score', 0)
            max_score = metric.get('max_possible_score', 1)
            percentage = (total / max_score * 100) if max_score > 0 else 0

            writer.writerow({
                'prompt_id': response.prompt_id,
                'image_name': response.image_name,
                'model_name': response.model_name,
                'gt_object': response.gt_object,
                'gt_distance_m': response.gt_object_distance,
                'gt_direction': response.gt_object_direction,
                'gt_scene': response.gt_scene,
                'parse_success': response.parse_success,
                'found_predicted': response.found,
                'found_score': metric.get('found', 0),
                'predicted_distance_inches': response.predicted_distance_inches,
                'predicted_direction': response.predicted_direction,
                'score_object_location_desc': metric.get('object_location_description', 0),
                'score_object_location_detailed': metric.get('object_location_detailed_area', 0),
                'score_distance': metric.get('distance', 0),
                'score_direction': metric.get('direction', 0),
                'score_navigation': metric.get('navigation_instructions', 0),
                'score_hand_guidance': metric.get('hand_guidance', 0),
                'score_fallback': metric.get('fallback', 0),
                'total_score': total,
                'max_score': max_score,
                'percentage': f"{percentage:.1f}%",
                'parse_error': response.parse_error or ''
            })

    logger.info(f"Evaluation report saved to {output_path}")


def main():
    """
    Main function to evaluate results.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Evaluate Third Eye model results using LLM-based evaluation'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: only evaluate 1 sample to save API costs'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to evaluate (for testing with specific count)'
    )
    args = parser.parse_args()

    for result_single_file in RESULTS_CSV_FILES:
        # Construct full path to results CSV
        results_csv_path = os.path.join(RESULTS_FOLDER, result_single_file)

        logger.info("="*70)
        logger.info("Third Eye Evaluation Pipeline")
        logger.info("="*70)
        logger.info(f"Results file: {results_csv_path}")
        if args.test:
            logger.info("Mode: TEST (1 sample only)")
        elif args.max_samples:
            logger.info(f"Mode: LIMITED ({args.max_samples} samples)")

        # Extract all responses from CSV
        extracted_responses = extract_all_responses(results_csv_path)

        # Limit samples if max_samples is specified
        if args.max_samples and not args.test:
            extracted_responses = extracted_responses[:args.max_samples]
            logger.info(f"Limited to {len(extracted_responses)} samples")

        # Evaluate the responses
        metrics = evaluate(extracted_responses, test_mode=args.test)

        # Save detailed evaluation report
        report_filename = result_single_file.replace('.csv', '_evaluation.csv')
        report_path = os.path.join(RESULTS_FOLDER, report_filename)
        save_evaluation_report(extracted_responses, metrics, report_path)

        # Calculate summary statistics
        total_samples = len(metrics)
        if total_samples > 0:
            summary = {
                'total_samples': total_samples,
                'average_scores': {
                    'found': sum(m['found'] for m in metrics) / total_samples,
                    'object_location_description': sum(m['object_location_description'] for m in metrics) / total_samples,
                    'object_location_detailed_area': sum(m['object_location_detailed_area'] for m in metrics) / total_samples,
                    'distance': sum(m['distance'] for m in metrics) / total_samples,
                    'direction': sum(m['direction'] for m in metrics) / total_samples,
                    'navigation_instructions': sum(m['navigation_instructions'] for m in metrics) / total_samples,
                    'hand_guidance': sum(m['hand_guidance'] for m in metrics) / total_samples,
                    'fallback': sum(m['fallback'] for m in metrics) / total_samples,
                    'total': sum(m['total_score'] for m in metrics) / total_samples,
                },
                'max_possible_scores': EVALUATION_METRICS,
                'total_max_possible': sum(EVALUATION_METRICS.values()),
                'overall_percentage': (sum(m['total_score'] for m in metrics) / (total_samples * sum(EVALUATION_METRICS.values())) * 100) if total_samples > 0 else 0,
                'individual_metrics': metrics
            }
        else:
            summary = {
                'total_samples': 0,
                'average_scores': {},
                'max_possible_scores': EVALUATION_METRICS,
                'overall_percentage': 0,
                'individual_metrics': []
            }

        # Save metrics summary
        metrics_filename = result_single_file.replace('.csv', '_metrics.json')
        metrics_path = os.path.join(RESULTS_FOLDER, metrics_filename)
        with open(metrics_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Metrics summary saved to {metrics_path}")

        logger.info("\n" + "="*70)
        logger.info("Evaluation Complete!")
        logger.info("="*70)


if __name__ == "__main__":
    main()
