import os
import csv
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION - Configure these settings
# ============================================================================
RESULTS_CSV_FILE = "results_Qwen_Qwen2.5-VL-3B-Instruct_20251126_064826.csv"  # Name of the results CSV file to evaluate
RESULTS_FOLDER = "./results"  # Folder containing results files
# ============================================================================


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
    gt_scene: str
    gt_object_bbox: List[float]

    # Parsed LLM response fields
    found: Optional[bool] = None
    predicted_bbox: Optional[List[float]] = None
    predicted_distance_feet: Optional[float] = None
    predicted_distance_inches: Optional[float] = None
    predicted_distance_meters: Optional[float] = None
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
    Extract JSON from LLM response text.
    The response may contain markdown code blocks or be plain JSON.

    Args:
        response: Raw LLM response text

    Returns:
        Parsed JSON dictionary or None if parsing fails
    """
    # Try to find JSON in markdown code blocks
    json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    matches = re.findall(json_pattern, response, re.DOTALL)

    if matches:
        # Try to parse the first match
        try:
            return json.loads(matches[0])
        except json.JSONDecodeError:
            pass

    # Try to parse the entire response as JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try to find JSON without code blocks (look for { ... })
    json_pattern_no_block = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern_no_block, response, re.DOTALL)

    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    return None


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
    goal_object = row.get('object', '')
    goal_object_distance = float(row.get('object_distance', 0))
    goal_object_direction = row.get('object_direction', '')
    scene = row.get('scene', '')
    goal_object_bbox = row.get('object_bbox', '[]')
    annotation = row.get('annotation', '[]')
    depth_image = row.get('depth_image', '')

    # Create response object
    extracted = ExtractedResponse(
        prompt_id=prompt_id,
        prompt_text=prompt_text,
        image_name=image_name,
        model_name=model_name,
        raw_response=raw_response,
        goal_object=goal_object,
        goal_object_distance=goal_object_distance,
        goal_object_direction=goal_object_direction,
        scene=scene,
        goal_object_bbox=goal_object_bbox if goal_object_bbox else [],
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
        bbox = obj_location.get('bounding_box')
        if bbox and isinstance(bbox, list) and len(bbox) == 4:
            extracted.predicted_bbox = [float(x) for x in bbox]

        # Extract distance and direction
        distance_direction = json_data.get('distance_and_direction_from_camera', {})
        if distance_direction:
            extracted.predicted_distance_feet = distance_direction.get('distance_feet')
            extracted.predicted_distance_inches = distance_direction.get('distance_inches')
            extracted.predicted_direction = distance_direction.get('direction')

            # Convert to meters if available
            if extracted.predicted_distance_feet is not None:
                try:
                    extracted.predicted_distance_meters = float(extracted.predicted_distance_feet) * 0.3048
                except (ValueError, TypeError):
                    pass

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


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        bbox1: [x_min, y_min, x_max, y_max]
        bbox2: [x_min, y_min, x_max, y_max]

    Returns:
        IoU score between 0 and 1
    """
    if not bbox1 or not bbox2 or len(bbox1) != 4 or len(bbox2) != 4:
        return 0.0

    # Calculate intersection
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate union
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = bbox1_area + bbox2_area - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def calculate_distance_error(predicted_meters: Optional[float],
                             gt_meters: float) -> Optional[float]:
    """
    Calculate absolute distance error in meters.

    Args:
        predicted_meters: Predicted distance in meters
        gt_meters: Ground truth distance in meters

    Returns:
        Absolute error in meters or None if prediction is missing
    """
    if predicted_meters is None or gt_meters is None:
        return None

    return abs(predicted_meters - gt_meters)


def evaluate(extracted_responses: List[ExtractedResponse]) -> Dict[str, Any]:
    """
    Evaluate the extracted responses and calculate metrics.

    Args:
        extracted_responses: List of ExtractedResponse objects

    Returns:
        Dictionary containing evaluation metrics
    """
    # TODO: Implement evaluation logic here
    # This function will calculate various metrics such as:
    # - Object detection accuracy (found vs not found)
    # - Bounding box IoU
    # - Distance prediction error
    # - Direction prediction accuracy
    # etc.

    metrics = {
        'total_samples': len(extracted_responses),
        'successful_parses': sum(1 for r in extracted_responses if r.parse_success),
        'parse_rate': 0.0,
    }

    if metrics['total_samples'] > 0:
        metrics['parse_rate'] = metrics['successful_parses'] / metrics['total_samples']

    # Filter successfully parsed responses
    valid_responses = [r for r in extracted_responses if r.parse_success]

    if not valid_responses:
        logger.warning("No valid responses to evaluate")
        return metrics

    # Calculate object detection metrics
    found_count = sum(1 for r in valid_responses if r.found)
    metrics['detection_rate'] = found_count / len(valid_responses)

    # Calculate bounding box IoU
    iou_scores = []
    for response in valid_responses:
        if response.predicted_bbox and response.gt_object_bbox:
            iou = calculate_iou(response.predicted_bbox, response.gt_object_bbox)
            iou_scores.append(iou)

    if iou_scores:
        metrics['mean_iou'] = sum(iou_scores) / len(iou_scores)
        metrics['iou_scores'] = iou_scores
    else:
        metrics['mean_iou'] = 0.0
        metrics['iou_scores'] = []

    # Calculate distance prediction errors
    distance_errors = []
    for response in valid_responses:
        if response.predicted_distance_meters is not None and response.gt_object_distance is not None:
            error = calculate_distance_error(response.predicted_distance_meters,
                                            response.gt_object_distance)
            if error is not None:
                distance_errors.append(error)

    if distance_errors:
        metrics['mean_distance_error'] = sum(distance_errors) / len(distance_errors)
        metrics['distance_errors'] = distance_errors
    else:
        metrics['mean_distance_error'] = None
        metrics['distance_errors'] = []

    # Log evaluation results
    logger.info("="*70)
    logger.info("Evaluation Results:")
    logger.info("="*70)
    logger.info(f"Total samples: {metrics['total_samples']}")
    logger.info(f"Parse rate: {metrics['parse_rate']:.2%}")
    logger.info(f"Detection rate: {metrics.get('detection_rate', 0):.2%}")
    logger.info(f"Mean IoU: {metrics.get('mean_iou', 0):.4f}")
    if metrics['mean_distance_error'] is not None:
        logger.info(f"Mean distance error: {metrics['mean_distance_error']:.2f} meters")
    logger.info("="*70)

    return metrics


def save_evaluation_report(extracted_responses: List[ExtractedResponse],
                          metrics: Dict[str, Any],
                          output_path: str):
    """
    Save detailed evaluation report to CSV.

    Args:
        extracted_responses: List of ExtractedResponse objects
        metrics: Evaluation metrics dictionary
        output_path: Path to save the report CSV
    """
    logger.info(f"Saving evaluation report to {output_path}")

    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'prompt_id', 'image_name', 'model_name',
            'gt_object', 'gt_distance_m', 'gt_direction', 'gt_scene',
            'parse_success', 'found',
            'predicted_distance_m', 'distance_error_m',
            'gt_bbox', 'predicted_bbox', 'iou',
            'predicted_direction', 'parse_error'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for response in extracted_responses:
            # Calculate IoU if both bboxes available
            iou = None
            if response.predicted_bbox and response.gt_object_bbox:
                iou = calculate_iou(response.predicted_bbox, response.gt_object_bbox)

            # Calculate distance error
            dist_error = None
            if response.predicted_distance_meters and response.gt_object_distance:
                dist_error = abs(response.predicted_distance_meters - response.gt_object_distance)

            writer.writerow({
                'prompt_id': response.prompt_id,
                'image_name': response.image_name,
                'model_name': response.model_name,
                'gt_object': response.gt_object,
                'gt_distance_m': response.gt_object_distance,
                'gt_direction': response.gt_object_direction,
                'gt_scene': response.gt_scene,
                'parse_success': response.parse_success,
                'found': response.found,
                'predicted_distance_m': response.predicted_distance_meters,
                'distance_error_m': dist_error,
                'gt_bbox': json.dumps(response.gt_object_bbox) if response.gt_object_bbox else '',
                'predicted_bbox': json.dumps(response.predicted_bbox) if response.predicted_bbox else '',
                'iou': f"{iou:.4f}" if iou is not None else '',
                'predicted_direction': response.predicted_direction,
                'parse_error': response.parse_error or ''
            })

    logger.info(f"Evaluation report saved to {output_path}")


def main():
    """
    Main function to evaluate results.
    """
    # Construct full path to results CSV
    results_csv_path = os.path.join(RESULTS_FOLDER, RESULTS_CSV_FILE)

    logger.info("="*70)
    logger.info("Starting Evaluation")
    logger.info("="*70)
    logger.info(f"Results file: {results_csv_path}")

    # Extract all responses from CSV
    extracted_responses = extract_all_responses(results_csv_path)
    # pretty print key value pair of extracted_responses[0], which is key value pairs
    for key, value in extracted_responses[0].__dict__.items():
        print(f"{key}: {value}")
    # # Evaluate the responses
    # metrics = evaluate(extracted_responses)

    # # Save detailed evaluation report
    # report_filename = RESULTS_CSV_FILE.replace('.csv', '_evaluation.csv')
    # report_path = os.path.join(RESULTS_FOLDER, report_filename)
    # save_evaluation_report(extracted_responses, metrics, report_path)

    # # Save metrics summary
    # metrics_filename = RESULTS_CSV_FILE.replace('.csv', '_metrics.json')
    # metrics_path = os.path.join(RESULTS_FOLDER, metrics_filename)
    # with open(metrics_path, 'w') as f:
    #     json.dump(metrics, f, indent=2)
    # logger.info(f"Metrics summary saved to {metrics_path}")

    # logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
