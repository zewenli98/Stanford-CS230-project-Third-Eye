"""
Agent Test Bbox Preparation Script

Extracts prompt_id, object, and object_bbox from prompts.csv
and saves them to target_prediction.txt for agent testing.
"""

import os
import csv
import logging
from pathlib import Path

# Configure query path (update this as needed)
QUERY_PATH = "./queries/"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_bbox_targets(query_path: str, output_filename: str = "target_prediction.txt"):
    """
    Extract prompt_id, object, and object_bbox from prompts.csv.

    Args:
        query_path: Path to folder containing prompts.csv
        output_filename: Name of output file (default: target_prediction.txt)

    Returns:
        Number of entries extracted
    """
    logger.info("="*60)
    logger.info("EXTRACTING BBOX TARGETS FROM PROMPTS.CSV")
    logger.info("="*60)

    # Construct paths
    csv_path = os.path.join(query_path, "prompts.csv")
    output_path = os.path.join(query_path, output_filename)

    # Check if CSV exists
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    logger.info(f"Reading from: {csv_path}")

    # Read CSV and extract data
    extracted_data = []

    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            prompt_id = row.get('prompt_id', '').strip()
            obj = row.get('object', '').strip()
            bbox = row.get('object_bbox', '').strip()

            # Only add if all fields are present
            if prompt_id and obj and bbox:
                extracted_data.append({
                    'prompt_id': prompt_id,
                    'object': obj,
                    'object_bbox': bbox
                })
            else:
                logger.warning(f"Skipping row with missing data: prompt_id={prompt_id}, object={obj}, bbox={bbox}")

    logger.info(f"Extracted {len(extracted_data)} entries")

    # Write to output file
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for entry in extracted_data:
            line = f"{entry['prompt_id']},{entry['object']},{entry['object_bbox']}\n"
            outfile.write(line)

    logger.info(f"Saved to: {output_path}")
    logger.info(f"Format: prompt_id,object,object_bbox")
    logger.info("="*60)

    return len(extracted_data)


def main():
    """Main function."""
    # Check if QUERY_PATH is still placeholder
    if QUERY_PATH == "place holder":
        logger.warning("QUERY_PATH is still set to 'place holder'")
        logger.warning("Please update QUERY_PATH in the script before running")
        logger.warning("Example: QUERY_PATH = './queries/' or './queries-sample10/'")
        return

    # Extract bbox targets
    try:
        num_entries = extract_bbox_targets(QUERY_PATH)
        print(f"\n✓ Successfully extracted {num_entries} bbox targets")
        print(f"✓ Saved to: {os.path.join(QUERY_PATH, 'target_prediction.txt')}")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
