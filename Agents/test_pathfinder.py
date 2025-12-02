"""
Test script for PathFinder module
Demonstrates usage with example queries.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from pathfinder import PathFinder
import pandas as pd
import json

def test_single_query():
    """Test with a single query."""
    print("="*80)
    print("TEST 1: Single Query Processing")
    print("="*80)

    # Initialize pathfinder
    pathfinder = PathFinder(output_dir="./pathfinder_outputs/test1")

    # Example query data
    rgb_path = "./queries/images/0001_002599_2014-06-24_14-26-41_094959634447_rgbf000065-resize.jpg"
    depth_path = "./queries/images/0001_002599_2014-06-24_14-26-41_094959634447_rgbf000065-resize_depth.png"
    object_name = "Chair"
    bbox_str = "[407, 163, 614, 325]"

    try:
        # Process query
        instruction = pathfinder.process_query(
            rgb_path=rgb_path,
            depth_path=depth_path,
            object_name=object_name,
            bbox_str=bbox_str
        )

        # Print results
        print(f"\n✓ Object: {instruction.object_name}")
        print(f"✓ Distance: {instruction.distance_meters} meters")
        print(f"✓ Direction: {instruction.direction_clock} ({instruction.direction_degrees}°)")
        print(f"✓ Reachable: {'Yes' if instruction.is_reachable else 'No'}")

        if instruction.reachable_position:
            print(f"✓ Target Position: ({instruction.reachable_position['x']}, {instruction.reachable_position['y']})")

        if instruction.safe_path:
            print(f"\n✓ Navigation Path ({len(instruction.safe_path)} steps):")
            for step in instruction.safe_path[:5]:  # Show first 5 steps
                print(f"   {step['step']}. {step['action']}")
            if len(instruction.safe_path) > 5:
                print(f"   ... and {len(instruction.safe_path) - 5} more steps")

        if instruction.warnings:
            print(f"\n⚠ Warnings:")
            for warning in instruction.warnings:
                print(f"   - {warning}")

        # Save JSON
        pathfinder.save_json(instruction, "test_chair")
        print(f"\n✓ JSON saved successfully")

        # Generate visualization
        pathfinder.visualize_navigation(
            rgb_path, depth_path, instruction,
            bbox_str, "test_chair"
        )
        print(f"✓ Visualization saved successfully")

        return True

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("   Make sure the queries/images directory exists with image files")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def test_batch_processing():
    """Test batch processing from CSV."""
    print("\n" + "="*80)
    print("TEST 2: Batch Processing from CSV")
    print("="*80)

    # Initialize pathfinder
    pathfinder = PathFinder(output_dir="./pathfinder_outputs/test2")

    try:
        # Load CSV
        csv_path = "./queries/prompts.csv"
        df = pd.read_csv(csv_path)
        print(f"\n✓ Loaded {len(df)} queries from CSV")

        # Process first 3 queries
        num_queries = min(3, len(df))
        print(f"✓ Processing first {num_queries} queries...\n")

        results = []

        for idx, row in df.head(num_queries).iterrows():
            rgb_path = f"./queries/images/{row['image_name']}"
            depth_path = f"./queries/images/{row['depth_image']}"

            print(f"[{idx+1}/{num_queries}] Processing: {row['object']}")

            try:
                instruction = pathfinder.process_query(
                    rgb_path=rgb_path,
                    depth_path=depth_path,
                    object_name=row['object'],
                    bbox_str=row['object_bbox']
                )

                # Save results
                output_name = f"query_{row['prompt_id']:04d}_{row['object']}"
                pathfinder.save_json(instruction, output_name)

                results.append({
                    'query_id': row['prompt_id'],
                    'object': row['object'],
                    'distance': instruction.distance_meters,
                    'direction': instruction.direction_clock,
                    'reachable': instruction.is_reachable
                })

                print(f"   ✓ Distance: {instruction.distance_meters}m, Direction: {instruction.direction_clock}")

            except Exception as e:
                print(f"   ✗ Error: {e}")

        # Print summary
        print("\n" + "-"*80)
        print("BATCH PROCESSING SUMMARY")
        print("-"*80)

        for result in results:
            reachable_str = "✓ Reachable" if result['reachable'] else "✗ Not reachable"
            print(f"Query {result['query_id']}: {result['object']}")
            print(f"  Distance: {result['distance']}m | Direction: {result['direction']} | {reachable_str}")

        print(f"\n✓ Successfully processed {len(results)}/{num_queries} queries")

        return True

    except FileNotFoundError as e:
        print(f"\n✗ Error: File not found - {e}")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def test_json_output():
    """Test JSON output format."""
    print("\n" + "="*80)
    print("TEST 3: JSON Output Validation")
    print("="*80)

    try:
        # Read a generated JSON file
        json_path = "./pathfinder_outputs/test1/test_chair.json"

        if not Path(json_path).exists():
            print("✗ JSON file not found. Run Test 1 first.")
            return False

        with open(json_path, 'r') as f:
            data = json.load(f)

        # Validate required fields
        required_fields = [
            'object_name', 'distance_meters', 'direction_clock',
            'direction_degrees', 'is_reachable'
        ]

        print("\n✓ Validating JSON structure...")

        all_valid = True
        for field in required_fields:
            if field in data:
                print(f"   ✓ {field}: {data[field]}")
            else:
                print(f"   ✗ Missing field: {field}")
                all_valid = False

        # Check optional fields
        if data.get('reachable_position'):
            print(f"   ✓ reachable_position: {data['reachable_position']}")

        if data.get('safe_path'):
            print(f"   ✓ safe_path: {len(data['safe_path'])} steps")

        if data.get('warnings'):
            print(f"   ⚠ warnings: {data['warnings']}")

        print(f"\n✓ JSON validation {'passed' if all_valid else 'failed'}")

        # Pretty print JSON
        print("\n" + "-"*80)
        print("JSON OUTPUT:")
        print("-"*80)
        print(json.dumps(data, indent=2))

        return all_valid

    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("PATHFINDER MODULE TESTS")
    print("="*80)

    results = []

    # Test 1: Single query
    print("\nRunning Test 1...")
    results.append(("Single Query", test_single_query()))

    # Test 2: Batch processing
    print("\nRunning Test 2...")
    results.append(("Batch Processing", test_batch_processing()))

    # Test 3: JSON validation
    print("\nRunning Test 3...")
    results.append(("JSON Validation", test_json_output()))

    # Print final summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<50} {status}")

    total_passed = sum(1 for _, passed in results if passed)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")

    if total_passed == len(results):
        print("\n🎉 All tests passed! PathFinder is working correctly.")
    else:
        print("\n⚠ Some tests failed. Check the errors above.")

    print("="*80)


if __name__ == "__main__":
    main()
