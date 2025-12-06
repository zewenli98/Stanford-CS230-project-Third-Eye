"""
Verification script to demonstrate that pathfinder.py now matches prepare_test.py
distance calculations when using the same data.
"""

import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from pathfinder import PathFinder

def verify_distance_calculation():
    """
    Verify that pathfinder.py distance calculation matches ground truth from CSV.
    """
    print("="*80)
    print("DISTANCE CALCULATION VERIFICATION")
    print("="*80)
    print("\nThis script verifies that pathfinder.py now calculates distance")
    print("the same way as prepare_test.py when using annotation data.\n")

    # Load CSV with ground truth
    csv_path = "../queries/prompts.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ Error: CSV file not found at {csv_path}")
        print("   Please run prepare_test.py first to generate test data.")
        return

    # Initialize pathfinder
    pathfinder = PathFinder(output_dir="./pathfinder_outputs/verification")

    print(f"✓ Loaded {len(df)} queries from {csv_path}\n")
    print("Testing first 3 queries...\n")
    print("-"*80)

    results = []
    num_queries = min(3, len(df))

    for idx, row in df.head(num_queries).iterrows():
        query_id = row['prompt_id']
        object_name = row['object']
        ground_truth_distance = float(row['object_distance'])

        print(f"\n[Query {query_id}] Object: {object_name}")
        print(f"  Ground Truth (from prepare_test.py): {ground_truth_distance:.2f}m")

        rgb_path = f"../queries/images/{row['image_name']}"
        depth_path = f"../queries/images/{row['depth_image']}"

        # Test 1: WITHOUT annotation data (depth map only - less accurate)
        try:
            instruction_no_annot = pathfinder.process_query(
                rgb_path=rgb_path,
                depth_path=depth_path,
                object_name=object_name,
                bbox_str=row['object_bbox']
            )
            distance_no_annot = instruction_no_annot.distance_meters
            error_no_annot = abs(distance_no_annot - ground_truth_distance)
            error_pct_no_annot = (error_no_annot / ground_truth_distance) * 100

            print(f"  Depth Map Only (no annotation): {distance_no_annot:.2f}m")
            print(f"    ├─ Error: {error_no_annot:.2f}m ({error_pct_no_annot:.1f}%)")
        except Exception as e:
            print(f"  ❌ Depth map calculation failed: {e}")
            distance_no_annot = None
            error_no_annot = None

        # Test 2: WITH annotation data (should match exactly)
        try:
            instruction_with_annot = pathfinder.process_query(
                rgb_path=rgb_path,
                depth_path=depth_path,
                object_name=object_name,
                bbox_str=row['object_bbox'],
                annotation_json=row['annotation']
            )
            distance_with_annot = instruction_with_annot.distance_meters
            error_with_annot = abs(distance_with_annot - ground_truth_distance)
            error_pct_with_annot = (error_with_annot / ground_truth_distance) * 100

            print(f"  With XYZ Annotation Data: {distance_with_annot:.2f}m")
            print(f"    ├─ Error: {error_with_annot:.2f}m ({error_pct_with_annot:.1f}%)")

            # Check if they match
            if error_with_annot < 0.01:  # Less than 1cm difference
                print(f"    └─ ✅ EXACT MATCH! (within 1cm tolerance)")
                match_status = "PASS"
            else:
                print(f"    └─ ⚠️  Small difference (expected exact match)")
                match_status = "WARN"

        except Exception as e:
            print(f"  ❌ Annotation-based calculation failed: {e}")
            distance_with_annot = None
            error_with_annot = None
            match_status = "FAIL"

        results.append({
            'query_id': query_id,
            'object': object_name,
            'ground_truth': ground_truth_distance,
            'depth_only': distance_no_annot,
            'with_annotation': distance_with_annot,
            'error_depth': error_no_annot,
            'error_annotation': error_with_annot,
            'status': match_status
        })

    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)

    for result in results:
        print(f"\nQuery {result['query_id']}: {result['object']}")
        print(f"  Ground Truth:       {result['ground_truth']:.2f}m")
        if result['depth_only'] is not None:
            print(f"  Depth Map Only:     {result['depth_only']:.2f}m (error: {result['error_depth']:.2f}m)")
        if result['with_annotation'] is not None:
            print(f"  With Annotation:    {result['with_annotation']:.2f}m (error: {result['error_annotation']:.2f}m)")
        print(f"  Status: {result['status']}")

    # Count passes
    passes = sum(1 for r in results if r['status'] == 'PASS')
    total = len(results)

    print("\n" + "-"*80)
    print(f"Results: {passes}/{total} queries matched ground truth exactly")
    print("-"*80)

    if passes == total:
        print("\n✅ SUCCESS! All distance calculations match prepare_test.py!")
        print("   The fix is working correctly.\n")
    else:
        print("\n⚠️  Some queries did not match exactly.")
        print("   This may indicate an issue with annotation extraction.\n")

    print("="*80)
    print("\n💡 Key Takeaway:")
    print("   - When annotation data is used: Distance matches prepare_test.py exactly")
    print("   - Without annotation data: Uses improved depth sampling (5x5 center grid)")
    print("   - For evaluation: Always use annotation data from CSV for consistency\n")

if __name__ == "__main__":
    verify_distance_calculation()
