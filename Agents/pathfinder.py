"""
Pathfinder Module for Blind Navigation
Processes RGB images, depth maps, and object bounding boxes to generate
navigation instructions including distance, direction, reachable positions, and safe paths.
"""

import os
import sys
from pathlib import Path
import numpy as np
import cv2
import json
import logging
from typing import Dict, List, Tuple, Optional
import argparse
import pandas as pd
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class NavigationInstruction:
    """Navigation instruction data structure."""
    object_name: str
    distance_meters: float
    direction_clock: str
    direction_degrees: float
    is_reachable: bool
    reachable_position: Optional[Dict[str, int]] = None
    safe_path: Optional[List[Dict]] = None
    warnings: Optional[List[str]] = None


class PathFinder:
    """
    PathFinder for blind navigation assistance.
    Analyzes RGB-D images to provide navigation instructions.
    """

    # Constants
    ARM_REACH_DISTANCE = 0.8  # meters (typical arm reach)
    SAFE_DISTANCE_THRESHOLD = 0.3  # meters (minimum safe distance from obstacles)
    GRID_RESOLUTION = 0.1  # meters per grid cell
    IMAGE_FOV_HORIZONTAL = 58.0  # degrees (typical RGB-D camera FOV)

    def __init__(self, output_dir: str = "./Agents/pathfinder_outputs"):
        """
        Initialize PathFinder.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def read_depth_image(self, depth_path: Path) -> np.ndarray:
        """
        Read depth image and convert to meters.

        Args:
            depth_path: Path to depth image

        Returns:
            Depth array in meters
        """
        # Read depth image (16-bit or 8-bit)
        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)

        if depth_img is None:
            raise ValueError(f"Could not read depth image: {depth_path}")

        # Convert to float and normalize
        if depth_img.dtype == np.uint16:
            # 16-bit depth image (0-65535)
            depth_meters = depth_img.astype(np.float32) / 1000.0
        else:
            # 8-bit depth image (normalized 0-255)
            depth_meters = (depth_img.astype(np.float32) / 255.0) * 10.0

        # Clamp invalid values
        depth_meters[depth_meters <= 0] = 0
        depth_meters[depth_meters > 10] = 10

        return depth_meters

    def parse_bbox(self, bbox_str: str) -> Tuple[int, int, int, int]:
        """
        Parse bounding box from string.

        Args:
            bbox_str: String like "[x1, y1, x2, y2]"

        Returns:
            Tuple of (x1, y1, x2, y2)
        """
        # Remove brackets and split
        bbox_str = bbox_str.strip('[]')
        coords = [int(x.strip()) for x in bbox_str.split(',')]

        if len(coords) != 4:
            raise ValueError(f"Invalid bbox format: {bbox_str}")

        return tuple(coords)

    def calculate_object_distance(self, depth_map: np.ndarray,
                                  bbox: Tuple[int, int, int, int]) -> float:
        """
        Calculate distance to object using depth map and bounding box.

        Args:
            depth_map: Depth image in meters
            bbox: Bounding box (x1, y1, x2, y2)

        Returns:
            Distance in meters
        """
        x1, y1, x2, y2 = bbox

        # Ensure bbox is within image bounds
        h, w = depth_map.shape
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        # Extract depth values in bounding box
        bbox_depth = depth_map[y1:y2, x1:x2]

        # Remove invalid depth values
        valid_depth = bbox_depth[bbox_depth > 0]

        if len(valid_depth) == 0:
            logger.warning("No valid depth values in bounding box")
            return -1.0

        # Use median depth to avoid outliers
        distance = np.median(valid_depth)

        return float(distance)

    def calculate_direction(self, bbox: Tuple[int, int, int, int],
                           image_width: int) -> Tuple[str, float]:
        """
        Calculate direction of object using clock position and degrees.

        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            image_width: Width of image

        Returns:
            Tuple of (clock_position, degrees_from_center)
        """
        x1, y1, x2, y2 = bbox

        # Calculate center of bounding box
        bbox_center_x = (x1 + x2) / 2
        image_center_x = image_width / 2

        # Calculate horizontal offset from center
        offset_pixels = bbox_center_x - image_center_x

        # Convert to degrees (assuming standard FOV)
        offset_ratio = offset_pixels / image_width
        degrees_from_center = offset_ratio * self.IMAGE_FOV_HORIZONTAL

        # Convert to clock position
        # 12 o'clock = 0°, 3 o'clock = 90°, etc.
        clock_position = self.degrees_to_clock(degrees_from_center)

        return clock_position, degrees_from_center

    def degrees_to_clock(self, degrees: float) -> str:
        """
        Convert degrees to clock position.

        Args:
            degrees: Degrees from center (-29 to +29 for typical FOV)

        Returns:
            Clock position string
        """
        # Map degrees to clock position
        # Negative = left, Positive = right

        if degrees < -20:
            return "9 o'clock (far left)"
        elif degrees < -10:
            return "10 o'clock (left)"
        elif degrees < -3:
            return "11 o'clock (slightly left)"
        elif degrees <= 3:
            return "12 o'clock (straight ahead)"
        elif degrees <= 10:
            return "1 o'clock (slightly right)"
        elif degrees <= 20:
            return "2 o'clock (right)"
        else:
            return "3 o'clock (far right)"

    def find_reachable_position(self, depth_map: np.ndarray,
                                bbox: Tuple[int, int, int, int],
                                target_distance: float) -> Optional[Dict[str, int]]:
        """
        Find a reachable position near the object.

        Args:
            depth_map: Depth image in meters
            bbox: Object bounding box
            target_distance: Target distance to reach

        Returns:
            Dictionary with x, y pixel coordinates of reachable position
        """
        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape

        # Center of object
        obj_center_x = (x1 + x2) // 2
        obj_center_y = (y1 + y2) // 2

        # Search for positions within arm's reach
        search_radius = 50  # pixels

        best_position = None
        min_distance_diff = float('inf')

        for dy in range(-search_radius, search_radius, 5):
            for dx in range(-search_radius, search_radius, 5):
                px = obj_center_x + dx
                py = obj_center_y + dy

                # Check bounds
                if px < 0 or px >= w or py < 0 or py >= h:
                    continue

                # Get depth at this position
                depth = depth_map[py, px]

                if depth <= 0:
                    continue

                # Check if this position is close to target distance
                distance_diff = abs(depth - target_distance)

                if distance_diff < min_distance_diff:
                    min_distance_diff = distance_diff
                    best_position = {'x': int(px), 'y': int(py)}

        return best_position

    def create_occupancy_grid(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Create occupancy grid from depth map for path planning.

        Args:
            depth_map: Depth image in meters

        Returns:
            Occupancy grid (0 = free, 1 = occupied)
        """
        # Resize depth map to grid resolution
        h, w = depth_map.shape
        grid_h = h // 10  # Downsample for efficiency
        grid_w = w // 10

        # Create grid
        grid = np.zeros((grid_h, grid_w), dtype=np.uint8)

        # Mark obstacles (too close or invalid depth)
        for gy in range(grid_h):
            for gx in range(grid_w):
                # Sample depth at this grid cell
                py = min(gy * 10, h - 1)
                px = min(gx * 10, w - 1)

                depth = depth_map[py, px]

                # Mark as obstacle if too close or no depth
                if depth <= 0 or depth < self.SAFE_DISTANCE_THRESHOLD:
                    grid[gy, gx] = 1

        return grid

    def find_safe_path(self, depth_map: np.ndarray,
                      start_pos: Dict[str, int],
                      goal_pos: Dict[str, int]) -> List[Dict]:
        """
        Find safe path using A* algorithm.

        Args:
            depth_map: Depth image in meters
            start_pos: Start position {x, y}
            goal_pos: Goal position {x, y}

        Returns:
            List of waypoints as dictionaries with x, y, action
        """
        # Create occupancy grid
        grid = self.create_occupancy_grid(depth_map)
        grid_h, grid_w = grid.shape

        # Convert positions to grid coordinates
        start_gx = start_pos['x'] // 10
        start_gy = start_pos['y'] // 10
        goal_gx = goal_pos['x'] // 10
        goal_gy = goal_pos['y'] // 10

        # Clamp to grid bounds
        start_gx = max(0, min(start_gx, grid_w - 1))
        start_gy = max(0, min(start_gy, grid_h - 1))
        goal_gx = max(0, min(goal_gx, grid_w - 1))
        goal_gy = max(0, min(goal_gy, grid_h - 1))

        # A* pathfinding
        path = self.astar(grid, (start_gx, start_gy), (goal_gx, goal_gy))

        if path is None:
            logger.warning("No safe path found")
            return []

        # Convert path to navigation steps
        steps = self.path_to_steps(path, depth_map)

        return steps

    def astar(self, grid: np.ndarray, start: Tuple[int, int],
             goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        A* pathfinding algorithm.

        Args:
            grid: Occupancy grid
            start: Start position (gx, gy)
            goal: Goal position (gx, gy)

        Returns:
            List of grid positions or None if no path found
        """
        from heapq import heappush, heappop

        h, w = grid.shape

        # Initialize
        open_set = []
        heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            current = heappop(open_set)[1]

            if current == goal:
                return self.reconstruct_path(came_from, current)

            # Explore neighbors
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                nx, ny = neighbor

                # Check bounds
                if nx < 0 or nx >= w or ny < 0 or ny >= h:
                    continue

                # Check obstacle
                if grid[ny, nx] == 1:
                    continue

                # Calculate cost
                tentative_g = g_score[current] + (1.414 if abs(dx) + abs(dy) == 2 else 1)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Euclidean distance heuristic."""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def reconstruct_path(self, came_from: Dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from came_from map."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def path_to_steps(self, path: List[Tuple[int, int]],
                     depth_map: np.ndarray) -> List[Dict]:
        """
        Convert grid path to navigation steps.

        Args:
            path: List of grid positions
            depth_map: Depth map for distance calculation

        Returns:
            List of step dictionaries
        """
        steps = []
        h, w = depth_map.shape

        for i in range(len(path) - 1):
            curr = path[i]
            next = path[i + 1]

            # Convert to pixel coordinates
            px = min(curr[0] * 10, w - 1)
            py = min(curr[1] * 10, h - 1)

            # Get depth
            depth = depth_map[py, px]

            # Determine direction
            dx = next[0] - curr[0]
            dy = next[1] - curr[1]

            if dx > 0:
                direction = "right"
            elif dx < 0:
                direction = "left"
            elif dy > 0:
                direction = "down/forward"
            elif dy < 0:
                direction = "up/back"
            else:
                direction = "stay"

            # Calculate step distance
            step_distance = np.sqrt(dx**2 + dy**2) * self.GRID_RESOLUTION

            steps.append({
                'step': i + 1,
                'direction': direction,
                'distance_meters': round(float(depth), 2),
                'action': f"Move {direction} for {step_distance:.1f}m"
            })

        # Add final step
        steps.append({
            'step': len(steps) + 1,
            'direction': 'arrived',
            'distance_meters': 0.0,
            'action': 'Reach for the object'
        })

        return steps

    def process_query(self, rgb_path: str, depth_path: str,
                     object_name: str, bbox_str: str) -> NavigationInstruction:
        """
        Process a single navigation query.

        Args:
            rgb_path: Path to RGB image
            depth_path: Path to depth image
            object_name: Name of the object
            bbox_str: Bounding box string "[x1, y1, x2, y2]"

        Returns:
            NavigationInstruction object
        """
        logger.info(f"Processing query for {object_name}")

        # Read images
        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None:
            raise ValueError(f"Could not read RGB image: {rgb_path}")

        depth_map = self.read_depth_image(Path(depth_path))

        # Parse bbox
        bbox = self.parse_bbox(bbox_str)

        # Calculate distance
        distance = self.calculate_object_distance(depth_map, bbox)

        # Calculate direction
        h, w = rgb_img.shape[:2]
        direction_clock, direction_degrees = self.calculate_direction(bbox, w)

        # Check if reachable
        is_reachable = distance <= self.ARM_REACH_DISTANCE and distance > 0

        # Find reachable position if not directly reachable
        reachable_pos = None
        safe_path = None
        warnings = []

        if not is_reachable and distance > 0:
            # Find position within arm's reach
            reachable_pos = self.find_reachable_position(
                depth_map, bbox, self.ARM_REACH_DISTANCE
            )

            if reachable_pos:
                # Find safe path
                start_pos = {'x': w // 2, 'y': h - 10}  # Bottom center (user position)
                safe_path = self.find_safe_path(depth_map, start_pos, reachable_pos)

                if not safe_path:
                    warnings.append("Could not find safe path to object")
            else:
                warnings.append("Could not find reachable position near object")

        if distance <= 0:
            warnings.append("Invalid depth data for object")

        # Create instruction
        instruction = NavigationInstruction(
            object_name=object_name,
            distance_meters=round(distance, 2),
            direction_clock=direction_clock,
            direction_degrees=round(direction_degrees, 2),
            is_reachable=is_reachable,
            reachable_position=reachable_pos,
            safe_path=safe_path if safe_path else None,
            warnings=warnings if warnings else None
        )

        return instruction

    def visualize_navigation(self, rgb_path: str, depth_path: str,
                            instruction: NavigationInstruction,
                            bbox_str: str, output_name: str):
        """
        Create visualization of navigation plan.

        Args:
            rgb_path: Path to RGB image
            depth_path: Path to depth image
            instruction: Navigation instruction
            bbox_str: Bounding box string
            output_name: Output file name
        """
        # Read images
        rgb_img = cv2.imread(rgb_path)
        depth_map = self.read_depth_image(Path(depth_path))
        bbox = self.parse_bbox(bbox_str)

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # RGB with bbox
        rgb_display = rgb_img.copy()
        x1, y1, x2, y2 = bbox
        cv2.rectangle(rgb_display, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(rgb_display, instruction.object_name, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        axes[0].imshow(cv2.cvtColor(rgb_display, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'{instruction.object_name}\nDistance: {instruction.distance_meters}m\nDirection: {instruction.direction_clock}')
        axes[0].axis('off')

        # Depth map
        axes[1].imshow(depth_map, cmap='plasma')
        axes[1].set_title('Depth Map')
        axes[1].axis('off')

        # Path visualization
        if instruction.safe_path and instruction.reachable_position:
            path_viz = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB).copy()

            # Draw path
            for i, step in enumerate(instruction.safe_path[:-1]):
                if i < len(instruction.safe_path) - 1:
                    # Path points are in step format, extract from grid
                    pass

            # Draw reachable position
            rx, ry = instruction.reachable_position['x'], instruction.reachable_position['y']
            cv2.circle(path_viz, (rx, ry), 10, (255, 0, 0), -1)
            cv2.putText(path_viz, 'Target', (rx + 15, ry),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            axes[2].imshow(path_viz)
            axes[2].set_title('Navigation Path')
        else:
            axes[2].text(0.5, 0.5, 'Object is reachable\nNo path needed' if instruction.is_reachable else 'No path found',
                        ha='center', va='center', fontsize=12)
            axes[2].set_xlim([0, 1])
            axes[2].set_ylim([0, 1])

        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / f'{output_name}_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Visualization saved to {self.output_dir / f'{output_name}_visualization.png'}")

    def save_json(self, instruction: NavigationInstruction, output_name: str):
        """
        Save navigation instruction as JSON.

        Args:
            instruction: Navigation instruction
            output_name: Output file name
        """
        output_path = self.output_dir / f'{output_name}.json'

        with open(output_path, 'w') as f:
            json.dump(asdict(instruction), f, indent=2)

        logger.info(f"JSON saved to {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="PathFinder for Blind Navigation")
    parser.add_argument('--rgb_image', type=str, help='Path to RGB image')
    parser.add_argument('--depth_image', type=str, help='Path to depth image')
    parser.add_argument('--object', type=str, help='Object name')
    parser.add_argument('--bbox', type=str, help='Bounding box "[x1, y1, x2, y2]"')
    parser.add_argument('--csv_file', type=str, default='./queries/prompts.csv',
                       help='CSV file with queries')
    parser.add_argument('--process_all', action='store_true',
                       help='Process all queries in CSV')
    parser.add_argument('--limit', type=int, default=10,
                       help='Limit number of queries to process')
    parser.add_argument('--output_dir', type=str, default='./Agents/pathfinder_outputs',
                       help='Output directory')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')

    args = parser.parse_args()

    # Create pathfinder
    pathfinder = PathFinder(output_dir=args.output_dir)

    if args.process_all:
        # Process CSV file
        logger.info(f"Loading queries from {args.csv_file}")
        df = pd.read_csv(args.csv_file)

        # Limit queries
        df = df.head(args.limit)

        logger.info(f"Processing {len(df)} queries...")

        for idx, row in df.iterrows():
            try:
                # Get paths
                rgb_path = f"./queries/images/{row['image_name']}"
                depth_path = f"./queries/images/{row['depth_image']}"

                # Process query
                instruction = pathfinder.process_query(
                    rgb_path=rgb_path,
                    depth_path=depth_path,
                    object_name=row['object'],
                    bbox_str=row['object_bbox']
                )

                # Save JSON
                output_name = f"query_{row['prompt_id']:04d}_{row['object']}"
                pathfinder.save_json(instruction, output_name)

                # Visualize if requested
                if args.visualize:
                    pathfinder.visualize_navigation(
                        rgb_path, depth_path, instruction,
                        row['object_bbox'], output_name
                    )

                logger.info(f"Processed query {row['prompt_id']}: {row['object']}")

            except Exception as e:
                logger.error(f"Error processing query {row['prompt_id']}: {e}")

    else:
        # Process single query
        if not all([args.rgb_image, args.depth_image, args.object, args.bbox]):
            logger.error("Please provide --rgb_image, --depth_image, --object, and --bbox")
            return

        instruction = pathfinder.process_query(
            rgb_path=args.rgb_image,
            depth_path=args.depth_image,
            object_name=args.object,
            bbox_str=args.bbox
        )

        # Save JSON
        pathfinder.save_json(instruction, f"{args.object}_navigation")

        # Visualize if requested
        if args.visualize:
            pathfinder.visualize_navigation(
                args.rgb_image, args.depth_image, instruction,
                args.bbox, f"{args.object}_navigation"
            )

        # Print result
        print("\n" + "="*80)
        print("NAVIGATION INSTRUCTION")
        print("="*80)
        print(f"Object: {instruction.object_name}")
        print(f"Distance: {instruction.distance_meters} meters")
        print(f"Direction: {instruction.direction_clock} ({instruction.direction_degrees}°)")
        print(f"Reachable: {'Yes' if instruction.is_reachable else 'No'}")

        if instruction.reachable_position:
            print(f"Reachable Position: ({instruction.reachable_position['x']}, {instruction.reachable_position['y']})")

        if instruction.safe_path:
            print(f"\nNavigation Steps ({len(instruction.safe_path)} steps):")
            for step in instruction.safe_path:
                print(f"  {step['step']}. {step['action']}")

        if instruction.warnings:
            print(f"\nWarnings:")
            for warning in instruction.warnings:
                print(f"  - {warning}")

        print("="*80)


if __name__ == "__main__":
    main()
