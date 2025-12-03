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
    is_fetchable: bool  # True if distance < 0.5m
    waypoints: Optional[List[Dict]] = None  # List of {"direction": str, "distance": float}
    warnings: Optional[List[str]] = None


class PathFinder:
    """
    PathFinder for blind navigation assistance.
    Analyzes RGB-D images to provide navigation instructions.
    """

    # Constants
    ARM_REACH_DISTANCE = 0.5  # meters (human arm fetchable distance)
    SAFE_DISTANCE_THRESHOLD = 0.3  # meters (minimum safe distance from obstacles)
    GRID_RESOLUTION = 0.05  # meters per grid cell in BEV
    IMAGE_FOV_HORIZONTAL = 58.0  # degrees (typical RGB-D camera FOV)
    BEV_SIZE = 8.0  # meters (8m x 8m BEV coverage)
    BEV_RESOLUTION = 160  # pixels (160x160 grid = 5cm per pixel)

    def __init__(self, output_dir: str = "./Agents/pathfinder_outputs"):
        """
        Initialize PathFinder.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def read_depth_image(self, depth_path: Path) -> np.ndarray:
        # 读取原始 16-bit 数据
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Could not read depth image: {depth_path}")

        # SUN RGB-D 特别处理
        if depth.dtype == np.uint16:
            # 1. 先把数据转换成 float32 容器（或者先在 uint16 上做位操作）
            # 推荐先做位运算，再转 float，这样更干净
            # ">> 3" 去掉低3位的 PlayerID，同时相当于除以 8
            depth_mm = (depth >> 3).astype(np.float32)
            
            # 2. 转换单位 mm -> m
            d = depth_mm / 1000.0

            # 3. 过滤逻辑
            # SUN RGB-D 的有效深度通常在 8m 以内
            # 0 或者是某些极小值通常是无效点
            d[(d < 0.1) | (d > 8.0)] = 0.0
            
            return d

        # 如果是 float (已经处理过的数据)，通常不需要位移
        elif depth.dtype == np.float32 or depth.dtype == np.float64:
            d = depth.astype(np.float32)
            d[~np.isfinite(d)] = 0.0
            d[(d <= 0) | (d > 10.0)] = 0.0
            return d
            
        else:
             raise ValueError(f"Unsupported dtype: {depth.dtype}")

    def depth_to_bev(self, depth_map: np.ndarray, intrinsics: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Convert depth map to Bird's Eye View (BEV) occupancy grid.

        Args:
            depth_map: Depth image in meters (H x W)
            intrinsics: Optional camera intrinsics matrix (3x3)
                       If None, uses estimated values

        Returns:
            BEV occupancy grid (160x160) where:
            - 0 = free space
            - 1 = occupied (obstacle)
            - Center of grid is camera position
        """
        h, w = depth_map.shape

        # Use default intrinsics if not provided (typical RGB-D camera)
        if intrinsics is None:
            fx = fy = 525.0  # Focal length (typical for Kinect/RealSense)
            cx = w / 2.0
            cy = h / 2.0
        else:
            fx = intrinsics[0, 0]
            fy = intrinsics[1, 1]
            cx = intrinsics[0, 2]
            cy = intrinsics[1, 2]

        # Initialize BEV grid
        bev_grid = np.zeros((self.BEV_RESOLUTION, self.BEV_RESOLUTION), dtype=np.uint8)

        # Grid parameters
        grid_size = self.BEV_SIZE  # 8m x 8m
        grid_res = self.BEV_RESOLUTION  # 160x160 pixels
        cell_size = grid_size / grid_res  # 0.05m per pixel

        # Camera is at the center of BEV
        cam_x = grid_res // 2
        cam_y = grid_res // 2

        # Convert each pixel to 3D point and project to BEV
        for v in range(0, h, 4):  # Subsample for efficiency
            for u in range(0, w, 4):
                depth = depth_map[v, u]

                if depth <= 0 or depth > grid_size / 2:
                    continue

                # Convert to 3D camera coordinates
                # X: right, Y: down, Z: forward (depth)
                Z = depth
                X = (u - cx) * Z / fx
                Y = (v - cy) * Z / fy

                # Convert to BEV coordinates
                # In BEV: X is right, Z is forward
                # Map to grid: (0,0) is top-left
                bev_x = int(cam_x + X / cell_size)
                bev_z = int(cam_y - Z / cell_size)  # Z maps to y-axis in BEV

                # Check bounds
                if 0 <= bev_x < grid_res and 0 <= bev_z < grid_res:
                    # Mark as occupied if object is a ground-level obstacle
                    # Y > 0 means below camera, Y values 0.5-2.0 are typical for ground objects
                    # Only mark obstacles that are on the floor and would block path
                    is_ground_obstacle = (Y > 0.5 and Y < 2.0)  # Ground level objects
                    is_too_close = depth < self.SAFE_DISTANCE_THRESHOLD  # Very close obstacles

                    if is_too_close or is_ground_obstacle:
                        bev_grid[bev_z, bev_x] = 1

        return bev_grid

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
                                  bbox: Tuple[int, int, int, int],
                                  annotation_xyz: Optional[np.ndarray] = None) -> float:
        """
        Calculate distance to object using depth map and bounding box.

        This method matches the calculation in prepare_test.py for consistency.
        When XYZ annotation data is available, it uses mean Z-coordinate (ground truth).
        Otherwise, it samples depth at the bbox center for better accuracy.

        Args:
            depth_map: Depth image in meters
            bbox: Bounding box (x1, y1, x2, y2)
            annotation_xyz: Optional XYZ coordinates from annotations (N x 3 array)
                          If provided, uses mean Z-coordinate like prepare_test.py

        Returns:
            Distance in meters
        """
        # If XYZ annotation data is available, use it (matches prepare_test.py)
        if annotation_xyz is not None:
            # Calculate 3D center position - same as prepare_test.py line 312
            center_3d = np.mean(annotation_xyz, axis=0)  # [x, y, z]
            x_cam, y_cam, z_cam = center_3d
            # Distance is z-coordinate in camera frame (prepare_test.py line 316)
            distance_meters = float(z_cam)
            logger.info(f"Using annotation XYZ data: distance = {distance_meters:.2f}m")
            return distance_meters

        x1, y1, x2, y2 = bbox

        # Ensure bbox is within image bounds
        h, w = depth_map.shape
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        # Calculate bbox center
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Sample depth at center and nearby points for robustness
        # This is more accurate than median of entire bbox which includes background
        sample_points = []
        for dy in range(-2, 3):  # 5x5 grid around center
            for dx in range(-2, 3):
                py = center_y + dy
                px = center_x + dx
                if 0 <= px < w and 0 <= py < h:
                    depth_val = depth_map[py, px]
                    if depth_val > 0:
                        sample_points.append(depth_val)

        if len(sample_points) == 0:
            logger.warning("No valid depth values at bbox center")
            # Fallback to bbox region
            bbox_depth = depth_map[y1:y2, x1:x2]
            valid_depth = bbox_depth[bbox_depth > 0]
            if len(valid_depth) == 0:
                return -1.0
            distance = np.median(valid_depth)
        else:
            # Use mean of center samples (similar to XYZ mean approach)
            distance = np.mean(sample_points)

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

    def object_to_bev_coords(self, depth_map: np.ndarray, bbox: Tuple[int, int, int, int],
                            intrinsics: Optional[np.ndarray] = None) -> Tuple[int, int]:
        """
        Convert object bounding box to BEV coordinates.

        Args:
            depth_map: Depth map in meters
            bbox: Object bounding box (x1, y1, x2, y2)
            intrinsics: Optional camera intrinsics

        Returns:
            (bev_x, bev_y) coordinates in BEV grid
        """
        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape

        # Get object center in image
        center_u = (x1 + x2) // 2
        center_v = (y1 + y2) // 2

        # Clamp to image bounds
        center_u = max(0, min(center_u, w - 1))
        center_v = max(0, min(center_v, h - 1))

        # Get depth at center
        depth = depth_map[center_v, center_u]

        if depth <= 0:
            # Fallback to average depth in bbox
            bbox_depth = depth_map[y1:y2, x1:x2]
            valid_depth = bbox_depth[bbox_depth > 0]
            if len(valid_depth) > 0:
                depth = np.mean(valid_depth)
            else:
                depth = 2.0  # Default 2m

        # Use default intrinsics if not provided
        if intrinsics is None:
            fx = fy = 525.0
            cx = w / 2.0
            cy = h / 2.0
        else:
            fx = intrinsics[0, 0]
            fy = intrinsics[1, 1]
            cx = intrinsics[0, 2]
            cy = intrinsics[1, 2]

        # Convert to 3D camera coordinates
        Z = depth
        X = (center_u - cx) * Z / fx

        # Convert to BEV coordinates
        grid_res = self.BEV_RESOLUTION
        cell_size = self.BEV_SIZE / grid_res
        cam_x = grid_res // 2
        cam_y = grid_res // 2

        bev_x = int(cam_x + X / cell_size)
        bev_z = int(cam_y - Z / cell_size)

        # Clamp to grid bounds
        bev_x = max(0, min(bev_x, grid_res - 1))
        bev_z = max(0, min(bev_z, grid_res - 1))

        return bev_x, bev_z

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

    def generate_waypoints(self, path: List[Tuple[int, int]]) -> List[Dict]:
        """
        Convert BEV path to waypoints with clock-based directions.

        Args:
            path: List of (x, y) positions in BEV grid coordinates

        Returns:
            List of waypoint dictionaries with format:
            {"direction": "2 o'clock", "distance": 2.5}
        """
        if not path or len(path) < 2:
            return []

        cell_size = self.BEV_SIZE / self.BEV_RESOLUTION  # 0.05m per cell

        # Simplified approach: Create waypoints by grouping consecutive path segments
        # with the same clock direction
        waypoints = []

        # Start from beginning of path
        i = 0
        while i < len(path) - 1:
            # Calculate direction from current point to next
            start_point = path[i]

            # Find how far we can go in the same general direction
            j = i + 1
            current_clock = None

            while j < len(path):
                dx = path[j][0] - start_point[0]
                dy = path[j][1] - start_point[1]

                # Calculate angle and clock direction
                angle = np.arctan2(dx, -dy) * 180 / np.pi
                clock = self.angle_to_clock(angle)

                if current_clock is None:
                    current_clock = clock
                elif clock != current_clock:
                    # Direction changed, end this segment
                    break

                j += 1

            # Create waypoint for this segment
            if j >= len(path):
                j = len(path) - 1

            end_point = path[j]
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            distance = np.sqrt(dx**2 + dy**2) * cell_size

            # Only add waypoint if distance is significant (> 0.3m)
            if distance >= 0.3:
                angle = np.arctan2(dx, -dy) * 180 / np.pi
                clock_direction = self.angle_to_clock(angle)

                waypoints.append({
                    "direction": clock_direction,
                    "distance": round(distance, 2)
                })

            # Move to next segment
            i = max(j, i + 1)  # Ensure we make progress

        # Merge consecutive waypoints with the same direction
        merged_waypoints = []
        for waypoint in waypoints:
            if merged_waypoints and merged_waypoints[-1]["direction"] == waypoint["direction"]:
                # Merge with previous waypoint
                merged_waypoints[-1]["distance"] += waypoint["distance"]
                merged_waypoints[-1]["distance"] = round(merged_waypoints[-1]["distance"], 2)
            else:
                merged_waypoints.append(waypoint)

        return merged_waypoints

    def angle_to_clock(self, angle_degrees: float) -> str:
        """
        Convert angle (in degrees) to clock position.

        Args:
            angle_degrees: Angle in degrees where:
                          0° = 12 o'clock (straight ahead)
                          90° = 3 o'clock (right)
                          -90° = 9 o'clock (left)

        Returns:
            Clock position string (e.g., "2 o'clock", "10 o'clock")
        """
        # Normalize angle to 0-360 range
        angle = angle_degrees % 360

        # Map to clock position (12 positions)
        # Each hour represents 30 degrees
        clock_hour = int((angle + 15) / 30) % 12
        if clock_hour == 0:
            clock_hour = 12

        # Determine precision label
        if -15 <= angle_degrees <= 15:
            return "12 o'clock (straight ahead)"
        elif 75 <= angle_degrees <= 105:
            return "3 o'clock (right)"
        elif angle_degrees <= -165 or angle_degrees >= 165:
            return "6 o'clock (behind)"
        elif -105 <= angle_degrees <= -75:
            return "9 o'clock (left)"
        else:
            return f"{clock_hour} o'clock"

    def extract_xyz_from_annotation(self, annotation_json: str, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Extract XYZ coordinates from annotation JSON for the object at given bbox.

        Args:
            annotation_json: JSON string containing annotation data
            bbox: Bounding box to match against polygons

        Returns:
            XYZ coordinates as numpy array (N x 3), or None if not found
        """
        try:
            annotation = json.loads(annotation_json)
            if 'frames' not in annotation or not annotation['frames']:
                return None

            polygons = annotation['frames'][0].get('polygon', [])

            # Find polygon matching the bbox
            for poly in polygons:
                if 'x' in poly and 'y' in poly and 'XYZ' in poly:
                    poly_x = poly['x']
                    poly_y = poly['y']
                    poly_bbox = [min(poly_x), min(poly_y), max(poly_x), max(poly_y)]

                    # Check if this polygon matches our bbox (with some tolerance)
                    if (abs(poly_bbox[0] - bbox[0]) < 10 and
                        abs(poly_bbox[1] - bbox[1]) < 10 and
                        abs(poly_bbox[2] - bbox[2]) < 10 and
                        abs(poly_bbox[3] - bbox[3]) < 10):

                        xyz_data = poly['XYZ']
                        # Convert to numpy array and filter out empty/zero entries
                        xyz_array = np.array(xyz_data)
                        if xyz_array.size > 0 and not np.allclose(xyz_array, 0):
                            return xyz_array

            return None
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to extract XYZ from annotation: {e}")
            return None

    def process_query(self, rgb_path: str, depth_path: str,
                     object_name: str, bbox_str: str,
                     annotation_json: Optional[str] = None) -> NavigationInstruction:
        """
        Process a single navigation query.

        Args:
            rgb_path: Path to RGB image
            depth_path: Path to depth image
            object_name: Name of the object
            bbox_str: Bounding box string "[x1, y1, x2, y2]"
            annotation_json: Optional JSON string with annotation data (from CSV)
                           If provided, extracts XYZ coordinates for accurate distance

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

        # Extract XYZ annotation data if available
        annotation_xyz = None
        if annotation_json:
            annotation_xyz = self.extract_xyz_from_annotation(annotation_json, bbox)
            if annotation_xyz is not None:
                logger.info("Using XYZ annotation data for distance calculation")

        # Calculate distance (matches prepare_test.py when annotation_xyz is provided)
        distance = self.calculate_object_distance(depth_map, bbox, annotation_xyz)

        # Calculate direction
        h, w = rgb_img.shape[:2]
        direction_clock, direction_degrees = self.calculate_direction(bbox, w)

        # Check if fetchable (< 0.5m)
        is_fetchable = distance < self.ARM_REACH_DISTANCE and distance > 0

        # Generate waypoints if not directly fetchable
        waypoints = None
        warnings = []

        if not is_fetchable and distance > 0:
            logger.info("Object not fetchable, generating BEV path...")

            # Convert depth map to BEV
            bev_grid = self.depth_to_bev(depth_map)

            # Get object position in BEV
            obj_bev_x, obj_bev_y = self.object_to_bev_coords(depth_map, bbox)

            # User position (center of BEV)
            user_x = self.BEV_RESOLUTION // 2
            user_y = self.BEV_RESOLUTION // 2

            # Find path using A* in BEV
            path = self.astar(bev_grid, (user_x, user_y), (obj_bev_x, obj_bev_y))

            if path and len(path) > 1:
                # Generate waypoints from path
                waypoints = self.generate_waypoints(path)
                logger.info(f"Generated {len(waypoints)} waypoints")
            else:
                warnings.append("Could not find safe path to object")
                logger.warning("No path found in BEV")

        if distance <= 0:
            warnings.append("Invalid depth data for object")

        # Create instruction
        instruction = NavigationInstruction(
            object_name=object_name,
            distance_meters=round(distance, 2),
            direction_clock=direction_clock,
            direction_degrees=round(direction_degrees, 2),
            is_fetchable=is_fetchable,
            waypoints=waypoints if waypoints else None,
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

        # Waypoints visualization
        if instruction.waypoints and len(instruction.waypoints) > 0:
            # Create text summary of waypoints
            waypoint_text = "Waypoints:\n"
            for i, wp in enumerate(instruction.waypoints, 1):
                waypoint_text += f"{i}. {wp['direction']}: {wp['distance']}m\n"

            axes[2].text(0.1, 0.9, waypoint_text,
                        ha='left', va='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                        verticalalignment='top',
                        family='monospace')
            axes[2].set_title('Navigation Waypoints')
            axes[2].set_xlim([0, 1])
            axes[2].set_ylim([0, 1])
        else:
            status_text = 'Object is fetchable!\nNo navigation needed' if instruction.is_fetchable else 'No path found'
            axes[2].text(0.5, 0.5, status_text,
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
    parser.add_argument('--csv_file', type=str, default='../queries/prompts.csv',
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
        print(f"Fetchable: <isFetchable>{instruction.is_fetchable}</isFetchable>")

        if instruction.is_fetchable:
            print("\n✓ Object is within reach! You can fetch it directly.")
        elif instruction.waypoints:
            print(f"\n📍 Navigation Waypoints ({len(instruction.waypoints)} waypoints):")
            for i, waypoint in enumerate(instruction.waypoints, 1):
                print(f"  {i}. Direction: {waypoint['direction']}, Distance: {waypoint['distance']}m")
        else:
            print("\n⚠ No path available to object")

        if instruction.warnings:
            print(f"\nWarnings:")
            for warning in instruction.warnings:
                print(f"  - {warning}")

        print("="*80)


if __name__ == "__main__":
    main()
