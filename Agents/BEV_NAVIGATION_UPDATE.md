# BEV-Based Navigation System Update

## Overview

Updated PathFinder to use Bird's Eye View (BEV) based navigation with clock-based waypoint directions. This provides more intuitive navigation instructions for blind users.

## Major Changes

### 1. **New Fetchability Threshold**
- Changed from 0.8m to **0.5m** (ARM_REACH_DISTANCE)
- Output format: `<isFetchable>True</isFetchable>` or `<isFetchable>False</isFetchable>`

### 2. **BEV (Bird's Eye View) Transformation**
- New function: `depth_to_bev()` - Converts depth map to top-down occupancy grid
- Grid size: 8m x 8m coverage area
- Resolution: 160x160 pixels (5cm per pixel)
- Camera positioned at center of grid

### 3. **Clock-Based Waypoint System**
- **Old format**: Step-by-step actions ("Move forward 2m", "Turn right", etc.)
- **New format**: Waypoints with clock directions
  ```json
  {
    "direction": "2 o'clock",
    "distance": 2.5
  }
  ```

### 4. **Updated NavigationInstruction Dataclass**

**Before:**
```python
@dataclass
class NavigationInstruction:
    object_name: str
    distance_meters: float
    direction_clock: str
    direction_degrees: float
    is_reachable: bool
    reachable_position: Optional[Dict[str, int]] = None
    safe_path: Optional[List[Dict]] = None
    warnings: Optional[List[str]] = None
```

**After:**
```python
@dataclass
class NavigationInstruction:
    object_name: str
    distance_meters: float
    direction_clock: str
    direction_degrees: float
    is_fetchable: bool  # Changed from is_reachable
    waypoints: Optional[List[Dict]] = None  # Changed from safe_path
    warnings: Optional[List[str]] = None
```

## New Functions

### `depth_to_bev(depth_map, intrinsics) -> np.ndarray`
Converts depth map to Bird's Eye View occupancy grid.

**Key features:**
- Projects 3D points from depth map to 2D top-down view
- Uses camera intrinsics (or default values for Kinect/RealSense)
- Marks obstacles and free space
- Returns 160x160 grid centered on camera

**Example:**
```python
bev_grid = pathfinder.depth_to_bev(depth_map)
# Result: 160x160 grid where 0=free, 1=occupied
```

### `object_to_bev_coords(depth_map, bbox, intrinsics) -> (int, int)`
Converts object bounding box to BEV coordinates.

**Returns:** (bev_x, bev_y) position in BEV grid

### `generate_waypoints(path) -> List[Dict]`
Converts A* path in BEV to clock-based waypoints.

**Features:**
- Groups consecutive path segments with similar direction
- Calculates clock direction for each segment
- Returns list of waypoints with direction and distance

**Output format:**
```python
[
    {"direction": "12 o'clock (straight ahead)", "distance": 1.5},
    {"direction": "2 o'clock", "distance": 2.3},
    {"direction": "1 o'clock", "distance": 0.8}
]
```

### `angle_to_clock(angle_degrees) -> str`
Converts angle to clock position.

**Mapping:**
- 0° = 12 o'clock (straight ahead)
- 90° = 3 o'clock (right)
- -90° = 9 o'clock (left)
- 180° = 6 o'clock (behind)

**Returns:** Clock position like "2 o'clock", "10 o'clock", etc.

## Workflow Changes

### Old Workflow:
1. Calculate distance to object
2. Check if reachable (< 0.8m)
3. If not reachable, find path in image space
4. Generate step-by-step actions

### New Workflow:
1. Calculate distance to object
2. Check if fetchable (< 0.5m)
3. If not fetchable:
   - Convert depth map to BEV
   - Convert object position to BEV coordinates
   - Run A* pathfinding in BEV space
   - Generate clock-based waypoints
4. Return navigation instruction with waypoints

## Example Usage

### Input:
```python
pathfinder = PathFinder()
instruction = pathfinder.process_query(
    rgb_path='scene.jpg',
    depth_path='depth.png',
    object_name='Chair',
    bbox_str='[407, 163, 614, 325]'
)
```

### Output (Fetchable):
```python
NavigationInstruction(
    object_name='Chair',
    distance_meters=0.45,
    direction_clock='2 o'clock (right)',
    direction_degrees=25.3,
    is_fetchable=True,  # < 0.5m
    waypoints=None,
    warnings=None
)
```

### Output (Not Fetchable - with Waypoints):
```python
NavigationInstruction(
    object_name='Chair',
    distance_meters=3.55,
    direction_clock='2 o'clock (right)',
    direction_degrees=25.3,
    is_fetchable=False,  # > 0.5m
    waypoints=[
        {"direction": "12 o'clock (straight ahead)", "distance": 2.1},
        {"direction": "2 o'clock", "distance": 1.5},
    ],
    warnings=None
)
```

## Console Output Format

### Fetchable Object:
```
================================================================================
NAVIGATION INSTRUCTION
================================================================================
Object: Chair
Distance: 0.45 meters
Direction: 2 o'clock (right) (25.3°)
Fetchable: <isFetchable>True</isFetchable>

✓ Object is within reach! You can fetch it directly.
================================================================================
```

### Non-Fetchable Object (with Waypoints):
```
================================================================================
NAVIGATION INSTRUCTION
================================================================================
Object: Chair
Distance: 3.55 meters
Direction: 2 o'clock (right) (25.3°)
Fetchable: <isFetchable>False</isFetchable>

📍 Navigation Waypoints (2 waypoints):
  1. Direction: 12 o'clock (straight ahead), Distance: 2.1m
  2. Direction: 2 o'clock, Distance: 1.5m
================================================================================
```

## JSON Output Format

### Fetchable:
```json
{
  "object_name": "Chair",
  "distance_meters": 0.45,
  "direction_clock": "2 o'clock (right)",
  "direction_degrees": 25.3,
  "is_fetchable": true,
  "waypoints": null,
  "warnings": null
}
```

### Non-Fetchable:
```json
{
  "object_name": "Chair",
  "distance_meters": 3.55,
  "direction_clock": "2 o'clock (right)",
  "direction_degrees": 25.3,
  "is_fetchable": false,
  "waypoints": [
    {
      "direction": "12 o'clock (straight ahead)",
      "distance": 2.1
    },
    {
      "direction": "2 o'clock",
      "distance": 1.5
    }
  ],
  "warnings": null
}
```

## Key Benefits

### 1. **More Intuitive Navigation**
- Clock-based directions are easier to understand for blind users
- "Walk toward 2 o'clock for 2 meters" is clearer than "Move right 45 degrees"

### 2. **Better Path Planning**
- BEV provides top-down view of environment
- More accurate obstacle avoidance
- Cleaner path representation

### 3. **Simplified Waypoints**
- Fewer waypoints (grouped by direction)
- Each waypoint is actionable
- No need to interpret multiple small steps

### 4. **Consistent Distance Threshold**
- 0.5m fetchable distance matches human arm reach
- Clear binary decision: fetchable or navigate

## Technical Details

### BEV Projection Math

```python
# Camera intrinsics (default for Kinect/RealSense)
fx = fy = 525.0  # Focal length
cx = w / 2.0     # Principal point x
cy = h / 2.0     # Principal point y

# Image pixel (u, v) with depth Z
X = (u - cx) * Z / fx  # 3D X (right)
Y = (v - cy) * Z / fy  # 3D Y (down)
Z = depth             # 3D Z (forward)

# Project to BEV grid
cell_size = 8.0 / 160  # 0.05m per pixel
bev_x = cam_x + X / cell_size
bev_y = cam_y + Z / cell_size  # Z becomes y in BEV
```

### Waypoint Segmentation

Waypoints are created by grouping path segments with similar direction:
- Consecutive points with direction change < 30° are grouped
- Each group becomes one waypoint
- Direction is calculated as clock position
- Distance is Euclidean distance in meters

## Backward Compatibility

❌ **Breaking Changes:**
- `is_reachable` renamed to `is_fetchable`
- `safe_path` replaced with `waypoints`
- `reachable_position` removed
- Output format completely changed

✅ **What Still Works:**
- `process_query()` function signature (except optional params)
- Distance calculation
- Direction calculation
- JSON export (different structure)
- Visualization (updated layout)

## Files Modified

1. **pathfinder.py** - Main implementation
   - Lines 49-54: Updated constants
   - Lines 29-38: Updated NavigationInstruction dataclass
   - Lines 99-167: Added `depth_to_bev()`
   - Lines 315-379: Added `object_to_bev_coords()`
   - Lines 574-656: Added `generate_waypoints()`
   - Lines 658-690: Added `angle_to_clock()`
   - Lines 769-821: Updated `process_query()` with BEV pathfinding
   - Lines 860-882: Updated visualization for waypoints
   - Lines 994-1017: Updated console output

## Testing

Run with example query:
```bash
cd Agents
python pathfinder.py \
  --rgb_image ../queries/images/0001_*.jpg \
  --depth_image ../queries/images/0001_*_depth.png \
  --object Chair \
  --bbox "[407, 163, 614, 325]"
```

Expected output: Waypoints with clock-based directions if distance > 0.5m

## Next Steps

1. **Test with real dataset**: Run on multiple queries from prompts.csv
2. **Tune parameters**: Adjust BEV resolution, segmentation threshold
3. **Add visualization**: Draw BEV grid with path overlay
4. **Optimize performance**: Cache BEV conversion, optimize A* search

## Summary

The PathFinder system now uses a modern BEV-based approach with intuitive clock-based waypoint navigation. This provides clearer instructions for blind users by converting complex depth maps into simple directional waypoints like "Walk toward 2 o'clock for 2 meters."

Key improvement: Instead of "Move forward 0.5m, turn right 30°, move forward 1.2m...", users now get "Walk toward 2 o'clock for 1.8m" - much simpler and more natural!
