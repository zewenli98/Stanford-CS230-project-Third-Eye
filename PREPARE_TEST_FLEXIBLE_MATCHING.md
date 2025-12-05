# prepare_test.py - Flexible Object Name Matching

**Date:** 2025-12-05
**Purpose:** Enhanced object matching to handle case variations and compound names

---

## Summary

Updated `prepare_test.py` to use flexible, case-insensitive partial matching for all goal objects. This ensures that variations like "woodchair", "Chair", "CHAIR", and "chairs" all match the "chair" category.

---

## What Changed

### **Previous Behavior:**

The code already had partial matching logic, but it wasn't explicitly documented and the matched category wasn't tracked separately.

```python
# Old: Basic partial matching
object_name_lower = object_name.lower()
is_goal_object = any(goal_obj in object_name_lower for goal_obj in goal_objects_lower)
```

### **New Behavior:**

Enhanced with explicit category tracking, logging, and normalization:

```python
# New: Track matched category and original name
for goal_obj in goal_objects_lower:
    if goal_obj in object_name_lower:
        matched_category = goal_obj  # e.g., "chair"
        break

result_objects.append({
    'name': original_name,  # Keep original (e.g., "woodchair")
    'matched_category': matched_category,  # Normalized (e.g., "chair")
    ...
})
```

---

## Object Matching Examples

### All Goal Objects Use Partial Matching:

**Chair:**
- ✅ `chair` → chair
- ✅ `Chair` → chair
- ✅ `CHAIR` → chair
- ✅ `woodchair` → chair
- ✅ `armchair` → chair
- ✅ `officechair` → chair
- ✅ `chairs` → chair
- ✅ `Chairs` → chair

**Mug:**
- ✅ `mug` → mug
- ✅ `Mug` → mug
- ✅ `MUG` → mug
- ✅ `coffeemug` → mug

**Book:**
- ✅ `book` → book
- ✅ `Book` → book
- ✅ `BOOK` → book
- ✅ `books` → book
- ✅ `notebook` → book
- ✅ `textbook` → book

**Cabinet:**
- ✅ `cabinet` → cabinet
- ✅ `Cabinet` → cabinet
- ✅ `CABINET` → cabinet
- ✅ `filecabinet` → cabinet
- ✅ `medicalcabinet` → cabinet

**Door:**
- ✅ `door` → door
- ✅ `Door` → door
- ✅ `DOOR` → door
- ✅ `glassdoor` → door
- ✅ `wooddoor` → door

**Lamp:**
- ✅ `lamp` → lamp
- ✅ `Lamp` → lamp
- ✅ `LAMP` → lamp
- ✅ `tablelamp` → lamp
- ✅ `floorlamp` → lamp

---

## Code Changes

### 1. Enhanced Object Selection (Lines 246-266)

**Added explicit category tracking:**

```python
# Check if object is in goal objects list (case-insensitive partial matching)
if goal_objects_lower:
    # Partial matching: "chair" matches "chair", "Chair", "CHAIR", "woodchair", "armchair", etc.
    # This handles:
    # - Case variations: Chair, CHAIR, chair
    # - Compound names: woodchair, officechair, armchair
    # - Plural forms: chairs, Chairs, CHAIRS
    object_name_lower = object_name.lower()
    matched_category = None

    for goal_obj in goal_objects_lower:
        if goal_obj in object_name_lower:
            matched_category = goal_obj
            break

    if not matched_category:
        continue

    # Store matched category for logging
    poly['matched_category'] = matched_category
    poly['original_name'] = object_name
```

### 2. Result Object Structure (Lines 287-323)

**Track both original name and matched category:**

```python
result_objects.append({
    'name': original_name,  # Keep original name (e.g., "woodchair")
    'matched_category': matched_category,  # Normalized category (e.g., "chair")
    'bbox': bbox,
    'polygon': poly,
    'object_index': object_idx
})

logger.info(f"Found {len(result_objects)} target object(s) in image")
logger.debug(f"  Matched categories: {matched_categories_count}")
for obj in result_objects:
    if obj['name'].lower() != obj['matched_category']:
        logger.debug(f"    '{obj['name']}' → matched as '{obj['matched_category']}'")
```

### 3. Sample Structure (Lines 484-505)

**Added primary_category field:**

```python
test_sample = {
    'scene_folder': str(scene_folder),
    'scene_type': scene_data['scene_type'],
    'image': scene_data['image'],
    'image_path': str(scene_data['image_path']),
    'goal_objects': objects_with_positions,  # List with matched_category
    'primary_object': primary_obj['name'],  # Original name (e.g., "woodchair")
    'primary_category': primary_obj['matched_category'],  # Normalized (e.g., "chair")
    'full_annotation': scene_data['annotations']
}
```

### 4. Category Balancing (Lines 609-616)

**Use matched category for balancing:**

```python
# Update category count using normalized category
primary_category = sample.get('primary_category', sample['primary_object'].lower())
if primary_category in category_counts:
    category_counts[primary_category] += 1
```

### 5. Enhanced Logging (Lines 499-503)

**Show original name with matched category:**

```python
for obj in objects_with_positions:
    if obj['name'].lower() != obj['matched_category']:
        logger.info(f"    - {obj['name']} (as {obj['matched_category']}): {obj['distance_meters']:.2f}m, {obj['direction']}")
    else:
        logger.info(f"    - {obj['name']}: {obj['distance_meters']:.2f}m, {obj['direction']}")
```

---

## Example Output

### Console Output:

```
Preparing positive sample 1/10
  Targeting category: chair (current count: 0)
  Found 3 target object(s) in image
    'woodchair' → matched as 'chair'
    'officeChair' → matched as 'chair'
Successfully prepared test sample:
  Scene: office
  Objects found: 3
    - woodchair (as chair): 2.50m, 2 o'clock
    - officeChair (as chair): 3.20m, 11 o'clock
    - chair: 1.80m, 12 o'clock

Category distribution: {'mug': 0, 'chair': 1, 'book': 0, 'cabinet': 0, 'door': 0, 'lamp': 0}
```

### CSV Output:

The CSV now includes objects with their original names:

```csv
prompt_id,prompt_text,object,object_bbox_list,object_distance_list,object_direction_list,...
1,where is woodchair,"[[100,200,300,400],[500,150,600,250],[200,300,350,450]]","[""2.50"",""3.20"",""1.80""]","[""2 o'clock"",""11 o'clock"",""12 o'clock""]",...
```

Note: The query uses the original name "woodchair", but the system correctly identifies it as matching the "chair" category.

---

## Benefits

1. ✅ **Handles Case Variations**: "Chair", "CHAIR", "chair" all match
2. ✅ **Handles Compound Names**: "woodchair", "armchair", "officechair" match "chair"
3. ✅ **Handles Plurals**: "chairs", "books", "lamps" match singular forms
4. ✅ **Preserves Original Names**: Keeps "woodchair" instead of normalizing to "chair"
5. ✅ **Better Logging**: Shows which objects matched and how
6. ✅ **Accurate Balancing**: Uses matched category for category distribution

---

## Data Flow Example

### Annotation File Contains:
```json
{
  "objects": [
    {"name": "woodchair", ...},
    {"name": "officeChair", ...},
    {"name": "table", ...}
  ]
}
```

### Goal Objects List:
```python
GOAL_OBJECTS = ["mug", "chair", "book", "cabinet", "door", "lamp"]
```

### Matching Process:

1. **"woodchair":**
   - Check: "chair" in "woodchair"? ✅ Yes
   - Matched category: "chair"
   - Keep original name: "woodchair"

2. **"officeChair":**
   - Check: "chair" in "officechair" (lowercased)? ✅ Yes
   - Matched category: "chair"
   - Keep original name: "officeChair"

3. **"table":**
   - Check: any goal_obj in "table"? ❌ No
   - Skip this object

### Result:
```python
{
  'goal_objects': [
    {'name': 'woodchair', 'matched_category': 'chair', ...},
    {'name': 'officeChair', 'matched_category': 'chair', ...}
  ],
  'primary_object': 'woodchair',
  'primary_category': 'chair'
}
```

---

## Testing

### Verify Matching Works:

Run the script and check the logs:

```bash
python prepare_test.py
```

Look for lines like:
```
Found 3 target object(s) in image
  'woodchair' → matched as 'chair'
  'officeChair' → matched as 'chair'
```

### Verify Category Balancing:

Check that balancing uses matched categories, not original names:

```
Category distribution: {'mug': 2, 'chair': 2, 'book': 2, 'cabinet': 2, 'door': 1, 'lamp': 1}
```

Even though original names might be "woodchair", "officechair", they're all counted under "chair".

---

## Comparison with parse_sunrgbd.py

Both files now use similar matching logic:

### parse_sunrgbd.py:
- Normalizes object names to 6 target classes
- Stores normalized name in annotations
- Used for training YOLO model

### prepare_test.py:
- Matches objects to goal categories
- Preserves original names for queries
- Used for generating test samples

**Key Difference:**
- `parse_sunrgbd.py`: Normalizes and saves as "chair"
- `prepare_test.py`: Matches as "chair" but keeps "woodchair" for queries

This is intentional - test queries should use realistic names like "where is woodchair" instead of always using the generic "where is chair".

---

## Documentation Update

Added comprehensive documentation to the class docstring (Lines 47-56):

```python
Object Matching Logic:
- Uses case-insensitive partial matching for all goal objects
- Examples:
    - "chair" matches: chair, Chair, CHAIR, woodchair, armchair, officechair, chairs
    - "mug" matches: mug, Mug, MUG, coffeemug
    - "book" matches: book, Book, BOOK, books, notebook, textbook
    - "cabinet" matches: cabinet, Cabinet, filecabinet, medicalcabinet
    - "door" matches: door, Door, glassdoor, wooddoor
    - "lamp" matches: lamp, Lamp, tablelamp, floorlamp
- Preserves original name (e.g., "woodchair") while tracking matched category (e.g., "chair")
```

---

## Status

✅ **All updates complete and ready to use**

The flexible matching system is now active and will automatically handle all case variations and compound names for the 6 goal object categories.
