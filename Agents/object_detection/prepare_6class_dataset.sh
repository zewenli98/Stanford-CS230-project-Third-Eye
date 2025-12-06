#!/bin/bash

# Quick-start script for preparing 6-class object detection dataset
# Filters SUN RGB-D for: Mug, Book, Cabinet, Door, Chair, Lamp

echo "================================================================================"
echo "SUN RGB-D Dataset Preparation - 6 Target Classes"
echo "Target: Mug, Book, Cabinet, Door, Chair, Lamp"
echo "================================================================================"
echo ""

# Configuration
SUNRGBD_PATH="../../SUNRGBD"
OUTPUT_PATH="./data/sunrgbd_6classes"
TRAIN_RATIO=0.8
VAL_RATIO=0.1
TEST_RATIO=0.1
SEED=42

# Check if SUNRGBD dataset exists
if [ ! -d "$SUNRGBD_PATH" ]; then
    echo "❌ Error: SUNRGBD dataset not found at $SUNRGBD_PATH"
    echo "   Please update SUNRGBD_PATH in this script or create symlink"
    exit 1
fi

echo "✓ Found SUNRGBD dataset at: $SUNRGBD_PATH"
echo "✓ Output will be saved to: $OUTPUT_PATH"
echo ""

# Run data preparation
echo "Running data preparation..."
echo "This may take 10-20 minutes depending on dataset size..."
echo ""

python data_prep/parse_sunrgbd.py \
    --sunrgbd_path "$SUNRGBD_PATH" \
    --output_path "$OUTPUT_PATH" \
    --train_ratio $TRAIN_RATIO \
    --val_ratio $VAL_RATIO \
    --test_ratio $TEST_RATIO \
    --seed $SEED

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "✅ Dataset preparation completed successfully!"
    echo "================================================================================"
    echo ""
    echo "Output files:"
    echo "  📁 Images:        $OUTPUT_PATH/images/{train,val,test}/"
    echo "  📄 Labels:        $OUTPUT_PATH/labels/{train,val,test}/"
    echo "  ⚙️  Config:        $OUTPUT_PATH/sunrgbd_6classes.yaml"
    echo "  📊 Statistics:    $OUTPUT_PATH/class_statistics.txt"
    echo ""
    echo "Next steps:"
    echo "  1. Review statistics: cat $OUTPUT_PATH/class_statistics.txt"
    echo "  2. Train model:       python training/train.py --data $OUTPUT_PATH/sunrgbd_6classes.yaml"
    echo "  3. Evaluate model:    python evaluation/evaluate.py --model ./models/best.pt"
    echo ""
else
    echo ""
    echo "================================================================================"
    echo "❌ Dataset preparation failed!"
    echo "================================================================================"
    echo "Check the error messages above for details"
    exit 1
fi
