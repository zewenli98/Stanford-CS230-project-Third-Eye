"""
Data Augmentation Pipeline for Indoor Scene Object Detection
Optimized for SUN RGB-D dataset with indoor-specific augmentations.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class IndoorAugmentation:
    """Augmentation pipeline optimized for indoor scenes."""

    @staticmethod
    def get_training_augmentation(img_size: int = 640) -> A.Compose:
        """
        Get training augmentation pipeline.

        Args:
            img_size: Target image size

        Returns:
            Albumentations compose object
        """
        return A.Compose([
            # Geometric transformations
            A.LongestMaxSize(max_size=img_size, p=1.0),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0
            ),

            # Horizontal flip (common for indoor scenes)
            A.HorizontalFlip(p=0.5),

            # Perspective and affine transforms (common in indoor navigation)
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent=(-0.1, 0.1),
                rotate=(-10, 10),
                shear=(-5, 5),
                p=0.4
            ),

            # Color and lighting augmentations (important for indoor scenes with varying lighting)
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=1.0
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=1.0
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            ], p=0.6),

            # Shadow and lighting effects (common in indoor scenes)
            A.OneOf([
                A.RandomShadow(
                    shadow_roi=(0, 0, 1, 1),
                    num_shadows_lower=1,
                    num_shadows_upper=2,
                    shadow_dimension=5,
                    p=1.0
                ),
                A.RandomToneCurve(scale=0.1, p=1.0),
            ], p=0.3),

            # Blur and noise (to simulate camera quality variations)
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.2),

            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            ], p=0.2),

            # Image quality degradation
            A.OneOf([
                A.ImageCompression(quality_lower=75, quality_upper=100, p=1.0),
                A.Downscale(scale_min=0.7, scale_max=0.9, p=1.0),
            ], p=0.2),

            # Color channel manipulation
            A.OneOf([
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),
                A.ChannelShuffle(p=1.0),
            ], p=0.1),

            # Normalize (important for model training)
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),

        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_area=0,
            min_visibility=0.3,
        ))

    @staticmethod
    def get_validation_augmentation(img_size: int = 640) -> A.Compose:
        """
        Get validation augmentation pipeline (minimal augmentation).

        Args:
            img_size: Target image size

        Returns:
            Albumentations compose object
        """
        return A.Compose([
            A.LongestMaxSize(max_size=img_size, p=1.0),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_area=0,
            min_visibility=0.0,
        ))

    @staticmethod
    def visualize_augmentation(
        image: np.ndarray,
        bboxes: list,
        class_labels: list,
        augmentation: A.Compose,
        num_examples: int = 5
    ) -> list:
        """
        Visualize augmentation examples.

        Args:
            image: Input image (numpy array)
            bboxes: List of bounding boxes in YOLO format
            class_labels: List of class labels
            augmentation: Augmentation pipeline
            num_examples: Number of augmented examples to generate

        Returns:
            List of augmented images with bboxes
        """
        results = []

        for i in range(num_examples):
            augmented = augmentation(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )

            results.append({
                'image': augmented['image'],
                'bboxes': augmented['bboxes'],
                'class_labels': augmented['class_labels']
            })

        return results


def test_augmentation():
    """Test augmentation pipeline with a sample image."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # Create a dummy image
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Dummy bboxes in YOLO format (x_center, y_center, width, height)
    bboxes = [
        [0.5, 0.5, 0.3, 0.3],
        [0.2, 0.3, 0.15, 0.2]
    ]
    class_labels = [0, 1]

    # Get augmentation
    aug = IndoorAugmentation.get_training_augmentation(img_size=640)

    # Apply augmentation
    augmented = aug(image=img, bboxes=bboxes, class_labels=class_labels)

    print(f"Original image shape: {img.shape}")
    print(f"Augmented image shape: {augmented['image'].shape}")
    print(f"Original bboxes: {bboxes}")
    print(f"Augmented bboxes: {augmented['bboxes']}")
    print(f"Class labels: {augmented['class_labels']}")


if __name__ == "__main__":
    test_augmentation()
