"""
Visualization utilities for object detection results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
import random


class DetectionVisualizer:
    """Utility class for visualizing object detection results."""

    def __init__(self, class_names: List[str] = None):
        """
        Initialize visualizer.

        Args:
            class_names: List of class names
        """
        self.class_names = class_names or []
        self.colors = self._generate_colors(len(self.class_names))

    def _generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """
        Generate distinct colors for each class.

        Args:
            n: Number of colors to generate

        Returns:
            List of RGB color tuples
        """
        colors = []
        for i in range(n):
            hue = int(180 * i / n)
            color = cv2.cvtColor(
                np.uint8([[[hue, 255, 255]]]),
                cv2.COLOR_HSV2BGR
            )[0][0]
            colors.append(tuple(map(int, color)))
        return colors

    def draw_detection(self, image: np.ndarray, bbox: Dict,
                      color: Tuple[int, int, int] = None,
                      thickness: int = 2) -> np.ndarray:
        """
        Draw a single detection on image.

        Args:
            image: Input image
            bbox: Bounding box dictionary with keys: x1, y1, x2, y2, class_id, class_name, confidence
            color: Box color (auto-generated if None)
            thickness: Line thickness

        Returns:
            Image with drawn detection
        """
        # Get coordinates
        x1, y1 = int(bbox['bbox']['x1']), int(bbox['bbox']['y1'])
        x2, y2 = int(bbox['bbox']['x2']), int(bbox['bbox']['y2'])

        # Get color
        if color is None:
            class_id = bbox['class_id']
            color = self.colors[class_id % len(self.colors)] if self.colors else (0, 255, 0)

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Draw label
        label = f"{bbox['class_name']}: {bbox['confidence']:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1

        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )

        # Draw background rectangle for text
        cv2.rectangle(
            image,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )

        # Draw text
        cv2.putText(
            image,
            label,
            (x1, y1 - baseline - 2),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness
        )

        return image

    def draw_detections(self, image: np.ndarray,
                       detections: List[Dict]) -> np.ndarray:
        """
        Draw multiple detections on image.

        Args:
            image: Input image
            detections: List of detection dictionaries

        Returns:
            Image with all detections drawn
        """
        result = image.copy()

        for det in detections:
            result = self.draw_detection(result, det)

        return result

    def create_comparison_grid(self, images: List[np.ndarray],
                              titles: List[str] = None,
                              cols: int = 3) -> np.ndarray:
        """
        Create a grid of images for comparison.

        Args:
            images: List of images
            titles: List of titles for each image
            cols: Number of columns in grid

        Returns:
            Grid image
        """
        n_images = len(images)
        rows = (n_images + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1 or cols == 1:
            axes = axes.reshape(rows, cols)

        for i in range(rows * cols):
            row = i // cols
            col = i % cols

            if i < n_images:
                # Convert BGR to RGB for matplotlib
                image_rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
                axes[row, col].imshow(image_rgb)

                if titles and i < len(titles):
                    axes[row, col].set_title(titles[i])

                axes[row, col].axis('off')
            else:
                axes[row, col].axis('off')

        plt.tight_layout()

        # Convert plot to image
        fig.canvas.draw()
        grid_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        grid_image = grid_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        return grid_image

    def annotate_with_stats(self, image: np.ndarray,
                           detections: List[Dict]) -> np.ndarray:
        """
        Annotate image with detection statistics.

        Args:
            image: Input image
            detections: List of detections

        Returns:
            Annotated image
        """
        result = image.copy()

        # Count detections per class
        class_counts = {}
        for det in detections:
            class_name = det['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        # Draw statistics box
        stats_text = [f"Total: {len(detections)}"]
        for class_name, count in sorted(class_counts.items()):
            stats_text.append(f"{class_name}: {count}")

        # Draw background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        padding = 10

        max_text_width = 0
        total_height = padding

        for text in stats_text:
            (text_width, text_height), _ = cv2.getTextSize(
                text, font, font_scale, thickness
            )
            max_text_width = max(max_text_width, text_width)
            total_height += text_height + padding

        # Draw semi-transparent background
        overlay = result.copy()
        cv2.rectangle(
            overlay,
            (10, 10),
            (10 + max_text_width + 2 * padding, 10 + total_height),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.7, result, 0.3, 0, result)

        # Draw text
        y = 10 + padding + 20
        for text in stats_text:
            cv2.putText(
                result,
                text,
                (10 + padding, y),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )
            y += 30

        return result


def visualize_dataset_samples(dataset_path: str, num_samples: int = 9):
    """
    Visualize random samples from a YOLO dataset.

    Args:
        dataset_path: Path to dataset directory
        num_samples: Number of samples to visualize
    """
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / 'images' / 'train'
    labels_dir = dataset_path / 'labels' / 'train'

    # Get random samples
    image_files = list(images_dir.glob('*.jpg'))
    random.shuffle(image_files)
    samples = image_files[:num_samples]

    # Create visualization
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    for i, img_path in enumerate(samples):
        # Read image
        image = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read label
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            with open(label_path, 'r') as f:
                labels = f.readlines()

            # Draw bounding boxes
            h, w = image.shape[:2]
            for label in labels:
                parts = label.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])

                    # Convert to pixel coordinates
                    x1 = int((x_center - width/2) * w)
                    y1 = int((y_center - height/2) * h)
                    x2 = int((x_center + width/2) * w)
                    y2 = int((y_center + height/2) * h)

                    # Draw rectangle
                    color = tuple(np.random.randint(0, 255, 3).tolist())
                    cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 2)

        axes[i].imshow(image_rgb)
        axes[i].set_title(img_path.stem)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(dataset_path / 'dataset_samples.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to {dataset_path / 'dataset_samples.png'}")


if __name__ == "__main__":
    # Example usage
    print("Visualization utilities loaded successfully!")
