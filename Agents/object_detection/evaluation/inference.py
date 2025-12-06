"""
Inference Script for Single Image/Video Prediction
Supports batch inference, video processing, and real-time visualization.
"""

import os
import sys
from pathlib import Path
import logging
import torch
import numpy as np
import cv2
from typing import Union, List, Tuple
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from ultralytics import YOLO
from configs.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ObjectDetector:
    """Inference wrapper for YOLOv8 object detection."""

    def __init__(self, model_path: str, config_path: str = None, device: str = None):
        """
        Initialize detector.

        Args:
            model_path: Path to trained model weights
            config_path: Path to configuration file
            device: Device to run inference on (cpu, cuda, mps)
        """
        self.model_path = Path(model_path)
        self.config = Config(config_path) if config_path else None

        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = YOLO(str(model_path))
        self.model.to(self.device)

        # Get inference parameters from config
        if self.config:
            self.conf_threshold = self.config.get('inference.conf_thres', 0.25)
            self.iou_threshold = self.config.get('inference.iou_thres', 0.45)
            self.max_det = self.config.get('inference.max_det', 300)
        else:
            self.conf_threshold = 0.25
            self.iou_threshold = 0.45
            self.max_det = 300

        # Output directory
        self.output_dir = Path('./Agents/outputs/inference')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def predict_image(self, image_path: Union[str, Path], save: bool = True,
                     show: bool = False) -> dict:
        """
        Run inference on a single image.

        Args:
            image_path: Path to input image
            save: Whether to save annotated image
            show: Whether to display result

        Returns:
            Dictionary with predictions and metadata
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.info(f"Processing image: {image_path}")

        # Run inference
        results = self.model(
            str(image_path),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_det,
            verbose=False
        )

        # Extract predictions
        predictions = self._extract_predictions(results[0])

        # Annotate image
        annotated_image = results[0].plot()

        # Save if requested
        if save:
            output_path = self.output_dir / f"{image_path.stem}_detected.jpg"
            cv2.imwrite(str(output_path), annotated_image)
            logger.info(f"Saved annotated image to {output_path}")

            # Save predictions as JSON
            json_path = self.output_dir / f"{image_path.stem}_predictions.json"
            with open(json_path, 'w') as f:
                json.dump(predictions, f, indent=2)
            logger.info(f"Saved predictions to {json_path}")

        # Show if requested
        if show:
            cv2.imshow('Detection Result', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return {
            'predictions': predictions,
            'annotated_image': annotated_image,
            'image_path': str(image_path)
        }

    def predict_batch(self, image_dir: Union[str, Path], save: bool = True) -> List[dict]:
        """
        Run inference on a batch of images.

        Args:
            image_dir: Directory containing images
            save: Whether to save annotated images

        Returns:
            List of prediction dictionaries
        """
        image_dir = Path(image_dir)

        if not image_dir.exists():
            raise FileNotFoundError(f"Directory not found: {image_dir}")

        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(image_dir.glob(ext))

        logger.info(f"Found {len(image_files)} images in {image_dir}")

        # Process each image
        all_results = []
        for img_path in image_files:
            try:
                result = self.predict_image(img_path, save=save, show=False)
                all_results.append(result)
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")

        # Save batch summary
        if save:
            summary_path = self.output_dir / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            summary = {
                'total_images': len(image_files),
                'processed': len(all_results),
                'results': [
                    {
                        'image': r['image_path'],
                        'num_detections': len(r['predictions']['detections'])
                    }
                    for r in all_results
                ]
            }
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Saved batch summary to {summary_path}")

        return all_results

    def predict_video(self, video_path: Union[str, Path], save: bool = True,
                     show: bool = False, skip_frames: int = 0) -> dict:
        """
        Run inference on a video.

        Args:
            video_path: Path to input video
            save: Whether to save annotated video
            show: Whether to display result in real-time
            skip_frames: Number of frames to skip between detections

        Returns:
            Dictionary with video processing results
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        logger.info(f"Processing video: {video_path}")

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Video properties: {width}x{height} @ {fps} FPS, {total_frames} frames")

        # Setup video writer if saving
        writer = None
        if save:
            output_path = self.output_dir / f"{video_path.stem}_detected.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Process frames
        frame_count = 0
        detection_count = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames if requested
                if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                    frame_count += 1
                    if save:
                        writer.write(frame)
                    continue

                # Run inference
                results = self.model(
                    frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    max_det=self.max_det,
                    verbose=False
                )

                # Get annotated frame
                annotated_frame = results[0].plot()

                # Count detections
                detection_count += len(results[0].boxes)

                # Save frame
                if save and writer is not None:
                    writer.write(annotated_frame)

                # Show frame
                if show:
                    cv2.imshow('Video Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                frame_count += 1

                if frame_count % 30 == 0:
                    logger.info(f"Processed {frame_count}/{total_frames} frames")

        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if show:
                cv2.destroyAllWindows()

        logger.info(f"Video processing complete: {frame_count} frames, {detection_count} total detections")

        if save:
            logger.info(f"Saved annotated video to {output_path}")

        return {
            'total_frames': frame_count,
            'total_detections': detection_count,
            'avg_detections_per_frame': detection_count / frame_count if frame_count > 0 else 0,
            'video_path': str(video_path),
        }

    def predict_webcam(self, camera_id: int = 0):
        """
        Run real-time inference on webcam feed.

        Args:
            camera_id: Camera device ID
        """
        logger.info(f"Starting webcam inference on camera {camera_id}")
        logger.info("Press 'q' to quit")

        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            raise ValueError(f"Cannot open camera {camera_id}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Run inference
                results = self.model(
                    frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    max_det=self.max_det,
                    verbose=False
                )

                # Get annotated frame
                annotated_frame = results[0].plot()

                # Display
                cv2.imshow('Webcam Detection', annotated_frame)

                # Quit on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

        logger.info("Webcam inference stopped")

    def _extract_predictions(self, result) -> dict:
        """
        Extract predictions from YOLO result object.

        Args:
            result: YOLO result object

        Returns:
            Dictionary with structured predictions
        """
        predictions = {
            'detections': [],
            'summary': {
                'total_objects': len(result.boxes),
                'classes_detected': {}
            }
        }

        # Extract each detection
        for box in result.boxes:
            # Get box coordinates
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = result.names[cls_id]

            detection = {
                'class_id': cls_id,
                'class_name': cls_name,
                'confidence': conf,
                'bbox': {
                    'x1': float(xyxy[0]),
                    'y1': float(xyxy[1]),
                    'x2': float(xyxy[2]),
                    'y2': float(xyxy[3])
                }
            }

            predictions['detections'].append(detection)

            # Update class count
            if cls_name not in predictions['summary']['classes_detected']:
                predictions['summary']['classes_detected'][cls_name] = 0
            predictions['summary']['classes_detected'][cls_name] += 1

        return predictions


def main():
    """Main inference function."""
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 Inference")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model weights'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='./configs/training_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Path to image, directory, or video file'
    )
    parser.add_argument(
        '--type',
        type=str,
        choices=['image', 'batch', 'video', 'webcam'],
        default='image',
        help='Type of inference'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to run inference (cpu, cuda, mps)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=None,
        help='Confidence threshold'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=None,
        help='IoU threshold for NMS'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save results'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display results'
    )

    args = parser.parse_args()

    # Initialize detector
    detector = ObjectDetector(args.model, args.config, args.device)

    # Override thresholds if provided
    if args.conf is not None:
        detector.conf_threshold = args.conf
    if args.iou is not None:
        detector.iou_threshold = args.iou

    # Run inference based on type
    if args.type == 'image':
        detector.predict_image(args.source, save=args.save, show=args.show)
    elif args.type == 'batch':
        detector.predict_batch(args.source, save=args.save)
    elif args.type == 'video':
        detector.predict_video(args.source, save=args.save, show=args.show)
    elif args.type == 'webcam':
        camera_id = int(args.source) if args.source.isdigit() else 0
        detector.predict_webcam(camera_id=camera_id)

    logger.info("Inference completed!")


if __name__ == "__main__":
    main()
