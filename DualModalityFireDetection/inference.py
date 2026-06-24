"""
Inference script for Dual Modality Fire Detection
"""

import os
import sys
import argparse
import torch
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config
from models import build_model
from utils import non_max_suppression, xyxy2xywh, xywh2xyxy


class FireDetector:
    """Fire detector for inference"""

    def __init__(self, checkpoint_path, device='cuda', conf_threshold=0.25, iou_threshold=0.45):
        self.config = Config.get_config()
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = self.config['data']['img_size']

        # Build model
        print(f"Loading model from {checkpoint_path}...")
        self.model = build_model(self.config, weights=checkpoint_path)
        self.model = self.model.to(device)
        self.model.eval()

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)

        print("Model loaded successfully!")

    def preprocess(self, rgb_img, thermal_img):
        """
        Preprocess images for inference
        Args:
            rgb_img: RGB image (numpy array or PIL Image)
            thermal_img: Thermal image (numpy array or PIL Image)
        Returns:
            Preprocessed tensors
        """
        # Convert PIL to numpy
        if isinstance(rgb_img, Image.Image):
            rgb_img = np.array(rgb_img)
        if isinstance(thermal_img, Image.Image):
            thermal_img = np.array(thermal_img)

        # Store original shape
        self.orig_shape = rgb_img.shape[:2]

        # Resize
        rgb_img = cv2.resize(rgb_img, (self.img_size, self.img_size))
        thermal_img = cv2.resize(thermal_img, (self.img_size, self.img_size))

        # Convert RGB to BGR if needed and then to RGB
        if rgb_img.shape[2] == 3 and rgb_img.dtype == np.uint8:
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        # Prepare RGB tensor
        rgb_tensor = torch.from_numpy(rgb_img).float() / 255.0
        rgb_tensor = rgb_tensor.permute(2, 0, 1)

        # Normalize RGB
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        rgb_tensor = (rgb_tensor - mean) / std

        # Prepare thermal tensor
        if len(thermal_img.shape) == 3:
            thermal_img = cv2.cvtColor(thermal_img, cv2.COLOR_BGR2GRAY)

        thermal_tensor = torch.from_numpy(thermal_img).float() / 255.0
        thermal_tensor = (thermal_tensor - 0.5) / 0.5
        thermal_tensor = thermal_tensor.unsqueeze(0).repeat(3, 1, 1)

        # Add batch dimension
        rgb_tensor = rgb_tensor.unsqueeze(0)
        thermal_tensor = thermal_tensor.unsqueeze(0)

        return rgb_tensor, thermal_tensor

    def postprocess(self, outputs):
        """
        Postprocess model outputs
        Returns detections in xyxy format with confidence scores
        """
        # This is a simplified postprocessing
        # Full implementation would decode boxes and apply NMS
        detections = []

        return detections

    def detect(self, rgb_img, thermal_img, return_vis=False):
        """
        Run detection on image pair
        Args:
            rgb_img: RGB image
            thermal_img: Thermal image
            return_vis: Return visualized image
        Returns:
            detections: List of [x1, y1, x2, y2, conf, cls] or visualized image
        """
        rgb_tensor, thermal_tensor = self.preprocess(rgb_img, thermal_img)

        rgb_tensor = rgb_tensor.to(self.device)
        thermal_tensor = thermal_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(rgb_tensor, thermal_tensor)

        detections = self.postprocess(outputs)

        if return_vis:
            vis_img = self.visualize_detections(rgb_img, detections)
            return detections, vis_img

        return detections

    def visualize_detections(self, img, detections, save_path=None):
        """Visualize detections on image"""
        if isinstance(img, Image.Image):
            img = np.array(img)

        vis_img = img.copy()

        for det in detections:
            x1, y1, x2, y2, conf, cls = det

            # Scale boxes to original size
            h, w = self.orig_shape
            scale_x = w / self.img_size
            scale_y = h / self.img_size

            x1, x2 = x1 * scale_x, x2 * scale_x
            y1, y2 = y1 * scale_y, y2 * scale_y

            # Draw box
            cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Draw label
            label = f'Fire: {conf:.2f}'
            cv2.putText(vis_img, label, (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
            print(f"Saved visualization to {save_path}")

        return vis_img


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Fire Detection Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--rgb', type=str, required=True,
                        help='Path to RGB image')
    parser.add_argument('--thermal', type=str, required=True,
                        help='Path to thermal image')
    parser.add_argument('--output', type=str, default='output.jpg',
                        help='Output image path')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS')

    args = parser.parse_args()
    return args


def main():
    """Main entry point"""
    args = parse_args()

    # Create detector
    detector = FireDetector(
        checkpoint_path=args.checkpoint,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )

    # Load images
    print(f"Loading images...")
    rgb_img = cv2.imread(args.rgb)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

    thermal_img = cv2.imread(args.thermal, cv2.IMREAD_GRAYSCALE)
    thermal_img = cv2.cvtColor(thermal_img, cv2.COLOR_GRAY2RGB)

    # Run detection
    print("Running detection...")
    detections, vis_img = detector.detect(rgb_img, thermal_img, return_vis=True)

    # Save result
    cv2.imwrite(args.output, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    print(f"Saved result to {args.output}")

    # Print detections
    print(f"\nDetected {len(detections)} fire(s):")
    for i, det in enumerate(detections):
        x1, y1, x2, y2, conf, cls = det
        print(f"  {i+1}. Fire at ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}) with confidence {conf:.3f}")


if __name__ == '__main__':
    main()
