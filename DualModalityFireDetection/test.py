"""
Testing script for Dual Modality Fire Detection Network
"""

import os
import sys
import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config
from models import build_model
from utils import (
    DetectionMetrics,
    non_max_suppression,
    xyxy2xywh,
    scale_boxes
)


class Tester:
    """Tester class for dual modality fire detection"""

    def __init__(self, config, checkpoint_path, device='cuda'):
        self.config = config
        self.device = device
        self.checkpoint_path = checkpoint_path

        # Build model
        self.model = build_model(config, weights=checkpoint_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Metrics
        self.metrics = DetectionMetrics(
            num_classes=config['detection_head']['num_classes']
        )

        # Test config
        self.test_config = config['test']

    def load_image_pair(self, rgb_path, thermal_path):
        """Load and preprocess image pair"""
        # Load RGB image
        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        # Load Thermal image
        thermal_img = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)

        orig_h, orig_w = rgb_img.shape[:2]

        # Resize to model input size
        img_size = self.config['data']['img_size']
        rgb_img = cv2.resize(rgb_img, (img_size, img_size))
        thermal_img = cv2.resize(thermal_img, (img_size, img_size))

        # Convert to tensor format [B, C, H, W]
        rgb_tensor = torch.from_numpy(rgb_img.transpose(2, 0, 1)).float() / 255.0
        thermal_tensor = torch.from_numpy(thermal_img).float() / 255.0

        # Normalize RGB
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        rgb_tensor = (rgb_tensor - mean) / std

        # Normalize thermal
        thermal_tensor = (thermal_tensor - 0.5) / 0.5
        thermal_tensor = thermal_tensor.unsqueeze(0)  # Add channel dim
        thermal_tensor = thermal_tensor.repeat(3, 1, 1)  # Repeat to 3 channels

        # Add batch dimension
        rgb_tensor = rgb_tensor.unsqueeze(0)
        thermal_tensor = thermal_tensor.unsqueeze(0)

        return rgb_tensor, thermal_tensor, (orig_h, orig_w)

    def test_single_image(self, rgb_path, thermal_path, gt_boxes=None):
        """Test on a single image pair"""
        rgb_tensor, thermal_tensor, orig_shape = self.load_image_pair(rgb_path, thermal_path)

        rgb_tensor = rgb_tensor.to(self.device)
        thermal_tensor = thermal_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(rgb_tensor, thermal_tensor)

        # Process predictions (simplified - would need full postprocessing)
        # Return format: list of [N, 6] tensors (x1, y1, x2, y2, conf, cls)
        predictions = []

        return predictions

    def test_dataset(self, test_loader):
        """Test on full dataset"""
        all_predictions = []
        all_targets = []

        print("Testing on dataset...")

        for rgb_imgs, thermal_imgs, targets in tqdm(test_loader):
            rgb_imgs = rgb_imgs.to(self.device)
            thermal_imgs = thermal_imgs.to(self.device)

            with torch.no_grad():
                outputs = self.model(rgb_imgs, thermal_imgs)

            # Process outputs
            predictions = self._process_outputs(outputs)

            # Apply NMS
            detections = non_max_suppression(
                predictions,
                conf_threshold=self.test_config['conf_threshold'],
                iou_threshold=self.test_config['iou_threshold'],
                max_det=self.test_config['max_detections']
            )

            all_predictions.extend(detections)

            # Format targets
            formatted_targets = []
            for target in targets:
                formatted_targets.append({
                    'boxes': target['boxes'],
                    'labels': target['labels']
                })
            all_targets.extend(formatted_targets)

        # Compute metrics
        self.metrics.update(all_predictions, all_targets)
        results = self.metrics.evaluate()

        return results

    def _process_outputs(self, outputs):
        """Process model outputs to detection format"""
        # This is a simplified version
        # Full implementation would decode boxes, apply sigmoid, etc.
        predictions = []

        # Process each modality (RGB, Thermal, Fused)
        for modality in ['rgb', 'thermal', 'fused']:
            scale_preds = []
            for scale_pred in outputs[modality]:
                cls_pred, reg_pred, conf_pred = scale_pred
                # Process predictions...
                scale_preds.append(None)  # Placeholder
            predictions.append(scale_preds)

        return predictions

    def visualize_predictions(self, rgb_path, thermal_path, output_path=None):
        """Visualize predictions on image"""
        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        # Get predictions
        predictions = self.test_single_image(rgb_path, thermal_path)

        # Draw boxes
        for pred in predictions:
            if pred.numel() > 0:
                x1, y1, x2, y2, conf, cls = pred[0].cpu().numpy()
                cv2.rectangle(rgb_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(rgb_img, f'Fire: {conf:.2f}', (int(x1), int(y1) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        if output_path:
            cv2.imwrite(output_path, rgb_img)
            print(f"Saved visualization to {output_path}")

        return rgb_img


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test Dual Modality Fire Detection')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--rgb_dir', type=str, default=None,
                        help='Directory containing RGB images')
    parser.add_argument('--thermal_dir', type=str, default=None,
                        help='Directory containing thermal images')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize predictions')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()
    return args


def main():
    """Main entry point"""
    args = parse_args()

    # Load configuration
    config = Config.get_config()

    # Create tester
    tester = Tester(config, args.checkpoint, device=args.device)

    print("=" * 50)
    print("Dual Modality Fire Detection - Test Results")
    print("=" * 50)

    if args.visualize and args.rgb_dir and args.thermal_dir:
        # Visualize predictions
        os.makedirs(args.output, exist_ok=True)

        rgb_files = sorted(os.listdir(args.rgb_dir))
        thermal_files = sorted(os.listdir(args.thermal_dir))

        for rgb_file, thermal_file in zip(rgb_files, thermal_files):
            rgb_path = os.path.join(args.rgb_dir, rgb_file)
            thermal_path = os.path.join(args.thermal_dir, thermal_file)

            output_path = os.path.join(args.output, f'vis_{rgb_file}')
            tester.visualize_predictions(rgb_path, thermal_path, output_path)

        print(f"\nVisualizations saved to {args.output}/")
    else:
        print("Please provide --rgb_dir and --thermal_dir for testing")
        print("Use --visualize flag to save visualizations")


if __name__ == '__main__':
    main()
