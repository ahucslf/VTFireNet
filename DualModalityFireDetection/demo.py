"""
Demo script for Dual Modality Fire Detection Network
Shows how to use the model for inference
"""

import torch
import numpy as np
import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config
from models import build_model


def create_dummy_images(size=640):
    """Create dummy RGB and thermal images for demonstration"""
    # RGB image (3 channels)
    rgb_img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    # Add some red-ish regions to simulate fire-like appearance
    rgb_img[100:200, 100:200, 0] = 255
    rgb_img[100:200, 100:200, 1] = 100
    rgb_img[100:200, 100:200, 2] = 0

    # Thermal image (1 channel)
    thermal_img = np.random.randint(50, 200, (size, size), dtype=np.uint8)
    # Add hot region
    thermal_img[100:200, 100:200] = 255

    return rgb_img, thermal_img


def preprocess_images(rgb_img, thermal_img, img_size=640):
    """Preprocess images for the model"""
    # Resize
    rgb_img = cv2.resize(rgb_img, (img_size, img_size))
    thermal_img = cv2.resize(thermal_img, (img_size, img_size))

    # Convert to tensor format [B, C, H, W]
    rgb_tensor = torch.from_numpy(rgb_img.transpose(2, 0, 1)).float() / 255.0
    thermal_tensor = torch.from_numpy(thermal_img).float() / 255.0

    # Normalize RGB
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    rgb_tensor = (rgb_tensor - mean) / std

    # Normalize thermal and repeat to 3 channels
    thermal_tensor = (thermal_tensor - 0.5) / 0.5
    thermal_tensor = thermal_tensor.unsqueeze(0).repeat(3, 1, 1)

    # Add batch dimension
    rgb_tensor = rgb_tensor.unsqueeze(0)
    thermal_tensor = thermal_tensor.unsqueeze(0)

    return rgb_tensor, thermal_tensor


def main():
    """Demo main function"""
    print("=" * 60)
    print("Dual Modality Fire Detection Network - Demo")
    print("=" * 60)

    # Load configuration
    config = Config.get_config()
    print("\nConfiguration loaded successfully!")

    # Build model
    print("\nBuilding model...")
    model = build_model(config)
    print("Model built successfully!")

    # Get model info
    model_info = model.model.get_model_info()
    print(f"\nModel Information:")
    print(f"  Total Parameters: {model_info['total_params']:,}")
    print(f"  Trainable Parameters: {model_info['trainable_params']:,}")
    print(f"  Model Size: {model_info['model_size_mb']:.2f} MB")
    print(f"  Backbone Parameters: {model_info['backbone_params']:,}")
    print(f"  Transformer Parameters: {model_info['transformer_params']:,}")
    print(f"  Neck Parameters: {model_info['neck_params']:,}")
    print(f"  Head Parameters: {model_info['head_params']:,}")

    # Create dummy images
    print("\nCreating dummy images...")
    rgb_img, thermal_img = create_dummy_images()

    # Preprocess
    print("Preprocessing images...")
    rgb_tensor, thermal_tensor = preprocess_images(rgb_img, thermal_img)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Move model and inputs to device
    model = model.to(device)
    rgb_tensor = rgb_tensor.to(device)
    thermal_tensor = thermal_tensor.to(device)

    # Inference
    print("\nRunning inference...")
    model.model.eval()
    with torch.no_grad():
        outputs = model(rgb_tensor, thermal_tensor)

    print("\nInference completed!")
    print("\nOutput Keys:")
    for key in outputs.keys():
        if isinstance(outputs[key], dict):
            print(f"  {key}: (nested dict)")
            for subkey in outputs[key].keys():
                if isinstance(outputs[key][subkey], list):
                    print(f"    {subkey}: list of {len(outputs[key][subkey])} items")
        else:
            print(f"  {key}: {type(outputs[key])}")

    # Print feature shapes
    if 'features' in outputs:
        print("\nFeature Shapes:")
        for modality, features in outputs['features'].items():
            print(f"  {modality}:")
            for i, feat in enumerate(features):
                print(f"    P{i+3}: {feat.shape}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)

    return model, outputs


if __name__ == '__main__':
    model, outputs = main()
