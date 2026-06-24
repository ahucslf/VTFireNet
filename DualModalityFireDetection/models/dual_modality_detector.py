"""
Complete Dual Modality Fire Detection Network
Integrates Backbone, Cross-modal Transformer Fusion, PAFPN, and Detection Heads
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import math

from .backbone import DualModalityBackbone, build_backbone
from .transformer_fusion import CrossModalTransformerFusion, build_transformer_fusion
from .pafpn import DualBranchPAFPN, build_neck
from .detection_head import DualModalityDetectionHead, build_detection_head


class DualModalityFireDetector(nn.Module):
    """
    Complete Dual Modality Fire Detection Network

    Architecture:
    1. Dual Branch Backbone (CSPDarknet) - Extract features from RGB and Thermal images
    2. Cross-modal Transformer Fusion - Fuses features using cross-attention
    3. Dual Branch PAFPN - Path aggregation feature pyramid network
    4. Dual Modality Detection Heads - Outputs detection predictions

    The network processes both RGB and Thermal images through shared-weight backbone,
    fuses features using cross-modal transformer, and outputs separate predictions
    for RGB branch, Thermal branch, and fused branch.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Build components
        self.backbone = build_backbone(config)
        self.transformer_fusion = build_transformer_fusion(config)
        self.neck = build_neck(config)
        self.detection_head = build_detection_head(config)

        # Strides for each feature level
        self.strides = [8, 16, 32]

        # Register anchor points for each scale
        self.register_buffer('anchor_points', None)

    def generate_anchor_points(self, img_size, strides):
        """Generate anchor points for all scales"""
        anchor_points = []
        for i, stride in enumerate(strides):
            h = w = img_size // stride
            y_grid, x_grid = torch.meshgrid(
                torch.arange(h, device=next(self.parameters()).device),
                torch.arange(w, device=next(self.parameters()).device),
                indexing='ij'
            )
            points = torch.stack([x_grid, y_grid], dim=-1).float()
            points = points.reshape(1, -1, 2) * stride + stride / 2
            anchor_points.append(points)
        return anchor_points

    def forward(self, rgb_img, thermal_img):
        """
        Forward pass through the complete network

        Args:
            rgb_img: RGB image tensor [B, 3, H, W]
            thermal_img: Thermal image tensor [B, 1, H, W]

        Returns:
            Dictionary containing:
            - rgb_predictions: Detection predictions from RGB branch
            - thermal_predictions: Detection predictions from Thermal branch
            - fused_predictions: Detection predictions from fused branch
        """
        # 1. Backbone feature extraction
        rgb_features, thermal_features = self.backbone(rgb_img, thermal_img)

        # 2. Cross-modal Transformer Fusion
        fused_features = self.transformer_fusion(rgb_features, thermal_features)

        # 3. Dual Branch PAFPN
        neck_outputs = self.neck(fused_features, rgb_features, thermal_features)

        # 4. Detection Heads
        predictions = self.detection_head(
            neck_outputs['rgb'],
            neck_outputs['thermal'],
            neck_outputs['fused']
        )

        return {
            'rgb': predictions['rgb'],
            'thermal': predictions['thermal'],
            'fused': predictions['fused'],
            'features': {
                'rgb': neck_outputs['rgb'],
                'thermal': neck_outputs['thermal'],
                'fused': neck_outputs['fused']
            }
        }

    def get_model_info(self):
        """Get model architecture information"""
        n_params = sum(p.numel() for p in self.parameters())
        n_grad = sum(p.numel() for p in self.parameters() if p.requires_grad)

        info = {
            'total_params': n_params,
            'trainable_params': n_grad,
            'model_size_mb': n_params * 4 / 1024 / 1024,  # Assuming float32
        }

        # Count parameters per component
        info['backbone_params'] = sum(p.numel() for p in self.backbone.parameters())
        info['transformer_params'] = sum(p.numel() for p in self.transformer_fusion.parameters())
        info['neck_params'] = sum(p.numel() for p in self.neck.parameters())
        info['head_params'] = sum(p.numel() for p in self.detection_head.parameters())

        return info

    def load_pretrained(self, checkpoint_path):
        """Load pretrained weights"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in checkpoint:
            self.load_state_dict(checkpoint['model'], strict=False)
        else:
            self.load_state_dict(checkpoint, strict=False)
        print(f"Loaded pretrained weights from {checkpoint_path}")

    def save_checkpoint(self, path, epoch, optimizer=None, **kwargs):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model': self.state_dict(),
        }
        if optimizer is not None:
            checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint.update(kwargs)
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")


class ModelWrapper(nn.Module):
    """
    Wrapper for the Dual Modality Fire Detector
    Handles preprocessing, inference, and postprocessing
    """

    def __init__(self, config, weights=None):
        super().__init__()
        self.config = config
        self.model = DualModalityFireDetector(config)

        if weights is not None:
            self.model.load_pretrained(weights)

        # Preprocessing parameters
        self.img_size = config['data']['img_size']
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.thermal_mean = torch.tensor([0.5]).view(1, 1, 1, 1)
        self.thermal_std = torch.tensor([0.5]).view(1, 1, 1, 1)

    def preprocess(self, rgb_img, thermal_img):
        """
        Preprocess images for inference
        Args:
            rgb_img: Input RGB image [B, 3, H, W] normalized to [0, 1]
            thermal_img: Input thermal image [B, 1, H, W] normalized to [0, 1]
        Returns:
            Preprocessed RGB and Thermal images
        """
        # Normalize RGB
        rgb_norm = (rgb_img - self.mean.to(rgb_img)) / self.std.to(rgb_img)

        # Normalize thermal
        thermal_norm = (thermal_img - self.thermal_mean.to(thermal_img)) / self.thermal_std.to(thermal_img)

        return rgb_norm, thermal_norm

    def forward(self, rgb_img, thermal_img):
        """Forward pass with preprocessing"""
        rgb_norm, thermal_norm = self.preprocess(rgb_img, thermal_img)
        return self.model(rgb_norm, thermal_norm)

    def inference(self, rgb_img, thermal_img, conf_threshold=0.25, iou_threshold=0.45):
        """
        Run inference and return detections
        Args:
            rgb_img: Input RGB image [B, 3, H, W]
            thermal_img: Input thermal image [B, 1, H, W]
            conf_threshold: Confidence threshold for filtering
            iou_threshold: IoU threshold for NMS
        Returns:
            List of detections for each image
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(rgb_img, thermal_img)
            # Postprocessing would be applied here
            # This would decode boxes, apply NMS, etc.
        return predictions


def build_model(config, weights=None):
    """Build complete model based on configuration"""
    model = ModelWrapper(config, weights)
    return model


# Import postprocessing utilities
from ..utils.postprocess import non_max_suppression, box_iou, xyxy2xywh, xywh2xyxy
