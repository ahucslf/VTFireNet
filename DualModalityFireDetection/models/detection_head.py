"""
Dual Branch Detection Head for Fire Detection
Outputs classification, bounding box regression, and confidence scores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
import math


class DFL(nn.Module):
    """
    Distribution Focal Loss for bounding box regression
    Converts box predictions to a more stable representation
    """
    def __init__(self, c1=16):
        super().__init__()
        self.c1 = c1
        self.conv = nn.Conv2d(c1, 1, 1, bias=False)
        x = torch.arange(c1, dtype=torch.float)
        self.register_buffer('project', x.view(1, c1, 1, 1))

    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)
        x = torch.matmul(x, self.project.view(1, self.c1, 1, 1).repeat(1, 1, a, 1))
        return x.view(b, 4, a)


class DetectionHead(nn.Module):
    """
    Detection Head for single branch
    Outputs classification scores, bbox regression, and objectness
    """
    def __init__(self, num_classes=1, in_channels=256, reg_max=16, num_groups=1):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.reg_max = reg_max
        self.num_groups = num_groups

        # Classification branch
        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, num_classes, 1)
        )

        # Regression branch
        self.reg_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, 4 * reg_max, 1)
        )

        # Objectness/confidence branch
        self.conf_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, 1, 1)
        )

        # DFL layer for box decoding
        self.dfl = DFL(reg_max)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize detection head weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through detection head
        Args:
            x: Input features [B, C, H, W]
        Returns:
            cls_scores: Classification scores [B, num_classes, H, W]
            bbox_pred: Bbox regression [B, 4*reg_max, H, W]
            conf_scores: Objectness scores [B, 1, H, W]
        """
        cls_scores = self.cls_conv(x)
        bbox_pred = self.reg_conv(x)
        conf_scores = self.conf_conv(x)

        return cls_scores, bbox_pred, conf_scores

    def decode_boxes(self, bbox_pred, stride, anchor_points):
        """
        Decode bounding box predictions to actual coordinates
        Args:
            bbox_pred: Raw bbox predictions [B, 4*reg_max, H, W]
            stride: Feature stride relative to input
            anchor_points: Anchor point coordinates
        Returns:
            decoded_boxes: [B, 4, N] where N = H*W
        """
        # Apply DFL to get distances
        dist = self.dfl(bbox_pred)  # [B, 4, H*W]
        dist = dist.permute(0, 2, 1)  # [B, N, 4]

        # Decode using anchor points
        # Assuming dist contains [l, t, r, b] distances
        l, t, r, b = dist.split(1, dim=-1)

        decoded_boxes = torch.cat([
            anchor_points[:, :, 0] - l * stride,
            anchor_points[:, :, 1] - t * stride,
            anchor_points[:, :, 0] + r * stride,
            anchor_points[:, :, 1] + b * stride
        ], dim=-1)

        return decoded_boxes


class DualModalityDetectionHead(nn.Module):
    """
    Dual Modality Detection Head
    Two separate detection heads for RGB and Thermal branches
    Outputs detection predictions for both modalities
    """
    def __init__(self, num_classes=1, in_channels=256, reg_max=16, num_outs=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_outs = num_outs

        # Detection heads for each scale and each modality
        self.rgb_heads = nn.ModuleList([
            DetectionHead(num_classes, in_channels, reg_max)
            for _ in range(num_outs)
        ])
        self.thermal_heads = nn.ModuleList([
            DetectionHead(num_classes, in_channels, reg_max)
            for _ in range(num_outs)
        ])

        # Fusion head for combined predictions
        self.fusion_heads = nn.ModuleList([
            DetectionHead(num_classes, in_channels, reg_max)
            for _ in range(num_outs)
        ])

    def forward(self, rgb_features, thermal_features, fused_features):
        """
        Forward pass through all detection heads
        Args:
            rgb_features: List of RGB features from PAFPN
            thermal_features: List of Thermal features from PAFPN
            fused_features: List of Fused features from PAFPN
        Returns:
            Dictionary with detection predictions for each modality
        """
        rgb_outputs = []
        thermal_outputs = []
        fused_outputs = []

        for i in range(self.num_outs):
            rgb_outputs.append(self.rgb_heads[i](rgb_features[i]))
            thermal_outputs.append(self.thermal_heads[i](thermal_features[i]))
            fused_outputs.append(self.fusion_heads[i](fused_features[i]))

        return {
            'rgb': rgb_outputs,  # List of (cls, reg, conf) for each scale
            'thermal': thermal_outputs,
            'fused': fused_outputs
        }

    def decode_predictions(self, predictions, strides, img_size):
        """
        Decode predictions to actual bounding boxes
        Args:
            predictions: List of raw predictions from forward
            strides: List of strides for each scale
            img_size: Input image size
        Returns:
            List of decoded boxes for each scale
        """
        decoded = []
        for pred, stride in zip(predictions, strides):
            cls_scores, bbox_pred, conf_scores = pred
            B, C, H, W = cls_scores.shape

            # Create anchor points
            y_grid, x_grid = torch.meshgrid(
                torch.arange(H, device=cls_scores.device),
                torch.arange(W, device=cls_scores.device),
                indexing='ij'
            )
            anchor_points = torch.stack([x_grid, y_grid], dim=-1).float()
            anchor_points = anchor_points.unsqueeze(0).repeat(B, 1, 1, 1)
            anchor_points = anchor_points.reshape(B, -1, 2)

            # Decode boxes
            bbox_pred = bbox_pred.reshape(B, 4, self.reg_max, H, W)
            boxes = self.decode_boxes(bbox_pred, stride, anchor_points)

            # Flatten scores
            cls_scores = cls_scores.reshape(B, self.num_classes, -1)
            conf_scores = conf_scores.reshape(B, 1, -1)

            decoded.append({
                'boxes': boxes,
                'cls_scores': cls_scores,
                'conf_scores': conf_scores,
                'stride': stride
            })

        return decoded


def build_detection_head(config):
    """Build detection head based on configuration"""
    head_config = config['detection_head']
    neck_config = config['neck']
    return DualModalityDetectionHead(
        num_classes=head_config['num_classes'],
        in_channels=neck_config['out_channels'],
        reg_max=head_config['reg_max'],
        num_outs=neck_config['num_outs']
    )
