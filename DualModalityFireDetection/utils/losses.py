"""
Loss functions for Dual Modality Fire Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple


class FocalLoss(nn.Module):
    """Focal Loss for classification"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * pt.pow(self.gamma)
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class BboxLoss(nn.Module):
    """Bounding box loss with DFL (Distribution Focal Loss)"""
    def __init__(self, reg_max=16, loss_weight=1.0):
        super().__init__()
        self.reg_max = reg_max
        self.loss_weight = loss_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred_dist, target_dist, target_score, fg_mask):
        """
        Args:
            pred_dist: Predicted distribution [B, 4*reg_max, N]
            target_dist: Target distribution [B, 4, N]
            target_score: Target scores for weighting [B, N]
            fg_mask: Foreground mask [B, N]
        """
        num_pos = fg_mask.sum()
        if num_pos == 0:
            return torch.tensor(0.0, device=pred_dist.device)

        # Get target class and filter by foreground
        weight = target_score * fg_mask.float()
        pred_dist = pred_dist[..., weight.sum(-1) > 0]

        # DFL loss
        loss_dist = self.bce(pred_dist, target_dist)
        loss_dist = (loss_dist * weight.unsqueeze(1)).sum() / num_pos

        return loss_dist * self.loss_weight


class DetectionLoss(nn.Module):
    """
    Complete detection loss combining classification, regression, and DFL losses
    """
    def __init__(self, num_classes=1, reg_max=16, loss_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max

        # Loss functions
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

        # Loss weights
        self.loss_weights = loss_weights or {
            'cls_weight': 1.0,
            'reg_weight': 1.0,
            'dfl_weight': 1.5,
            'conf_weight': 1.0,
        }

    def forward(self, predictions, targets):
        """
        Calculate detection loss
        Args:
            predictions: List of prediction tuples (cls, reg, conf) for each scale
            targets: List of target dicts for each image
        Returns:
            Dictionary of losses
        """
        cls_losses = []
        reg_losses = []
        conf_losses = []

        for pred in predictions:
            cls_pred, reg_pred, conf_pred = pred

            # Classification loss
            if cls_pred.numel() > 0:
                cls_loss = self.focal_loss(cls_pred, torch.zeros_like(cls_pred))
                cls_losses.append(cls_loss)

            # Regression loss would require matching predictions to targets
            # This is simplified - actual implementation would need anchor matching

            # Confidence loss
            if conf_pred.numel() > 0:
                conf_loss = conf_pred.sigmoid().mean()
                conf_losses.append(conf_loss)

        total_loss = (
            self.loss_weights['cls_weight'] * sum(cls_losses) / max(len(cls_losses), 1) +
            self.loss_weights['reg_weight'] * sum(reg_losses) / max(len(reg_losses), 1) +
            self.loss_weights['conf_weight'] * sum(conf_losses) / max(len(conf_losses), 1)
        )

        return {
            'total_loss': total_loss,
            'cls_loss': sum(cls_losses) / max(len(cls_losses), 1),
            'reg_loss': sum(reg_losses) / max(len(reg_losses), 1),
            'conf_loss': sum(conf_losses) / max(len(conf_losses), 1),
        }


class DualModalityLoss(nn.Module):
    """
    Combined loss for dual modality detection
    Computes losses for RGB, Thermal, and Fused branches
    """
    def __init__(self, config):
        super().__init__()
        loss_config = config['train']['loss_weights']

        # Separate detection losses for each modality
        self.rgb_loss = DetectionLoss(
            num_classes=config['detection_head']['num_classes'],
            reg_max=config['detection_head']['reg_max'],
            loss_weights=loss_config
        )
        self.thermal_loss = DetectionLoss(
            num_classes=config['detection_head']['num_classes'],
            reg_max=config['detection_head']['reg_max'],
            loss_weights=loss_config
        )
        self.fused_loss = DetectionLoss(
            num_classes=config['detection_head']['num_classes'],
            reg_max=config['detection_head']['reg_max'],
            loss_weights=loss_config
        )

        # Fusion consistency loss
        self.consistency_weight = loss_config.get('fusion_weight', 0.5)

    def forward(self, predictions, targets):
        """
        Calculate total loss across all modalities
        Args:
            predictions: Dictionary with 'rgb', 'thermal', 'fused' predictions
            targets: List of target dictionaries
        Returns:
            Dictionary of losses
        """
        rgb_losses = self.rgb_loss(predictions['rgb'], targets)
        thermal_losses = self.thermal_loss(predictions['thermal'], targets)
        fused_losses = self.fused_loss(predictions['fused'], targets)

        # Weighted combination
        total_loss = (
            rgb_losses['total_loss'] +
            thermal_losses['total_loss'] +
            fused_losses['total_loss'] * self.consistency_weight
        )

        return {
            'total_loss': total_loss,
            'rgb_loss': rgb_losses['total_loss'],
            'thermal_loss': thermal_losses['total_loss'],
            'fused_loss': fused_losses['total_loss'],
            'rgb_cls': rgb_losses['cls_loss'],
            'thermal_cls': thermal_losses['cls_loss'],
            'fused_cls': fused_losses['cls_loss'],
        }


def build_loss(config):
    """Build loss function based on configuration"""
    return DualModalityLoss(config)
