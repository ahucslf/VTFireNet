"""
Utils module for Dual Modality Fire Detection
"""

from .losses import (
    FocalLoss,
    BboxLoss,
    DetectionLoss,
    DualModalityLoss,
    build_loss
)
from .postprocess import (
    xyxy2xywh,
    xywh2xyxy,
    box_iou,
    nms,
    non_max_suppression,
    scale_boxes,
    clip_boxes
)
from .metrics import (
    DetectionMetrics,
    AverageMeter,
    MetricLogger
)
from .data_augmentation import (
    Compose,
    Resize,
    RandomFlip,
    RandomHSV,
    RandomScale,
    RandomTranslate,
    Normalize,
    ToTensor,
    build_train_transforms,
    build_val_transforms
)
from .dataset import (
    DualModalityFireDataset,
    FireDatasetBuilder,
    create_dataloaders
)

__all__ = [
    # Losses
    'FocalLoss',
    'BboxLoss',
    'DetectionLoss',
    'DualModalityLoss',
    'build_loss',
    # Postprocess
    'xyxy2xywh',
    'xywh2xyxy',
    'box_iou',
    'nms',
    'non_max_suppression',
    'scale_boxes',
    'clip_boxes',
    # Metrics
    'DetectionMetrics',
    'AverageMeter',
    'MetricLogger',
    # Data augmentation
    'Compose',
    'Resize',
    'RandomFlip',
    'RandomHSV',
    'RandomScale',
    'RandomTranslate',
    'Normalize',
    'ToTensor',
    'build_train_transforms',
    'build_val_transforms',
    # Dataset
    'DualModalityFireDataset',
    'FireDatasetBuilder',
    'create_dataloaders',
]
