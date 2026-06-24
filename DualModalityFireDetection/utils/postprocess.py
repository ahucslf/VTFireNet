"""
Postprocessing utilities for detection outputs
"""

import torch
import numpy as np
from typing import List, Tuple, Dict


def xyxy2xywh(x):
    """Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]"""
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def xywh2xyxy(x):
    """Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]"""
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y


def box_iou(box1, box2):
    """
    Calculate IoU between two sets of boxes
    Args:
        box1: [N, 4] in xyxy format
        box2: [M, 4] in xyxy format
    Returns:
        iou: [N, M] IoU matrix
    """
    def box_area(box):
        return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])

    area1 = box_area(box1)
    area2 = box_area(box2)

    # Intersection area
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)

    return inter / (area1[:, None] + area2 - inter)


def nms(boxes, scores, iou_threshold=0.45):
    """
    Non-Maximum Suppression
    Args:
        boxes: [N, 4] bounding boxes in xyxy format
        scores: [N] confidence scores
        iou_threshold: IoU threshold for NMS
    Returns:
        keep: Indices of boxes to keep
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    # Sort by confidence
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break

        i = order[0].item()
        keep.append(i)

        # Compute IoU
        ious = box_iou(boxes[i:i+1], boxes[order[1:]])[0]

        # Keep boxes with IoU less than threshold
        idx = (ious <= iou_threshold).nonzero(as_tuple=False).squeeze()
        if idx.numel() == 0:
            break
        order = order[idx + 1]

    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)


def non_max_suppression(predictions, conf_threshold=0.25, iou_threshold=0.45, max_det=100):
    """
    Perform NMS on detection predictions
    Args:
        predictions: Raw model predictions
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        max_det: Maximum number of detections per image
    Returns:
        List of detections for each image, each detection is [N, 6] (x1, y1, x2, y2, conf, cls)
    """
    output = []

    for pred in predictions:
        if pred is None:
            output.append(torch.zeros((0, 6)))
            continue

        # Process each scale and concatenate
        all_boxes = []
        all_scores = []
        all_cls = []

        for scale_pred in pred:
            if scale_pred is None:
                continue

            # Extract boxes, scores, classes
            boxes = scale_pred.get('boxes', [])
            cls_scores = scale_pred.get('cls_scores', [])
            conf_scores = scale_pred.get('conf_scores', [])

            if boxes.numel() == 0:
                continue

            # Compute final scores
            scores = (cls_scores.sigmoid() * conf_scores.sigmoid()).squeeze()
            scores = scores.max(dim=1)[0]

            # Filter by confidence
            mask = scores > conf_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            cls_ids = cls_scores[mask].argmax(dim=0)

            if boxes.numel() > 0:
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_cls.append(cls_ids.float())

        if not all_boxes:
            output.append(torch.zeros((0, 6)))
            continue

        # Concatenate all detections
        boxes = torch.cat(all_boxes, dim=1).squeeze(0)
        scores = torch.cat(all_scores, dim=1).squeeze(0)
        cls_ids = torch.cat(all_cls, dim=1).squeeze(0)

        # Apply NMS
        keep = nms(boxes, scores, iou_threshold)

        # Limit detections
        keep = keep[:max_det]

        # Format output
        detections = torch.cat([
            boxes[keep],
            scores[keep].unsqueeze(1),
            cls_ids[keep].unsqueeze(1)
        ], dim=1)

        output.append(detections)

    return output


def scale_boxes(boxes, orig_shape, target_shape):
    """Scale boxes from target shape to original shape"""
    h_ratio = orig_shape[0] / target_shape[0]
    w_ratio = orig_shape[1] / target_shape[1]

    boxes[:, [0, 2]] *= w_ratio
    boxes[:, [1, 3]] *= h_ratio

    return boxes


def clip_boxes(boxes, shape):
    """Clip boxes to image boundaries"""
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, shape[0])
    return boxes
