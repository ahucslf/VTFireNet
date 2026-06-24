"""
Evaluation metrics for fire detection
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from .postprocess import box_iou


class DetectionMetrics:
    """
    Metrics for evaluating fire detection performance
    Computes precision, recall, mAP, and F1 score
    """

    def __init__(self, num_classes=1, iou_thresholds=None):
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.reset()

    def reset(self):
        """Reset accumulated metrics"""
        self.predictions = []
        self.targets = []

    def update(self, predictions, targets):
        """
        Update metrics with new batch
        Args:
            predictions: List of predictions for each image [N, 6] (x1, y1, x2, y2, conf, cls)
            targets: List of targets for each image [{'boxes': [...], 'labels': [...]}]
        """
        self.predictions.extend(predictions)
        self.targets.extend(targets)

    def compute_ap(self, recalls, precisions):
        """Compute average precision using 11-point interpolation"""
        # Add sentinel values
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([0.0], precisions, [0.0]))

        # Compute precision envelope
        for i in range(precisions.size - 1, 0, -1):
            precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])

        # Calculate area under curve
        indices = np.where(recalls[1:] != recalls[:-1])[0]
        ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])

        return ap

    def evaluate(self):
        """
        Compute final metrics
        Returns:
            Dictionary with evaluation results
        """
        if len(self.predictions) == 0:
            return {
                'mAP': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }

        all_ap = []

        for iou_th in self.iou_thresholds:
            aps = []
            for cls in range(self.num_classes):
                # Collect predictions and targets for this class
                cls_preds = []
                cls_targets = []

                for pred, target in zip(self.predictions, self.targets):
                    if pred.numel() > 0:
                        mask = pred[:, 5] == cls
                        cls_preds.append(pred[mask])

                    if target['boxes'].numel() > 0:
                        mask = target['labels'] == cls
                        cls_targets.append(target['boxes'][mask])

                if len(cls_preds) == 0 or len(cls_targets) == 0:
                    continue

                # Concatenate all
                all_pred_boxes = torch.cat(cls_preds, dim=0)
                all_pred_scores = all_pred_boxes[:, 4]
                all_pred_cls = all_pred_boxes[:, :4]

                num_gt = sum(len(t) for t in cls_targets)

                if len(all_pred_boxes) == 0 or num_gt == 0:
                    continue

                # Sort by score
                sorted_idx = all_pred_scores.argsort(descending=True)
                all_pred_cls = all_pred_cls[sorted_idx]
                all_pred_scores = all_pred_scores[sorted_idx]

                # Compute TP, FP
                tp = torch.zeros(len(all_pred_cls))
                fp = torch.zeros(len(all_pred_cls))
                gt_matched = set()

                for pred_idx, pred_box in enumerate(all_pred_cls):
                    max_iou = 0
                    max_iou_idx = -1

                    for tgt_idx, tgt_boxes in enumerate(cls_targets):
                        for gt_idx, gt_box in enumerate(tgt_boxes):
                            iou = box_iou(pred_box.unsqueeze(0), gt_box.unsqueeze(0))[0, 0]
                            if iou > max_iou:
                                max_iou = iou
                                max_iou_idx = (tgt_idx, gt_idx)

                    if max_iou >= iou_th and max_iou_idx not in gt_matched:
                        tp[pred_idx] = 1
                        gt_matched.add(max_iou_idx)
                    else:
                        fp[pred_idx] = 1

                # Compute precision and recall
                tp_cumsum = tp.cumsum(0)
                fp_cumsum = fp.cumsum(0)

                recalls = tp_cumsum / num_gt
                precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

                ap = self.compute_ap(recalls.cpu().numpy(), precisions.cpu().numpy())
                aps.append(ap)

            all_ap.append(np.mean(aps) if aps else 0.0)

        # Compute final metrics
        mAP = np.mean(all_ap)

        # Compute overall precision and recall at IoU=0.5
        iou_th = 0.5
        tp_total = 0
        fp_total = 0
        fn_total = 0

        for pred, target in zip(self.predictions, self.targets):
            if pred.numel() > 0 and target['boxes'].numel() > 0:
                for pred_box in pred:
                    matched = False
                    for gt_box in target['boxes']:
                        iou = box_iou(pred_box[:4].unsqueeze(0), gt_box.unsqueeze(0))[0, 0]
                        if iou >= iou_th:
                            tp_total += 1
                            matched = True
                            break
                    if not matched:
                        fp_total += 1

                fn_total += len(target['boxes']) - tp_total

        precision = tp_total / (tp_total + fp_total + 1e-10)
        recall = tp_total / (tp_total + fn_total + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        return {
            'mAP': mAP,
            'mAP50': all_ap[0] if all_ap else 0.0,
            'mAP75': all_ap[5] if len(all_ap) > 5 else 0.0,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricLogger:
    """Logger for tracking metrics during training"""

    def __init__(self):
        self.metrics = {}
        self.history = {}

    def update(self, **kwargs):
        """Update metrics with new values"""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = AverageMeter()
                self.history[key] = []
            self.metrics[key].update(value)
            self.history[key].append(value)

    def get_avg(self, key=None):
        """Get average value for a metric or all metrics"""
        if key:
            return self.metrics[key].avg if key in self.metrics else 0.0
        return {k: v.avg for k, v in self.metrics.items()}

    def get_current(self, key=None):
        """Get current value for a metric or all metrics"""
        if key:
            return self.metrics[key].val if key in self.metrics else 0.0
        return {k: v.val for k, v in self.metrics.items()}

    def reset(self):
        """Reset all metrics"""
        for meter in self.metrics.values():
            meter.reset()
        self.history = {}

    def summary(self):
        """Get summary of all metrics"""
        summary = {}
        for key, meter in self.metrics.items():
            summary[key] = {
                'avg': meter.avg,
                'min': min(self.history[key]) if key in self.history else 0,
                'max': max(self.history[key]) if key in self.history else 0,
                'last': self.history[key][-1] if key in self.history and self.history[key] else 0
            }
        return summary
