"""
Dataset classes for dual modality fire detection
"""

import os
import json
import cv2
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import numpy as np

from .data_augmentation import build_train_transforms, build_val_transforms


class DualModalityFireDataset(Dataset):
    """
    Dataset for dual modality fire detection
    Loads paired RGB and thermal images with annotations
    """

    def __init__(self, rgb_dir, thermal_dir, annotations_file,
                 img_size=640, transforms=None):
        self.rgb_dir = rgb_dir
        self.thermal_dir = thermal_dir
        self.img_size = img_size
        self.transforms = transforms

        # Load annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)

        self.image_ids = list(self.annotations.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        ann = self.annotations[img_id]

        # Load RGB image
        rgb_path = os.path.join(self.rgb_dir, ann['rgb_file'])
        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        # Load Thermal image
        thermal_path = os.path.join(self.thermal_dir, ann['thermal_file'])
        thermal_img = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
        thermal_img = cv2.cvtColor(thermal_img, cv2.COLOR_GRAY2RGB)  # Convert to 3 channels

        # Get bounding boxes and labels
        boxes = []
        labels = []
        for obj in ann['objects']:
            x1, y1, x2, y2 = obj['bbox']
            boxes.append([x1, y1, x2, y2])
            labels.append(obj.get('label', 0))  # Default to class 0 (fire)

        targets = {
            'boxes': np.array(boxes, dtype=np.float32),
            'labels': np.array(labels, dtype=np.int64),
            'image_id': img_id
        }

        # Apply transforms
        if self.transforms is not None:
            rgb_img, thermal_img, targets = self.transforms(rgb_img, thermal_img, targets)

        return rgb_img, thermal_img, targets

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for batching"""
        rgb_imgs, thermal_imgs, targets = zip(*batch)

        # Stack images
        rgb_imgs = torch.stack(rgb_imgs, dim=0)
        thermal_imgs = torch.stack(thermal_imgs, dim=0)

        # Keep targets as list (variable number of objects)
        return rgb_imgs, thermal_imgs, list(targets)


class FireDatasetBuilder:
    """Builder for creating fire detection datasets"""

    def __init__(self, config):
        self.config = config
        self.train_transforms = build_train_transforms(config)
        self.val_transforms = build_val_transforms(config)

    def build_train_dataset(self):
        """Build training dataset"""
        path_config = self.config['path']
        data_config = self.config['data']

        return DualModalityFireDataset(
            rgb_dir=path_config['train_rgb_dir'],
            thermal_dir=path_config['train_thermal_dir'],
            annotations_file=path_config['train_annotations'],
            img_size=data_config['img_size'],
            transforms=self.train_transforms
        )

    def build_val_dataset(self):
        """Build validation dataset"""
        path_config = self.config['path']
        data_config = self.config['data']

        return DualModalityFireDataset(
            rgb_dir=path_config['val_rgb_dir'],
            thermal_dir=path_config['val_thermal_dir'],
            annotations_file=path_config['val_annotations'],
            img_size=data_config['img_size'],
            transforms=self.val_transforms
        )

    def build_test_dataset(self):
        """Build test dataset (uses val transforms)"""
        path_config = self.config['path']
        data_config = self.config['data']

        return DualModalityFireDataset(
            rgb_dir=path_config['test_rgb_dir'],
            thermal_dir=path_config['test_thermal_dir'],
            annotations_file=path_config['test_annotations'],
            img_size=data_config['img_size'],
            transforms=self.val_transforms
        )

    def build_dataloader(self, dataset, batch_size, shuffle=True, num_workers=4):
        """Build dataloader for a dataset"""
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=DualModalityFireDataset.collate_fn,
            pin_memory=True
        )


def create_dataloaders(config):
    """Create all dataloaders from config"""
    builder = FireDatasetBuilder(config)

    train_config = config['train']

    dataloaders = {
        'train': builder.build_dataloader(
            builder.build_train_dataset(),
            batch_size=train_config['batch_size'],
            shuffle=True,
            num_workers=train_config['num_workers']
        )
    }

    if os.path.exists(config['path']['val_rgb_dir']):
        dataloaders['val'] = builder.build_dataloader(
            builder.build_val_dataset(),
            batch_size=train_config['batch_size'],
            shuffle=False,
            num_workers=train_config['num_workers']
        )

    return dataloaders
