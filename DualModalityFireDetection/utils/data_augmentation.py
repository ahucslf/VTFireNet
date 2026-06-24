"""
Data augmentation for dual modality fire detection
"""

import torch
import random
import numpy as np
import cv2
from typing import Tuple, Dict, List


class Compose:
    """Compose multiple transforms together"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, rgb_img, thermal_img, targets):
        for t in self.transforms:
            rgb_img, thermal_img, targets = t(rgb_img, thermal_img, targets)
        return rgb_img, thermal_img, targets


class Resize:
    """Resize images to target size"""

    def __init__(self, size):
        self.size = size

    def __call__(self, rgb_img, thermal_img, targets):
        h, w = rgb_img.shape[:2]

        # Resize RGB
        rgb_img = cv2.resize(rgb_img, (self.size, self.size))

        # Resize Thermal
        thermal_img = cv2.resize(thermal_img, (self.size, self.size))

        # Scale bounding boxes
        scale_x = self.size / w
        scale_y = self.size / h
        targets['boxes'][:, [0, 2]] *= scale_x
        targets['boxes'][:, [1, 3]] *= scale_y

        return rgb_img, thermal_img, targets


class RandomFlip:
    """Random horizontal flip"""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, rgb_img, thermal_img, targets):
        if random.random() < self.prob:
            rgb_img = cv2.flip(rgb_img, 1)
            thermal_img = cv2.flip(thermal_img, 1)

            w = rgb_img.shape[1]
            targets['boxes'][:, [0, 2]] = w - targets['boxes'][:, [2, 0]]

        return rgb_img, thermal_img, targets


class RandomHSV:
    """Randomly change hue, saturation, and value"""

    def __init__(self, h_gain=0.015, s_gain=0.7, v_gain=0.4):
        self.h_gain = h_gain
        self.s_gain = s_gain
        self.v_gain = v_gain

    def __call__(self, rgb_img, thermal_img, targets):
        # Random hue
        r = np.random.uniform(-1, 1, 3) * [self.h_gain, self.s_gain, self.v_gain] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV))

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype('uint8')
        lut_sat = np.clip(x * r[1], 0, 255).astype('uint8')
        lut_val = np.clip(x * r[2], 0, 255).astype('uint8')

        hue = cv2.LUT(hue, lut_hue)
        sat = cv2.LUT(sat, lut_sat)
        val = cv2.LUT(val, lut_val)

        rgb_img = cv2.cvtColor(cv2.merge([hue, sat, val]), cv2.COLOR_HSV2BGR)

        return rgb_img, thermal_img, targets


class RandomScale:
    """Randomly scale the image"""

    def __init__(self, scale_range=(0.5, 1.5)):
        self.scale_range = scale_range

    def __call__(self, rgb_img, thermal_img, targets):
        scale = random.uniform(*self.scale_range)
        h, w = rgb_img.shape[:2]

        new_h, new_w = int(h * scale), int(w * scale)
        rgb_img = cv2.resize(rgb_img, (new_w, new_h))
        thermal_img = cv2.resize(thermal_img, (new_w, new_h))

        # Scale boxes
        targets['boxes'][:, [0, 2]] *= scale
        targets['boxes'][:, [1, 3]] *= scale

        return rgb_img, thermal_img, targets


class RandomTranslate:
    """Randomly translate the image"""

    def __init__(self, translate=0.1):
        self.translate = translate

    def __call__(self, rgb_img, thermal_img, targets):
        h, w = rgb_img.shape[:2]

        tx = random.uniform(-self.translate, self.translate) * w
        ty = random.uniform(-self.translate, self.translate) * h

        M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
        rgb_img = cv2.warpAffine(rgb_img, M, (w, h))
        thermal_img = cv2.warpAffine(thermal_img, M, (w, h))

        # Translate boxes
        targets['boxes'][:, [0, 2]] += tx
        targets['boxes'][:, [1, 3]] += ty

        # Clip boxes
        targets['boxes'][:, [0, 2]] = targets['boxes'][:, [0, 2]].clip(0, w)
        targets['boxes'][:, [1, 3]] = targets['boxes'][:, [1, 3]].clip(0, h)

        return rgb_img, thermal_img, targets


class Normalize:
    """Normalize images"""

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, rgb_img, thermal_img, targets):
        # Normalize RGB
        rgb_img = rgb_img.astype(np.float32) / 255.0
        rgb_img = (rgb_img - self.mean) / self.std

        # Normalize thermal
        thermal_img = thermal_img.astype(np.float32) / 255.0
        thermal_img = (thermal_img - 0.5) / 0.5

        return rgb_img, thermal_img, targets


class ToTensor:
    """Convert numpy arrays to PyTorch tensors"""

    def __call__(self, rgb_img, thermal_img, targets):
        # Convert to tensor format [C, H, W]
        rgb_img = torch.from_numpy(rgb_img.transpose(2, 0, 1))
        thermal_img = torch.from_numpy(thermal_img.transpose(2, 0, 1))

        targets['boxes'] = torch.from_numpy(targets['boxes'])
        targets['labels'] = torch.from_numpy(targets['labels'])

        return rgb_img, thermal_img, targets


def build_train_transforms(config):
    """Build training transforms from config"""
    aug_config = config['train']['augmentation']

    transforms = [
        Resize(config['data']['img_size']),
    ]

    if aug_config.get('fliplr', 0) > 0:
        transforms.append(RandomFlip(prob=aug_config['fliplr']))

    if aug_config.get('hsv_h', 0) > 0:
        transforms.append(RandomHSV(
            h_gain=aug_config['hsv_h'],
            s_gain=aug_config['hsv_s'],
            v_gain=aug_config['hsv_v']
        ))

    if aug_config.get('translate', 0) > 0:
        transforms.append(RandomTranslate(translate=aug_config['translate']))

    if aug_config.get('scale', 0) > 0:
        transforms.append(RandomScale(scale_range=(1 - aug_config['scale'], 1 + aug_config['scale'])))

    transforms.extend([
        Normalize(),
        ToTensor()
    ])

    return Compose(transforms)


def build_val_transforms(config):
    """Build validation transforms from config"""
    transforms = [
        Resize(config['data']['img_size']),
        Normalize(),
        ToTensor()
    ]
    return Compose(transforms)
