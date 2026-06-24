"""
Configuration file for Dual Modality Fire Detection Network
"""

import torch

class Config:
    # ============== Data Configuration ==============
    DATA_CONFIG = {
        'img_size': 640,           # Input image size
        'num_classes': 1,          # Fire detection (1 class)
        'rgb_channels': 3,        # RGB image channels
        'thermal_channels': 1,     # Thermal image channels
    }

    # ============== Model Configuration ==============
    MODEL_CONFIG = {
        'backbone': {
            'depth_multiple': 1.0,      # Network depth
            'width_multiple': 1.0,     # Network width
            'out_indices': (3, 4, 5),  # Output indices for P3, P4, P5
        },
        'neck': {
            'in_channels': [256, 512, 1024],  # Input channels from backbone
            'out_channels': 256,             # Output channels
            'num_outs': 3,                    # Number of output levels
        },
        'detection_head': {
            'num_classes': 1,
            'in_channels': 256,
            'reg_max': 16,
        },
        'transformer_fusion': {
            'embed_dim': 256,
            'num_heads': 8,
            'num_layers': 2,
            'dropout': 0.1,
        }
    }

    # ============== Training Configuration ==============
    TRAIN_CONFIG = {
        'batch_size': 8,
        'num_epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 0.0005,
        'momentum': 0.9,
        'num_workers': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',

        # Learning rate scheduler
        'lr_scheduler': {
            'type': 'cosine',
            'warmup_epochs': 3,
            'min_lr': 0.00001,
        },

        # Loss weights
        'loss_weights': {
            'cls_weight': 1.0,
            'reg_weight': 1.0,
            'dfl_weight': 1.5,
            'fusion_weight': 0.5,
        },

        # Data augmentation
        'augmentation': {
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 0.5,
            'mixup': 0.0,
        }
    }

    # ============== Testing Configuration ==============
    TEST_CONFIG = {
        'batch_size': 1,
        'conf_threshold': 0.25,
        'iou_threshold': 0.45,
        'max_detections': 100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    # ============== Path Configuration ==============
    PATH_CONFIG = {
        'train_rgb_dir': './data/train/rgb',
        'train_thermal_dir': './data/train/thermal',
        'train_annotations': './data/train/annotations.json',
        'val_rgb_dir': './data/val/rgb',
        'val_thermal_dir': './data/val/thermal',
        'val_annotations': './data/val/annotations.json',
        'test_rgb_dir': './data/test/rgb',
        'test_thermal_dir': './data/test/thermal',
        'checkpoint_dir': './checkpoints',
        'log_dir': './logs',
    }

    @classmethod
    def get_config(cls, category=None):
        """Get specific configuration category or all config"""
        if category:
            return getattr(cls, f'{category.upper()}_CONFIG', None)
        return {
            'data': cls.DATA_CONFIG,
            'model': cls.MODEL_CONFIG,
            'train': cls.TRAIN_CONFIG,
            'test': cls.TEST_CONFIG,
            'path': cls.PATH_CONFIG,
        }
