"""
CSPDarknet Backbone for Dual Modality Fire Detection
Based on YOLOv8 architecture
"""

import torch
import torch.nn as nn
from typing import List, Tuple


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution with activation"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck block"""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3f(nn.Module):
    """
    CSP Bottleneck with 3 convolutions and flexible short connections
    Used as main feature extraction block in backbone
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast module"""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class CSPDarknet(nn.Module):
    """
    CSPDarknet Backbone for dual modality feature extraction
    Outputs multi-scale features: P3(80x80), P4(40x40), P5(20x20)
    """
    def __init__(self, depth_multiple=1.0, width_multiple=1.0, out_indices=(3, 4, 5)):
        super().__init__()
        self.out_indices = out_indices

        # Define depth scaling
        def make_divisible(x, divisor=8):
            return int((x + divisor / 2) // divisor * divisor)

        # Calculate channels based on width multiple
        c1 = 3
        c2 = make_divisible(64 * width_multiple)
        c3 = make_divisible(128 * width_multiple)
        c4 = make_divisible(256 * width_multiple)
        c5 = make_divisible(512 * width_multiple)
        c6 = make_divisible(1024 * width_multiple)

        # Calculate depth scaling
        n1 = max(round(3 * depth_multiple), 1)
        n2 = max(round(6 * depth_multiple), 1)
        n3 = max(round(6 * depth_multiple), 1)
        n4 = max(round(3 * depth_multiple), 1)

        # Stem
        self.stem = Conv(c1, c2, 3, 2)

        # Stage 1
        self.stage1_conv = Conv(c2, c3, 3, 2)
        self.stage1_c3f = C3f(c3, c3, n=n1)

        # Stage 2
        self.stage2_conv = Conv(c3, c4, 3, 2)
        self.stage2_c3f = C3f(c4, c4, n=n2)

        # Stage 3
        self.stage3_conv = Conv(c4, c5, 3, 2)
        self.stage3_c3f = C3f(c5, c5, n=n3)

        # Stage 4
        self.stage4_conv = Conv(c5, c6, 3, 2)
        self.stage4_c3f = C3f(c6, c6, n=n4)
        self.sppf = SPPF(c6, c6)

        # Store output channels for later use
        self.out_channels = {
            3: c4,   # P3
            4: c5,   # P4
            5: c6,   # P5 (SPP output)
        }

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass
        Returns multi-scale features at indices 3, 4, 5
        """
        outputs = []

        x = self.stem(x)

        x = self.stage1_conv(x)
        x = self.stage1_c3f(x)
        if 1 in self.out_indices:
            outputs.append(x)

        x = self.stage2_conv(x)
        x = self.stage2_c3f(x)
        if 2 in self.out_indices:
            outputs.append(x)

        x = self.stage3_conv(x)
        x = self.stage3_c3f(x)
        if 3 in self.out_indices:
            outputs.append(x)

        x = self.stage4_conv(x)
        x = self.stage4_c3f(x)
        x = self.sppf(x)
        if 4 in self.out_indices:
            outputs.append(x)

        return outputs

    def get_output_channels(self):
        """Return output channels for each feature level"""
        return self.out_channels


class DualModalityBackbone(nn.Module):
    """
    Dual Modality Backbone for RGB and Thermal images
    Shares weights between the two branches as shown in the architecture
    """
    def __init__(self, depth_multiple=1.0, width_multiple=1.0, out_indices=(3, 4, 5)):
        super().__init__()
        # Shared CSPDarknet backbone
        self.shared_backbone = CSPDarknet(
            depth_multiple=depth_multiple,
            width_multiple=width_multiple,
            out_indices=out_indices
        )

        # Modality-specific adaptation layers
        # RGB input: 3 channels -> shared backbone expects 3 channels
        # Thermal input: 1 channel -> needs adaptation

        def make_divisible(x, divisor=8):
            return int((x + divisor / 2) // divisor * divisor)

        c1 = 3  # RGB channels
        c2 = 1  # Thermal channels
        c_shared = make_divisible(64 * width_multiple)

        # Thermal channel adaptation: 1 -> 3 channels
        self.thermal_adapter = Conv(c2, c1, kernel_size=3, stride=1, padding=1)

    def forward(self, rgb_x, thermal_x):
        """
        Forward pass for both modalities
        Args:
            rgb_x: RGB image tensor [B, 3, H, W]
            thermal_x: Thermal image tensor [B, 1, H, W]
        Returns:
            rgb_features: List of RGB features [P3, P4, P5]
            thermal_features: List of Thermal features [P3, P4, P5]
        """
        # Adapt thermal input to match RGB backbone input
        thermal_adapted = self.thermal_adapter(thermal_x)

        # Extract features using shared backbone weights
        rgb_features = self.shared_backbone(rgb_x)
        thermal_features = self.shared_backbone(thermal_adapted)

        return rgb_features, thermal_features

    def get_output_channels(self):
        """Return output channels for each feature level"""
        return self.shared_backbone.get_output_channels()


def build_backbone(config):
    """Build backbone based on configuration"""
    depth_multiple = config['backbone']['depth_multiple']
    width_multiple = config['backbone']['width_multiple']
    out_indices = config['backbone']['out_indices']
    return DualModalityBackbone(depth_multiple, width_multiple, out_indices)
