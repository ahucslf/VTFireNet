"""
PAFPN (Path Aggregation Feature Pyramid Network) Neck
Fuses multi-scale features from cross-modal transformer fusion
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
    """CSP Bottleneck with 3 convolutions"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class ChannelAttention(nn.Module):
    """Channel attention module for feature recalibration"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention module"""
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(out))


class PAFPNNeck(nn.Module):
    """
    Path Aggregation Feature Pyramid Network with Attention
    Processes fused multi-scale features and outputs enhanced features
    """
    def __init__(self, in_channels=[256, 512, 1024], out_channels=256, num_outs=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs

        # Lateral convolutions to reduce channel dimension
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                Conv(in_c, out_channels, 1),
                ChannelAttention(out_channels)
            ) for in_c in in_channels
        ])

        # Top-down pathway C3f blocks
        self.fpn_convs = nn.ModuleList([
            C3f(out_channels, out_channels, n=1, shortcut=True, e=0.5)
            for _ in range(len(in_channels) - 1)
        ])

        # Bottom-up pathway for PAFPN
        self.downsample_convs = nn.ModuleList([
            Conv(out_channels, out_channels, 3, 2)
            for _ in range(len(in_channels) - 1)
        ])

        self.pafpn_convs = nn.ModuleList([
            C3f(out_channels * 2, out_channels, n=1, shortcut=True, e=0.5)
            for _ in range(len(in_channels) - 1)
        ])

        # Spatial attention for each level
        self.spatial_atts = nn.ModuleList([
            SpatialAttention() for _ in range(len(in_channels))
        ])

        # Extra levels if needed
        if num_outs > len(in_channels):
            self.extra_convs = nn.ModuleList([
                Conv(in_channels[-1], out_channels, 3, 2)
            ])

    def forward(self, inputs):
        """
        Forward pass through PAFPN
        Args:
            inputs: List of multi-scale features from transformer fusion
        Returns:
            outputs: List of enhanced multi-scale features
        """
        assert len(inputs) == len(self.in_channels)

        # Lateral connections
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(inputs[i]))

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + nn.functional.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode='nearest'
            )
            laterals[i - 1] = self.fpn_convs[i - 1](laterals[i - 1])

        # Apply spatial attention
        laterals = [sp_att(lateral) for sp_att, lateral in zip(self.spatial_atts, laterals)]

        # Bottom-up pathway (PAFPN enhancement)
        for i in range(len(laterals) - 1):
            downsampled = self.downsample_convs[i](laterals[i])
            laterals[i + 1] = torch.cat([laterals[i + 1], downsampled], dim=1)
            laterals[i + 1] = self.pafpn_convs[i](laterals[i + 1])

        # Generate extra outputs if needed
        outputs = laterals[:self.num_outs]

        if self.num_outs > len(laterals):
            for i in range(self.num_outs - len(laterals)):
                if i == 0:
                    extra = self.extra_convs[0](laterals[-1])
                else:
                    extra = nn.functional.max_pool2d(extra, 2)
                outputs.append(extra)

        return outputs


class DualBranchPAFPN(nn.Module):
    """
    Dual Branch PAFPN for both RGB and Thermal branches
    Maintains separate feature pyramids for each modality after fusion
    """
    def __init__(self, in_channels=[256, 512, 1024], out_channels=256, num_outs=3):
        super().__init__()
        # Shared PAFPN for processing fused features
        self.fusion_pafpn = PAFPNNeck(in_channels, out_channels, num_outs)

        # Separate PAFPNs for RGB and Thermal branches
        self.rgb_pafpn = PAFPNNeck(in_channels, out_channels, num_outs)
        self.thermal_pafpn = PAFPNNeck(in_channels, out_channels, num_outs)

        # Fusion gate to balance fused vs branch-specific features
        self.fusion_gate = nn.Parameter(torch.ones(1))

    def forward(self, fused_features, rgb_features, thermal_features):
        """
        Process features through all branches
        Args:
            fused_features: Fused features from transformer
            rgb_features: Original RGB features
            thermal_features: Original Thermal features
        Returns:
            Dictionary with outputs for each branch
        """
        # Process through respective PAFPNs
        rgb_outputs = self.rgb_pafpn(rgb_features)
        thermal_outputs = self.thermal_pafpn(thermal_features)
        fused_outputs = self.fusion_pafpn(fused_features)

        return {
            'rgb': rgb_outputs,
            'thermal': thermal_outputs,
            'fused': fused_outputs
        }


def build_neck(config):
    """Build PAFPN neck based on configuration"""
    neck_config = config['neck']
    return DualBranchPAFPN(
        in_channels=neck_config['in_channels'],
        out_channels=neck_config['out_channels'],
        num_outs=neck_config['num_outs']
    )
