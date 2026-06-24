"""
Cross-modal Transformer Fusion Module
Implements the cross-attention mechanism for RGB and Thermal feature fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class PixelAdaptiveFusion(nn.Module):
    """
    Pixel-adaptive Fusion (PAF) module
    Adaptively combines features at pixel level using learnable weights
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, rgb_feat, thermal_feat):
        """
        Fuse RGB and Thermal features with learned pixel-wise weights
        Args:
            rgb_feat: RGB feature tensor [B, C, H, W]
            thermal_feat: Thermal feature tensor [B, C, H, W]
        Returns:
            fused: Fused feature tensor [B, C, H, W]
        """
        concat = torch.cat([rgb_feat, thermal_feat], dim=1)
        weights = self.conv(concat)
        w1, w2 = weights[:, 0:1], weights[:, 1:2]
        fused = w1 * rgb_feat + w2 * thermal_feat
        return fused


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head Cross Attention for cross-modal feature fusion
    Thermal Query interacts with RGB Key-Value pairs and vice versa
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.scale = self.head_dim ** -0.5

        # Query, Key, Value projections for thermal query with RGB context
        self.q_thermal_from_rgb = nn.Linear(embed_dim, embed_dim)
        self.k_rgb = nn.Linear(embed_dim, embed_dim)
        self.v_rgb = nn.Linear(embed_dim, embed_dim)

        # Query, Key, Value projections for RGB query with thermal context
        self.q_rgb_from_thermal = nn.Linear(embed_dim, embed_dim)
        self.k_thermal = nn.Linear(embed_dim, embed_dim)
        self.v_thermal = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

        # Learnable temperature parameters for attention scaling
        self.temp_thermal = nn.Parameter(torch.ones(1))
        self.temp_rgb = nn.Parameter(torch.ones(1))

    def forward(self, rgb_feat, thermal_feat):
        """
        Cross-modal attention fusion
        Args:
            rgb_feat: RGB features [B, C, H, W] or [B, L, C] where L = H*W
            thermal_feat: Thermal features [B, C, H, W] or [B, L, C]
        Returns:
            fused_feat: Fused features from both directions
        """
        B, C, H, W = rgb_feat.shape

        # Reshape to sequence format [B, H*W, C]
        rgb_seq = rgb_feat.flatten(2).transpose(1, 2)
        thermal_seq = thermal_feat.flatten(2).transpose(1, 2)

        # ----- Thermal Query with RGB Key-Value (Cross-attention 1) -----
        q_t = self.q_thermal_from_rgb(thermal_seq)
        k_r = self.k_rgb(rgb_seq)
        v_r = self.v_rgb(rgb_seq)

        # Reshape for multi-head attention
        q_t = q_t.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k_r = k_r.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_r = v_r.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with temperature
        attn_t = (q_t @ k_r.transpose(-2, -1)) * (self.scale * self.temp_thermal)
        attn_t = F.softmax(attn_t, dim=-1)
        attn_t = self.attn_drop(attn_t)
        thermal_cross = (attn_t @ v_r).transpose(1, 2).reshape(B, -1, self.embed_dim)

        # ----- RGB Query with Thermal Key-Value (Cross-attention 2) -----
        q_r = self.q_rgb_from_thermal(rgb_seq)
        k_t = self.k_thermal(thermal_seq)
        v_t = self.v_thermal(thermal_seq)

        q_r = q_r.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k_t = k_t.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_t = v_t.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_r = (q_r @ k_t.transpose(-2, -1)) * (self.scale * self.temp_rgb)
        attn_r = F.softmax(attn_r, dim=-1)
        attn_r = self.attn_drop(attn_r)
        rgb_cross = (attn_r @ v_t).transpose(1, 2).reshape(B, -1, self.embed_dim)

        # Project back and combine
        thermal_cross = self.proj_drop(self.proj(thermal_cross))
        rgb_cross = self.proj_drop(self.proj(rgb_cross))

        # Reshape back to spatial format [B, C, H, W]
        thermal_cross = thermal_cross.transpose(1, 2).reshape(B, C, H, W)
        rgb_cross = rgb_cross.transpose(1, 2).reshape(B, C, H, W)

        return thermal_cross, rgb_cross


class CrossModalTransformerBlock(nn.Module):
    """
    Cross-modal Transformer Block with FFN
    Contains cross-attention and feed-forward network
    """
    def __init__(self, embed_dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.cross_attn = MultiHeadCrossAttention(embed_dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),  # GELU activation as shown in architecture
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

        # Residual connections with learnable weights
        self.alpha1 = nn.Parameter(torch.zeros(1))
        self.alpha2 = nn.Parameter(torch.zeros(1))

    def forward(self, rgb_feat, thermal_feat):
        """
        Forward pass through transformer block
        """
        B, C, H, W = rgb_feat.shape
        residual_t = thermal_feat
        residual_r = rgb_feat

        # Cross-attention with residual
        thermal_cross, rgb_cross = self.cross_attn(rgb_feat, thermal_feat)

        # Reshape for layer norm
        thermal_in = thermal_feat + self.alpha1 * thermal_cross
        rgb_in = rgb_feat + self.alpha1 * rgb_cross

        thermal_seq = thermal_in.flatten(2).transpose(1, 2)
        rgb_seq = rgb_in.flatten(2).transpose(1, 2)

        # MLP with residual
        thermal_out = thermal_seq + self.alpha2 * self.mlp(self.norm1(thermal_seq))
        rgb_out = rgb_seq + self.alpha2 * self.mlp(self.norm1(rgb_seq))

        # Reshape back to spatial format
        thermal_out = thermal_out.transpose(1, 2).reshape(B, C, H, W)
        rgb_out = rgb_out.transpose(1, 2).reshape(B, C, H, W)

        return thermal_out, rgb_out


class CrossModalTransformerFusion(nn.Module):
    """
    Cross-modal Transformer Fusion Module
    Stacks multiple transformer blocks for deep feature fusion
    """
    def __init__(self, embed_dim=256, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossModalTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # Final fusion with Pixel-adaptive Fusion
        self.paf = PixelAdaptiveFusion(embed_dim)

    def forward(self, rgb_features, thermal_features):
        """
        Apply cross-modal transformer fusion to multi-scale features
        Args:
            rgb_features: List of RGB features [B, C, H, W]
            thermal_features: List of Thermal features [B, C, H, W]
        Returns:
            fused_features: List of fused features for each scale level
        """
        fused_features = []

        for rgb_feat, thermal_feat in zip(rgb_features, thermal_features):
            # Pass through transformer blocks
            rgb_fused = rgb_feat
            thermal_fused = thermal_feat

            for layer in self.layers:
                thermal_fused, rgb_fused = layer(rgb_fused, thermal_fused)

            # Apply pixel-adaptive fusion
            fused = self.paf(rgb_fused, thermal_fused)
            fused_features.append(fused)

        return fused_features


def build_transformer_fusion(config):
    """Build transformer fusion module based on configuration"""
    transformer_config = config['transformer_fusion']
    return CrossModalTransformerFusion(
        embed_dim=transformer_config['embed_dim'],
        num_heads=transformer_config['num_heads'],
        num_layers=transformer_config['num_layers'],
        dropout=transformer_config['dropout']
    )
