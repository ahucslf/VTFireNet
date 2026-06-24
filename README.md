# VTFireNet


<img width="2048" height="2048" alt="跨模态Transformer融合模块详解_带输入" src="https://github.com/user-attachments/assets/eda7c6cb-2bf1-4718-b429-a0db34be3f26" />

README.md
Dual Modality Fire Detection Network
基于跨模态Transformer融合的可见光与热红外火焰目标检测网络

Network Architecture Overview
整体架构
RGB Image (640×640×3)          Thermal Image (640×640×1)
        │                              │
        │                              │
        ▼                              ▼
┌──────────────────────┐     ┌──────────────────────┐
│   CSPDarknet Backbone │     │   CSPDarknet Backbone │
│   (Shared Weights)    │◄───►│   (Shared Weights)    │
│                      │     │                       │
│   Output: P3, P4, P5  │     │   Output: P3, P4, P5  │
└──────────────────────┘     └──────────────────────┘
        │                              │
        │                              │
        ▼                              ▼
┌──────────────────────────────────────────────────────┐
│           Cross-modal Transformer Fusion             │
│                                                      │
│  ┌────────────────────────────────────────────────┐  │
│  │  Cross-Attention: Thermal Query + RGB K/V      │  │
│  │  Cross-Attention: RGB Query + Thermal K/V      │  │
│  │  + Feed-Forward Network (GELU)                 │  │
│  │  + Pixel-adaptive Fusion (PAF)                │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────┐
│              Dual Branch PAFPN Neck                  │
│                                                      │
│   ┌─────────┐   ┌─────────┐   ┌─────────┐           │
│   │ P3 Out  │   │ P4 Out  │   │ P5 Out  │           │
│   └─────────┘   └─────────┘   └─────────┘           │
└──────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────┐
│              Dual Modality Detection Heads          │
│                                                      │
│   ┌─────────────────────────────────────────────┐    │
│   │  RGB Head     │ Thermal Head │ Fused Head   │    │
│   │  (cls, reg,   │ (cls, reg,   │ (cls, reg,   │    │
│   │   conf)       │  conf)       │  conf)       │    │
│   └─────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────┘
        │
        ▼
   Fire Detections
核心模块详解
1. Dual Modality Backbone (双分支骨干网络)
使用CSPDarknet作为骨干网络，两个分支共享权重：

Stage 1-4: 逐步提取多尺度特征
SPPF: Spatial Pyramid Pooling - Fast 模块
输出特征: P3 (80×80), P4 (40×40), P5 (20×20)
2. Cross-modal Transformer Fusion (跨模态Transformer融合)
这是网络的核心创新模块：

Multi-head Cross Attention:

Thermal Query + RGB Key/Value → Thermal Fusion Features
RGB Query + Thermal Key/Value → RGB Fusion Features
Transformer Block结构:

Layer Normalization
Multi-head Cross Attention
Residual Connection (α1)
Feed Forward Network (MLP + GELU)
Residual Connection (α2)
3. Pixel-adaptive Fusion (像素自适应融合)
使用可学习的像素级权重融合两个模态的特征：

Fused = w1 * RGB_features + w2 * Thermal_features
其中 w1, w2 通过卷积网络学习得到
4. Dual Branch PAFPN (双分支路径聚合特征金字塔)
Top-down Pathway: 高级语义信息向下传递
Bottom-up Pathway: 低级细节信息向上传递
Channel Attention: 通道注意力机制
Spatial Attention: 空间注意力机制
5. Dual Modality Detection Heads (双模态检测头)
三个独立的检测头：

RGB Head: 可见光分支预测
Thermal Head: 热红外分支预测
Fused Head: 融合分支预测
每个检测头输出：

Classification: 类别置信度
Regression: 边界框回归 (使用DFL)
Objectness: 目标置信度
