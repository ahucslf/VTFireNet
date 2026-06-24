# 基于跨模态Transformer融合的可见光与热红外火焰目标检测网络（VTFireNet）

<img width="2048" height="2048" alt="跨模态Transformer融合模块详解_带输入" src="https://github.com/user-attachments/assets/eda7c6cb-2bf1-4718-b429-a0db34be3f26" />



## Network Architecture Overview

### 整体架构

```
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
```

### 核心模块详解

#### 1. Dual Modality Backbone (双分支骨干网络)

使用CSPDarknet作为骨干网络，两个分支**共享权重**：

- **Stage 1-4**: 逐步提取多尺度特征
- **SPPF**: Spatial Pyramid Pooling - Fast 模块
- **输出特征**: P3 (80×80), P4 (40×40), P5 (20×20)

#### 2. Cross-modal Transformer Fusion (跨模态Transformer融合)

这是网络的核心创新模块：

**Multi-head Cross Attention:**
```
Thermal Query + RGB Key/Value → Thermal Fusion Features
RGB Query + Thermal Key/Value → RGB Fusion Features
```

**Transformer Block结构:**
- Layer Normalization
- Multi-head Cross Attention
- Residual Connection (α1)
- Feed Forward Network (MLP + GELU)
- Residual Connection (α2)

#### 3. Pixel-adaptive Fusion (像素自适应融合)

使用可学习的像素级权重融合两个模态的特征：

```
Fused = w1 * RGB_features + w2 * Thermal_features
其中 w1, w2 通过卷积网络学习得到
```

#### 4. Dual Branch PAFPN (双分支路径聚合特征金字塔)

- **Top-down Pathway**: 高级语义信息向下传递
- **Bottom-up Pathway**: 低级细节信息向上传递
- **Channel Attention**: 通道注意力机制
- **Spatial Attention**: 空间注意力机制

#### 5. Dual Modality Detection Heads (双模态检测头)

三个独立的检测头：
- **RGB Head**: 可见光分支预测
- **Thermal Head**: 热红外分支预测
- **Fused Head**: 融合分支预测

每个检测头输出：
- **Classification**: 类别置信度
- **Regression**: 边界框回归 (使用DFL)
- **Objectness**: 目标置信度

## 文件结构

```
DualModalityFireDetection/
├── configs/
│   └── config.py              # 配置文件
├── models/
│   ├── __init__.py
│   ├── backbone.py            # CSPDarknet骨干网络
│   ├── transformer_fusion.py  # 跨模态Transformer融合
│   ├── pafpn.py               # PAFPN颈部网络
│   ├── detection_head.py      # 检测头
│   └── dual_modality_detector.py  # 完整模型
├── utils/
│   ├── __init__.py
│   ├── losses.py              # 损失函数
│   ├── metrics.py             # 评估指标
│   ├── postprocess.py         # 后处理
│   ├── data_augmentation.py   # 数据增强
│   └── dataset.py             # 数据集类
├── train.py                   # 训练脚本
├── test.py                    # 测试脚本
├── inference.py               # 推理脚本
└── README.md                  # 本文档
```

## 快速开始

### 环境要求

- Python >= 3.8
- PyTorch >= 1.10
- OpenCV
- NumPy
- TensorBoard

### 安装依赖

```bash
pip install torch torchvision opencv-python numpy tensorboard tqdm
```

### 训练模型

```bash
python train.py --epochs 100 --batch_size 8 --lr 0.001
```

### 测试模型

```bash
python test.py --checkpoint checkpoints/best_model.pth --visualize
```

### 推理预测

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --rgb path/to/rgb/image.jpg \
    --thermal path/to/thermal/image.jpg \
    --output result.jpg
```

## 配置说明

在 `configs/config.py` 中可以修改：

```python
DATA_CONFIG = {
    'img_size': 640,           # 输入图像大小
    'num_classes': 1,          # 火焰类别数
    'rgb_channels': 3,        # RGB通道数
    'thermal_channels': 1,     # 热红外通道数
}

MODEL_CONFIG = {
    'backbone': {
        'depth_multiple': 1.0,   # 网络深度
        'width_multiple': 1.0,  # 网络宽度
    },
    'transformer_fusion': {
        'embed_dim': 256,       # 嵌入维度
        'num_heads': 8,         # 注意力头数
        'num_layers': 2,        # Transformer层数
    }
}
```

## 数据集格式

期望的数据格式：

```json
{
    "image_id_1": {
        "rgb_file": "rgb_001.jpg",
        "thermal_file": "thermal_001.jpg",
        "objects": [
            {
                "bbox": [x1, y1, x2, y2],
                "label": 0
            }
        ]
    }
}
```

## 性能指标

| 指标 | 描述 |
|------|------|
| mAP | 平均精度均值 |
| mAP@50 | IoU=0.5时的精度 |
| mAP@75 | IoU=0.75时的精度 |
| Precision | 精确率 |
| Recall | 召回率 |
| F1 | F1分数 |

## 技术创新点

1. **跨模态Transformer融合**: 使用双向交叉注意力机制实现RGB和热红外特征的深度融合
2. **权重共享骨干网络**: 两个模态共享CSPDarknet参数，减少参数量
3. **像素自适应融合**: 学习像素级的融合权重，更好地保留各模态的独特信息
4. **多分支检测输出**: 同时输出三个分支的检测结果，可根据场景灵活选择

## 适用场景

- 森林火灾监测
- 工业消防安全
- 室内火灾预警
- 夜间/低光照环境下的火焰检测

## License

MIT License

## Citation

如果您在研究中使用本代码，请引用：

```
@software{dual_modality_fire_detection,
  title={Dual Modality Fire Detection Network},
  author={Your Name},
  year={2024}
}
```
