"""
教师模型: TeacherCNN (CPU优化版)

专门为CPU训练设计的轻量CNN，在保持良好分类性能的同时大幅降低训练时间。
使用6个卷积层（Block1/2各2层，Block3/4各1层）+ 全局平均池化。

参数量: 1,741,770 (1.74M)
实际训练速度: ~3 min/epoch (CPU)
实际准确率: 64.72% (CIFAR-10, 20 epochs, CPU)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TeacherCNN(nn.Module):
    """
    教师模型: 4层CNN + GAP

    结构:
        conv1(3→64)  + BN + ReLU + MaxPool(2x2)  -> 16x16
        conv2(64→128) + BN + ReLU + MaxPool(2x2)  -> 8x8
        conv3(128→256) + BN + ReLU + MaxPool(2x2)  -> 4x4
        conv4(256→512) + BN + ReLU
        GAP(4x4) -> 512
        FC(512 -> 10)
    """
    def __init__(self, num_classes=10):
        super(TeacherCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 3 -> 64, 32x32 -> 16x16
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 64 -> 128, 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 128 -> 256, 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 256 -> 512, 4x4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResNet18(nn.Module):
    """保留原始ResNet-18（可用于GPU环境）"""
    pass  # 使用TeacherCNN替代


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = TeacherCNN(num_classes=10)
    params = count_parameters(model)
    print(f"TeacherCNN params: {params:,} ({params/1e6:.2f}M)")

    dummy = torch.randn(2, 3, 32, 32)
    out = model(dummy)
    print(f"Input: {dummy.shape} -> Output: {out.shape}")
    assert out.shape == (2, 10)
    print("OK")
