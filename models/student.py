"""
学生模型: SmallCNN (CPU优化版)

更轻量的3层CNN，使用全局平均池化替代大FC层，大幅减少参数量。

参数量: ~220K (教师模型的 1/4)
预期准确率: 65-75% (直接训练) / 70-78% (蒸馏)
"""
import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """
    学生模型: 3层CNN + GAP

    结构:
        conv1(3→32)  + BN + ReLU + MaxPool(2x2)  -> 16x16
        conv2(32→64) + BN + ReLU + MaxPool(2x2)  -> 8x8
        conv3(64→128) + BN + ReLU
        GAP -> FC(128 -> 10)
    """
    def __init__(self, num_classes=10):
        super(SmallCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 3 -> 32, 32x32 -> 16x16
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 32 -> 64, 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 64 -> 128, 8x8
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = SmallCNN(num_classes=10)
    params = count_parameters(model)
    print(f"SmallCNN params: {params:,} ({params/1e6:.2f}M)")

    dummy = torch.randn(2, 3, 32, 32)
    out = model(dummy)
    print(f"Input: {dummy.shape} -> Output: {out.shape}")
    assert out.shape == (2, 10)
    print("OK")
