"""
核心代码整合文件
================
将项目中模型定义、蒸馏损失、数据加载、配置常量整合为一个自包含的 Python 文件。

来源说明:
  - TeacherCNN 类、count_parameters 函数    <- models/resnet.py
  - SmallCNN 类、count_parameters 函数      <- models/student.py
  - distillation_loss 函数                   <- train_student_distill.py
  - get_cifar10_loaders 函数                 <- utils/dataset.py
  - 配置常量 (DEVICE, CIFAR10_MEAN, 等)      <- config.py
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# ============================================================
# 配置常量  (来源: config.py)
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0  # Windows 兼容
PIN_MEMORY = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)


# ============================================================
# 教师模型 TeacherCNN  (来源: models/resnet.py)
# ============================================================
class TeacherCNN(nn.Module):
    """
    教师模型: 4层卷积块 + GAP

    结构:
        Block 1: Conv(3->64) + Conv(64->64) + BN + ReLU + MaxPool  -> 16x16
        Block 2: Conv(64->128) + Conv(128->128) + BN + ReLU + MaxPool  -> 8x8
        Block 3: Conv(128->256) + BN + ReLU + MaxPool  -> 4x4
        Block 4: Conv(256->512) + BN + ReLU
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


# ============================================================
# 学生模型 SmallCNN  (来源: models/student.py)
# ============================================================
class SmallCNN(nn.Module):
    """
    学生模型: 3层卷积 + GAP

    结构:
        Block 1: Conv(3->32) + BN + ReLU + MaxPool  -> 16x16
        Block 2: Conv(32->64) + BN + ReLU + MaxPool  -> 8x8
        Block 3: Conv(64->128) + BN + ReLU
        GAP -> Dropout(0.3) -> FC(128 -> 10)
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


# ============================================================
# 辅助函数  (来源: models/resnet.py / models/student.py)
# ============================================================
def count_parameters(model):
    """统计模型可训练参数总量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# 蒸馏损失函数  (来源: train_student_distill.py)
# ============================================================
def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    """
    知识蒸馏损失: L = (1-alpha)*CE + alpha*T^2*KL(softmax(s/T) || softmax(t/T))

    Args:
        student_logits:  学生模型输出 logits
        teacher_logits:  教师模型输出 logits
        labels:          真实标签 (硬标签)
        T:               温度参数
        alpha:           软标签损失权重

    Returns:
        loss:        总蒸馏损失
        hard_loss:   硬标签损失 (标量)
        soft_loss:   软标签损失 (标量)
    """
    hard_loss = F.cross_entropy(student_logits, labels)
    soft_teacher = F.softmax(teacher_logits / T, dim=-1)
    soft_student = F.log_softmax(student_logits / T, dim=-1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T * T)
    loss = (1 - alpha) * hard_loss + alpha * soft_loss
    return loss, hard_loss.item(), soft_loss.item()


# ============================================================
# 数据加载函数  (来源: utils/dataset.py)
# ============================================================
def get_cifar10_loaders(batch_size=128, num_workers=None, pin_memory=None):
    """
    获取 CIFAR-10 训练集和测试集的 DataLoader

    数据预处理:
        训练集: RandomCrop(32, padding=4) + RandomHorizontalFlip + Normalize
        测试集: Normalize only

    Args:
        batch_size:   批大小 (默认 128)
        num_workers:  数据加载线程数 (默认使用全局配置 NUM_WORKERS)
        pin_memory:   是否锁页内存 (默认使用全局配置 PIN_MEMORY)

    Returns:
        trainloader, testloader, num_classes(=10)
    """
    if num_workers is None:
        num_workers = NUM_WORKERS
    if pin_memory is None:
        pin_memory = PIN_MEMORY

    # 训练集数据增强
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # 测试集: 仅标准化
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform_test
    )

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return trainloader, testloader, 10


# ============================================================
# 自测试代码
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Core Code Verification")
    print("=" * 60)
    print(f"  Device: {DEVICE}")

    # --- Teacher Model ---
    teacher = TeacherCNN(num_classes=10)
    tp = count_parameters(teacher)
    print(f"\n  [TeacherCNN]")
    print(f"    Parameters: {tp:,} ({tp/1e6:.2f}M)")
    dummy = torch.randn(2, 3, 32, 32)
    out = teacher(dummy)
    print(f"    Input: {dummy.shape} -> Output: {out.shape}")
    assert out.shape == (2, 10), f"Expected (2, 10), got {out.shape}"

    # Count conv layers
    conv_count = sum(1 for m in teacher.modules() if isinstance(m, nn.Conv2d))
    print(f"    Conv layers: {conv_count}")
    print(f"    FC layers: {sum(1 for m in teacher.modules() if isinstance(m, nn.Linear))}")

    # --- Student Model ---
    student = SmallCNN(num_classes=10)
    sp = count_parameters(student)
    print(f"\n  [SmallCNN]")
    print(f"    Parameters: {sp:,} ({sp/1e3:.1f}K)")
    out = student(dummy)
    print(f"    Input: {dummy.shape} -> Output: {out.shape}")
    assert out.shape == (2, 10), f"Expected (2, 10), got {out.shape}"

    conv_count_s = sum(1 for m in student.modules() if isinstance(m, nn.Conv2d))
    print(f"    Conv layers: {conv_count_s}")

    # --- Ratio ---
    print(f"\n  [Ratio] Teacher/Student = {tp/sp:.1f} : 1")
    print(f"  [Ratio] Student = {sp/tp*100:.1f}% of Teacher")

    # --- Distillation Loss ---
    print(f"\n  [Distillation Loss]")
    labels = torch.tensor([3, 5])  # cat, dog
    t_logits = teacher(dummy)
    s_logits = student(dummy)

    loss, hard, soft = distillation_loss(s_logits, t_logits, labels, T=4.0, alpha=0.7)
    print(f"    T=4.0, alpha=0.7: total={loss.item():.4f}, hard={hard:.4f}, soft={soft:.4f}")

    loss2, hard2, soft2 = distillation_loss(s_logits, t_logits, labels, T=2.0, alpha=0.7)
    print(f"    T=2.0, alpha=0.7: total={loss2.item():.4f}, hard={hard2:.4f}, soft={soft2:.4f}")

    # --- CIFAR-10 Normalization Constants ---
    print(f"\n  [CIFAR-10 Config]")
    print(f"    Mean: {CIFAR10_MEAN}")
    print(f"    Std:  {CIFAR10_STD}")
    print(f"    Classes: {len(CIFAR10_CLASSES)}")

    print(f"\n{'=' * 60}")
    print("  All verifications passed!")
    print(f"{'=' * 60}")
