"""
CIFAR-10 数据加载与预处理模块

功能:
- 自动下载 CIFAR-10 数据集
- 训练集数据增强: RandomCrop + RandomHorizontalFlip
- 测试集仅做标准化处理
- 返回 PyTorch DataLoader
"""
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from config import (
    DATA_DIR, CIFAR10_MEAN, CIFAR10_STD,
    DEVICE, NUM_WORKERS, PIN_MEMORY
)


def get_cifar10_loaders(batch_size=128, num_workers=None, pin_memory=None):
    """
    获取 CIFAR-10 训练集和测试集的 DataLoader

    Args:
        batch_size: 批大小 (默认128)
        num_workers: 数据加载线程数 (默认使用全局配置)
        pin_memory: 是否锁页内存 (默认使用全局配置)

    Returns:
        trainloader: 训练集 DataLoader
        testloader: 测试集 DataLoader
        num_classes: 类别数 (10)
    """
    if num_workers is None:
        num_workers = NUM_WORKERS
    if pin_memory is None:
        pin_memory = PIN_MEMORY

    # 训练集数据增强
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),           # 随机裁剪 (先填充4像素)
        transforms.RandomHorizontalFlip(),               # 随机水平翻转
        transforms.ToTensor(),                           # 转为Tensor
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD), # 标准化
    ])

    # 测试集: 仅标准化，不做增强
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # 自动下载 CIFAR-10 数据集
    print(f"加载 CIFAR-10 数据集 (保存至 {DATA_DIR})...")
    trainset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform_test
    )

    # 创建 DataLoader
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    print(f"  训练集: {len(trainset)} 张图片")
    print(f"  测试集: {len(testset)} 张图片")
    print(f"  批大小: {batch_size}")
    print(f"  训练批次数: {len(trainloader)}")

    return trainloader, testloader, 10


def get_test_loader(batch_size=128):
    """仅获取测试集 DataLoader (用于评估)"""
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    testset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    return testloader
