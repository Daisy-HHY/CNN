"""
脚本2: 训练学生模型 (无蒸馏, 对照组)

CPU优化版, ~220K参数, ~1min/epoch
用法: python train_student_baseline.py --epochs 30
"""
import argparse
import csv
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from config import DEVICE, STUDENT, get_device_info
from models.student import SmallCNN, count_parameters
from utils.dataset import get_cifar10_loaders


def evaluate(model, testloader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return test_loss / total, 100.0 * correct / total


def train_one_epoch(model, trainloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(trainloader)
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if (batch_idx + 1) % 100 == 0:
            print(f"    batch [{batch_idx+1}/{num_batches}] loss={loss.item():.4f}")
    return running_loss / total, 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description="Train Student Baseline SmallCNN")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    print("=" * 60)
    print("  Training Student Baseline: SmallCNN (CIFAR-10)")
    print("=" * 60)
    print(f"  Device: {get_device_info()}")
    print(f"  Epochs: {args.epochs}  LR: {args.lr}  Batch: {args.batch_size}")

    trainloader, testloader, _ = get_cifar10_loaders(batch_size=args.batch_size)

    model = SmallCNN(num_classes=10).to(DEVICE)
    print(f"  Parameters: {count_parameters(model):,} ({count_parameters(model)/1e6:.2f}M)")
    print()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    log_file = STUDENT["log_file"]
    log_fields = ["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "lr", "time"]
    with open(log_file, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    best_acc = 0.0
    total_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, DEVICE)
        test_loss, test_acc = evaluate(model, testloader, criterion, DEVICE)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        elapsed = time.time() - t0

        with open(log_file, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=log_fields)
            w.writerow({"epoch": epoch, "train_loss": f"{train_loss:.4f}",
                        "train_acc": f"{train_acc:.2f}", "test_loss": f"{test_loss:.4f}",
                        "test_acc": f"{test_acc:.2f}", "lr": f"{lr:.6f}", "time": f"{elapsed:.1f}"})

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), STUDENT["checkpoint"])

        print(f"  Epoch [{epoch:3d}/{args.epochs}] "
              f"Train:{train_acc:.1f}% Test:{test_acc:.1f}% "
              f"Best:{best_acc:.1f}% LR:{lr:.5f} [{elapsed:.0f}s]")

    total_time = time.time() - total_start
    print(f"\n  Done! Best={best_acc:.2f}% Time={total_time/60:.1f}min")
    print(f"  Saved: {STUDENT['checkpoint']}")
    print(f"  Log: {log_file}")


if __name__ == "__main__":
    main()
