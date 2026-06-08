"""
脚本3: 知识蒸馏训练学生模型

蒸馏损失: L = (1-a)*CE + a*T^2*KL(softmax(s/T) || softmax(t/T))
用法: python train_student_distill.py -T 4 -a 0.7 --epochs 30
"""
import argparse
import csv
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import DEVICE, TEACHER, DISTILLATION, get_device_info
from models.resnet import TeacherCNN
from models.student import SmallCNN, count_parameters
from utils.dataset import get_cifar10_loaders


def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    hard_loss = F.cross_entropy(student_logits, labels)
    soft_teacher = F.softmax(teacher_logits / T, dim=-1)
    soft_student = F.log_softmax(student_logits / T, dim=-1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T * T)
    loss = (1 - alpha) * hard_loss + alpha * soft_loss
    return loss, hard_loss.item(), soft_loss.item()


def evaluate(model, testloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
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


def train_one_epoch(student, teacher, trainloader, optimizer, device, T, alpha):
    student.train()
    teacher.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(trainloader)
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            teacher_logits = teacher(inputs)
        student_logits = student(inputs)
        loss, _, _ = distillation_loss(student_logits, teacher_logits, labels, T, alpha)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(student_logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if (batch_idx + 1) % 100 == 0:
            print(f"    batch [{batch_idx+1}/{num_batches}] loss={loss.item():.4f}")
    return running_loss / total, 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description="Knowledge Distillation Training")
    parser.add_argument("--temperature", "-T", type=float, default=4.0)
    parser.add_argument("--alpha", "-a", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--teacher_checkpoint", type=str, default=TEACHER["checkpoint"])
    args = parser.parse_args()

    T = args.temperature
    alpha = args.alpha
    ckpt_path = DISTILLATION["checkpoint_template"].format(T=T, alpha=alpha)
    log_file = DISTILLATION["log_template"].format(T=T, alpha=alpha)

    print("=" * 60)
    print(f"  Distillation: T={T}, alpha={alpha}")
    print("=" * 60)
    print(f"  Device: {get_device_info()}")
    print(f"  Loss = (1-{alpha})*Hard + {alpha}*{T}^2*Soft")
    print(f"  Epochs: {args.epochs}")

    # Load teacher
    teacher = TeacherCNN(num_classes=10).to(DEVICE)
    assert os.path.exists(args.teacher_checkpoint), f"Teacher not found: {args.teacher_checkpoint}"
    teacher.load_state_dict(torch.load(args.teacher_checkpoint, map_location=DEVICE, weights_only=True))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"  Teacher loaded")

    student = SmallCNN(num_classes=10).to(DEVICE)
    print(f"  Student params: {count_parameters(student):,} ({count_parameters(student)/1e6:.2f}M)")
    print()

    trainloader, testloader, _ = get_cifar10_loaders(batch_size=args.batch_size)

    optimizer = optim.Adam(student.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    log_fields = ["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "lr", "time"]
    with open(log_file, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    best_acc = 0.0
    total_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(student, teacher, trainloader, optimizer, DEVICE, T, alpha)
        test_loss, test_acc = evaluate(student, testloader, DEVICE)
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
            torch.save(student.state_dict(), ckpt_path)

        print(f"  Epoch [{epoch:3d}/{args.epochs}] "
              f"Train:{train_acc:.1f}% Test:{test_acc:.1f}% "
              f"Best:{best_acc:.1f}% T={T} a={alpha} [{elapsed:.0f}s]")

    total_time = time.time() - total_start
    print(f"\n  Done! T={T} a={alpha} Best={best_acc:.2f}% Time={total_time/60:.1f}min")
    print(f"  Saved: {ckpt_path}")
    print(f"  Log: {log_file}")


if __name__ == "__main__":
    main()
