"""
可视化工具模块

功能:
- 绘制训练曲线 (loss + accuracy)
- 绘制多模型准确率对比图
- 绘制温度对比柱状图
- 绘制混淆矩阵热力图
- 输出汇总表格
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无GUI后端，保存为文件
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from config import OUTPUT_DIR, CIFAR10_CLASSES

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150


def plot_training_curve(log_csv, save_path=None, title="Training Curve"):
    """
    绘制单个模型的训练曲线

    Args:
        log_csv: 训练日志CSV文件路径或DataFrame
        save_path: 图片保存路径 (默认自动生成)
        title: 图表标题
    """
    if isinstance(log_csv, str):
        df = pd.read_csv(log_csv)
    else:
        df = log_csv

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = df["epoch"]

    # 左图: Loss曲线
    ax1.plot(epochs, df["train_loss"], "b-", label="Train Loss", alpha=0.8)
    ax1.plot(epochs, df["test_loss"], "r-", label="Test Loss", alpha=0.8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} - Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 右图: Accuracy曲线
    ax2.plot(epochs, df["train_acc"], "b-", label="Train Acc", alpha=0.8)
    ax2.plot(epochs, df["test_acc"], "r-", label="Test Acc", alpha=0.8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"{title} - Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, f"{title.lower().replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"训练曲线已保存: {save_path}")


def plot_comparison(log_paths, labels, save_path=None):
    """
    绘制多模型准确率对比图

    Args:
        log_paths: 各模型训练日志CSV路径列表
        labels: 各模型的标签名列表
        save_path: 图片保存路径
    """
    plt.figure(figsize=(12, 6))

    for log_path, label in zip(log_paths, labels):
        if log_path is None or not os.path.exists(log_path):
            continue
        df = pd.read_csv(log_path)
        plt.plot(df["epoch"], df["test_acc"], label=label, alpha=0.8, linewidth=1.5)

    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Model Comparison - Test Accuracy")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"对比图已保存: {save_path}")


def plot_temperature_bar(results, baseline_acc, teacher_acc, save_path=None):
    """
    绘制不同温度T下的蒸馏准确率柱状图

    Args:
        results: dict, {T值: accuracy}
        baseline_acc: 学生模型baseline准确率
        teacher_acc: 教师模型准确率
        save_path: 图片保存路径
    """
    temperatures = sorted(results.keys())
    accuracies = [results[t] for t in temperatures]

    plt.figure(figsize=(10, 6))
    bars = plt.bar([f"T={t}" for t in temperatures], accuracies,
                   color="#4C72B0", alpha=0.8, edgecolor="black", linewidth=0.5)

    # 添加基准线
    plt.axhline(y=baseline_acc, color="red", linestyle="--", linewidth=1.5,
                label=f"Student Baseline: {baseline_acc:.2f}%")
    plt.axhline(y=teacher_acc, color="green", linestyle="--", linewidth=1.5,
                label=f"Teacher: {teacher_acc:.2f}%")

    # 在柱子上方标注数值
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
                 f"{acc:.2f}%", ha="center", va="bottom", fontsize=10)

    plt.xlabel("Temperature")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Knowledge Distillation - Temperature Comparison")
    plt.legend(fontsize=10)
    plt.ylim(min(baseline_acc - 5, min(accuracies) - 2),
             max(teacher_acc + 2, max(accuracies) + 2))
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "temperature_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"温度对比图已保存: {save_path}")


def plot_confusion_matrix(model, dataloader, device, class_names=None,
                          save_path=None, title="Confusion Matrix"):
    """
    绘制混淆矩阵热力图

    Args:
        model: PyTorch模型
        dataloader: 测试集DataLoader
        device: 计算设备
        class_names: 类别名称列表
        save_path: 图片保存路径
        title: 图表标题
    """
    if class_names is None:
        class_names = list(CIFAR10_CLASSES)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=1)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, f"cm_{title.lower().replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"混淆矩阵已保存: {save_path}")

    # 返回准确率
    accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels) * 100
    return accuracy


def print_summary_table(results_dict, save_path=None):
    """
    输出汇总表格

    Args:
        results_dict: dict, {模型名: {"accuracy": float, "params": int}}
        save_path: 保存路径
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "summary_table.txt")

    lines = []
    lines.append("=" * 80)
    lines.append("实验结果汇总表")
    lines.append("=" * 80)
    lines.append(f"{'Model':<35} {'Params':>10} {'Accuracy':>10} {'Delta':>10}")
    lines.append("-" * 80)

    baseline_acc = None
    for name, info in results_dict.items():
        acc = info["accuracy"]
        params = info["params"]
        params_str = f"{params / 1e6:.2f}M"

        if "Baseline" in name:
            baseline_acc = acc
            delta_str = "-"
        elif baseline_acc is not None:
            delta = acc - baseline_acc
            delta_str = f"+{delta:.2f}%" if delta >= 0 else f"{delta:.2f}%"
        else:
            delta_str = "-"

        lines.append(f"{name:<35} {params_str:>10} {acc:>9.2f}% {delta_str:>10}")

    lines.append("=" * 80)

    table_text = "\n".join(lines)
    print(table_text)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(table_text)
    print(f"\n汇总表已保存: {save_path}")
