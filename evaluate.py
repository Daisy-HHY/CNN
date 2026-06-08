"""
脚本5: 评估所有模型并生成报告图表

读取所有训练好的模型和日志，生成:
1. 各模型训练曲线图
2. 多模型准确率对比折线图
3. 温度对比柱状图
4. 混淆矩阵热力图
5. 汇总表格

用法:
    python evaluate.py                  # 生成所有图表
    python evaluate.py --skip_cm       # 跳过混淆矩阵 (较快)
"""
import argparse
import os

import pandas as pd
import torch

from config import (
    DEVICE, TEACHER, STUDENT, DISTILLATION,
    OUTPUT_DIR, get_device_info
)
from models.resnet import TeacherCNN, count_parameters as count_teacher_params
from models.student import SmallCNN, count_parameters
from utils.dataset import get_cifar10_loaders
from utils.visualization import (
    plot_training_curve,
    plot_comparison,
    plot_temperature_bar,
    plot_confusion_matrix,
    print_summary_table,
)


def load_model(model, checkpoint_path, device):
    """加载模型权重"""
    if not os.path.exists(checkpoint_path):
        print(f"  警告: 找不到权重文件 {checkpoint_path}")
        return None
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def compute_accuracy(model, dataloader, device):
    """计算模型在测试集上的准确率"""
    if model is None:
        return None
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description="评估所有模型并生成报告图表")
    parser.add_argument("--skip_cm", action="store_true",
                        help="跳过混淆矩阵生成 (较快)")
    args = parser.parse_args()

    print("=" * 60)
    print("  评估模型 & 生成报告图表")
    print("=" * 60)
    print(f"  设备: {get_device_info()}")
    print()

    # 加载测试数据
    _, testloader, _ = get_cifar10_loaders(batch_size=100)

    # =====================================================
    # 1. 评估教师模型
    # =====================================================
    print("[1/6] 评估教师模型...")
    teacher = TeacherCNN(num_classes=10)
    teacher = load_model(teacher, TEACHER["checkpoint"], DEVICE)
    teacher_acc = compute_accuracy(teacher, testloader, DEVICE) if teacher else None
    teacher_params = count_teacher_params(teacher)
    if teacher_acc:
        print(f"  教师模型 (ResNet-18): {teacher_acc:.2f}% ({teacher_params/1e6:.2f}M参数)")

    # 生成教师训练曲线
    if os.path.exists(TEACHER["log_file"]):
        plot_training_curve(
            TEACHER["log_file"],
            save_path=os.path.join(OUTPUT_DIR, "teacher_training_curve.png"),
            title="Teacher (TeacherCNN)"
        )

    # =====================================================
    # 2. 评估学生Baseline
    # =====================================================
    print("\n[2/6] 评估学生模型 (Baseline)...")
    student_baseline = SmallCNN(num_classes=10)
    student_baseline = load_model(student_baseline, STUDENT["checkpoint"], DEVICE)
    baseline_acc = compute_accuracy(student_baseline, testloader, DEVICE) if student_baseline else None
    student_params = count_parameters(student_baseline)
    if baseline_acc:
        print(f"  学生Baseline (SmallCNN): {baseline_acc:.2f}% ({student_params/1e6:.2f}M参数)")

    # 生成学生baseline训练曲线
    if os.path.exists(STUDENT["log_file"]):
        plot_training_curve(
            STUDENT["log_file"],
            save_path=os.path.join(OUTPUT_DIR, "student_baseline_curve.png"),
            title="Student Baseline (SmallCNN)"
        )

    # =====================================================
    # 3. 收集蒸馏实验结果
    # =====================================================
    print("\n[3/6] 收集蒸馏实验结果...")
    distill_results = {}  # {(T, alpha): acc}
    best_distill_log = None
    best_distill_acc = 0
    best_distill_T = None
    best_distill_alpha = None

    summary_file = DISTILLATION["summary_file"]
    if os.path.exists(summary_file):
        df = pd.read_csv(summary_file)
        for _, row in df.iterrows():
            T = row["temperature"]
            alpha = row["alpha"]
            acc_str = row["best_test_acc"]
            if acc_str != "N/A":
                acc = float(acc_str)
                distill_results[(T, alpha)] = acc
                if acc > best_distill_acc:
                    best_distill_acc = acc
                    best_distill_T = T
                    best_distill_alpha = alpha
                    best_distill_log = DISTILLATION["log_template"].format(T=T, alpha=alpha)

        print(f"  共找到 {len(distill_results)} 组蒸馏实验结果")
        if best_distill_T:
            print(f"  最佳蒸馏配置: T={best_distill_T}, α={best_distill_alpha}, "
                  f"准确率={best_distill_acc:.2f}%")
    else:
        print("  未找到蒸馏实验汇总文件，跳过蒸馏分析")

    # =====================================================
    # 4. 生成对比图
    # =====================================================
    print("\n[4/6] 生成对比图...")

    # 多模型对比折线图
    log_paths = [
        TEACHER["log_file"],
        STUDENT["log_file"],
        best_distill_log,
    ]
    labels = [
        "Teacher (TeacherCNN)",
        "Student Baseline",
        f"Student Distilled (T={best_distill_T}, α={best_distill_alpha})" if best_distill_T else None,
    ]
    # 过滤掉不存在的
    valid = [(p, l) for p, l in zip(log_paths, labels) if p and os.path.exists(p)]
    if valid:
        plot_comparison(
            [p for p, _ in valid],
            [l for _, l in valid],
            save_path=os.path.join(OUTPUT_DIR, "model_comparison.png")
        )

    # 蒸馏学生训练曲线
    if best_distill_log and os.path.exists(best_distill_log):
        plot_training_curve(
            best_distill_log,
            save_path=os.path.join(OUTPUT_DIR, "student_distilled_curve.png"),
            title=f"Student Distilled (T={best_distill_T}, α={best_distill_alpha})"
        )

    # =====================================================
    # 5. 温度对比柱状图
    # =====================================================
    print("\n[5/6] 生成温度对比图...")

    # 按温度汇总: 取每个T下的最佳alpha结果
    temp_results = {}
    for (T, alpha), acc in distill_results.items():
        if T not in temp_results or acc > temp_results[T]:
            temp_results[T] = acc

    if temp_results and teacher_acc and baseline_acc:
        plot_temperature_bar(
            temp_results, baseline_acc, teacher_acc,
            save_path=os.path.join(OUTPUT_DIR, "temperature_comparison.png")
        )

    # =====================================================
    # 6. 混淆矩阵 & 汇总表
    # =====================================================
    if not args.skip_cm:
        print("\n[6/6] 生成混淆矩阵...")

        if teacher:
            plot_confusion_matrix(
                teacher, testloader, DEVICE,
                save_path=os.path.join(OUTPUT_DIR, "confusion_matrix_teacher.png"),
                title="Teacher (TeacherCNN)"
            )

        if student_baseline:
            plot_confusion_matrix(
                student_baseline, testloader, DEVICE,
                save_path=os.path.join(OUTPUT_DIR, "confusion_matrix_student_baseline.png"),
                title="Student Baseline"
            )

        # 加载最佳蒸馏学生模型
        if best_distill_T and best_distill_alpha:
            ckpt = DISTILLATION["checkpoint_template"].format(
                T=best_distill_T, alpha=best_distill_alpha
            )
            student_distill = SmallCNN(num_classes=10)
            student_distill = load_model(student_distill, ckpt, DEVICE)
            if student_distill:
                plot_confusion_matrix(
                    student_distill, testloader, DEVICE,
                    save_path=os.path.join(OUTPUT_DIR, "confusion_matrix_student_distilled.png"),
                    title=f"Student Distilled (T={best_distill_T})"
                )
    else:
        print("\n[6/6] 跳过混淆矩阵 (--skip_cm)")

    # 汇总表
    print("\n" + "=" * 60)
    results_dict = {}

    if teacher_acc:
        results_dict["Teacher (TeacherCNN)"] = {
            "accuracy": teacher_acc, "params": teacher_params
        }
    if baseline_acc:
        results_dict["Student Baseline (SmallCNN)"] = {
            "accuracy": baseline_acc, "params": student_params
        }

    for (T, alpha), acc in sorted(distill_results.items()):
        name = f"Student Distilled (T={T}, α={alpha})"
        results_dict[name] = {"accuracy": acc, "params": student_params}

    if results_dict:
        print_summary_table(
            results_dict,
            save_path=os.path.join(OUTPUT_DIR, "summary_table.txt")
        )

    print("\n所有图表已生成至:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
