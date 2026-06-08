"""
脚本4: 批量运行知识蒸馏实验

自动遍历所有温度T和alpha的组合, 记录结果到汇总CSV。

用法:
    python run_all_experiments.py                 # 默认50 epochs
    python run_all_experiments.py --epochs 20     # 快速模式
    python run_all_experiments.py --temperatures 2 4 8  # 只跑指定温度
"""
import argparse
import csv
import os
import subprocess
import sys

from config import DISTILLATION, TEACHER


def check_prerequisites():
    if not os.path.exists(TEACHER["checkpoint"]):
        print(f"ERROR: Teacher checkpoint not found: {TEACHER['checkpoint']}")
        print("Run: python train_teacher.py first")
        return False
    return True


def run_single_experiment(python_exe, T, alpha, epochs):
    cmd = [
        python_exe, "train_student_distill.py",
        "-T", str(T), "-a", str(alpha),
        "--epochs", str(epochs),
    ]
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Batch Distillation Experiments")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--temperatures", type=float, nargs="+",
                        default=DISTILLATION["temperatures"])
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=DISTILLATION["alphas"])
    args = parser.parse_args()

    print("=" * 60)
    print("  Batch Knowledge Distillation Experiments")
    print("=" * 60)

    if not check_prerequisites():
        return

    temperatures = args.temperatures
    alphas = args.alphas
    total = len(temperatures) * len(alphas)

    print(f"  Temperatures: {temperatures}")
    print(f"  Alphas: {alphas}")
    print(f"  Total experiments: {total}")
    print(f"  Epochs per experiment: {args.epochs}")

    python_exe = sys.executable
    summary_file = DISTILLATION["summary_file"]
    summary_fields = ["temperature", "alpha", "best_test_acc", "checkpoint", "log_file"]
    with open(summary_file, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=summary_fields).writeheader()

    results = []
    completed = 0

    for T in temperatures:
        for alpha in alphas:
            completed += 1
            print(f"\n>>> Experiment [{completed}/{total}]: T={T}, alpha={alpha}")

            success = run_single_experiment(python_exe, T, alpha, args.epochs)

            log_path = DISTILLATION["log_template"].format(T=T, alpha=alpha)
            ckpt_path = DISTILLATION["checkpoint_template"].format(T=T, alpha=alpha)

            best_acc = None
            if success and os.path.exists(log_path):
                import pandas as pd
                df = pd.read_csv(log_path)
                best_acc = df["test_acc"].max()

            result_row = {
                "temperature": T,
                "alpha": alpha,
                "best_test_acc": f"{best_acc:.2f}" if best_acc else "N/A",
                "checkpoint": ckpt_path,
                "log_file": log_path,
            }
            results.append(result_row)

            with open(summary_file, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=summary_fields).writerow(result_row)

            if best_acc:
                print(f"  OK T={T}, a={alpha}: acc={best_acc:.2f}%")
            else:
                print(f"  FAIL T={T}, a={alpha}")

    # Summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  {'T':>5} {'Alpha':>6} {'Accuracy':>10}")
    print("  " + "-" * 25)

    best_result = None
    best_acc_val = 0
    for r in results:
        print(f"  {r['temperature']:>5} {r['alpha']:>6} {r['best_test_acc']:>10}")
        if r["best_test_acc"] != "N/A":
            acc_val = float(r["best_test_acc"])
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                best_result = r

    if best_result:
        print(f"\n  Best: T={best_result['temperature']}, "
              f"a={best_result['alpha']}, acc={best_result['best_test_acc']}%")

    print(f"\n  Summary saved: {summary_file}")


if __name__ == "__main__":
    main()
