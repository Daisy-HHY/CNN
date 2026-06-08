"""
全局配置文件 - 集中管理所有超参数和路径
"""
import torch
import os

# ======================== 设备配置 ========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0  # Windows兼容: 设为0避免多进程问题
PIN_MEMORY = False  # CPU模式不需要pin_memory

# ======================== 路径配置 ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# 自动创建目录
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================== CIFAR-10 数据配置 ========================
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)

# ======================== 教师模型训练配置 ========================
TEACHER = {
    "model": "resnet18",
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "batch_size": 128,
    "epochs": 200,
    "checkpoint": os.path.join(CHECKPOINT_DIR, "teacher_best.pth"),
    "log_file": os.path.join(OUTPUT_DIR, "teacher_log.csv"),
}

# ======================== 学生模型训练配置 ========================
STUDENT = {
    "model": "smallcnn",
    "lr": 0.05,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "batch_size": 128,
    "epochs": 200,
    "checkpoint": os.path.join(CHECKPOINT_DIR, "student_baseline_best.pth"),
    "log_file": os.path.join(OUTPUT_DIR, "student_baseline_log.csv"),
}

# ======================== 蒸馏配置 ========================
DISTILLATION = {
    "temperatures": [2, 4, 8, 10, 20],      # 温度参数候选值
    "alphas": [0.3, 0.5, 0.7, 0.9],         # 软标签权重候选值
    "lr": 0.05,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "batch_size": 128,
    "epochs": 200,
    "checkpoint_template": os.path.join(CHECKPOINT_DIR, "student_distill_T{T}_a{alpha}_best.pth"),
    "log_template": os.path.join(OUTPUT_DIR, "student_distill_T{T}_a{alpha}_log.csv"),
    "summary_file": os.path.join(OUTPUT_DIR, "distillation_summary.csv"),
}


def get_device_info():
    """返回当前设备信息字符串"""
    if torch.cuda.is_available():
        return f"GPU: {torch.cuda.get_device_name(0)}"
    return "CPU (无GPU加速)"
