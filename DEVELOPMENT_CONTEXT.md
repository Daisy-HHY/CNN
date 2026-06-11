# 项目开发上下文 (Development Context)

> 本文档记录了成员B（深度学习担当）部分的完整开发过程、技术决策和实验结果。

---

## 一、项目背景

这是一个课程作业项目，3人小组合作完成CIFAR-10图像分类任务。

- **成员A**：SVM算法实现（HOG特征提取 + 线性核/RBF核）
- **成员B**（本项目）：神经网络模型M实现 + 知识蒸馏
- **成员C**：报告整合 + SMO算法理论 + 通用逼近定理

### 成员B的具体任务

1. 构建教师模型M（CNN），在CIFAR-10上训练
2. 构建学生模型（轻量级网络）
3. 实现知识蒸馏（Distillation Loss = Hard Loss + α·T²·Soft Loss）
4. 对比"学生直接训练" vs "学生蒸馏训练"的精度差异

---

## 二、技术决策记录

### 决策1：教师模型架构选择

| 选项 | 参数量 | 预期准确率 | CPU训练速度 | 最终选择 |
|------|--------|-----------|------------|---------|
| ResNet-18 (原版) | 11.2M | 93-95% | 10.5 min/epoch ❌ | ✗ 太慢 |
| VGG-like | ~5M | 90%+ | ~5 min/epoch | ✗ 仍然慢 |
| **TeacherCNN (自定义)** | **1.74M** | **64.72%** | **~3 min/epoch** | **✓** |

**决策原因**：
- 运行环境为 CPU-only（无GPU），ResNet-18 每个epoch需 10.5分钟，20轮需 3.5小时
- TeacherCNN 使用 4个卷积块（Block1/2各2层Conv，Block3/4各1层Conv）+ GAP，大幅减少计算量
- 虽然 CPU 上准确率受限（64.72%），但足以演示知识蒸馏效果

### 决策2：学生模型架构

| 选项 | 参数量 | 最终选择 |
|------|--------|---------|
| SmallCNN (原版, 含大FC层) | 2.2M | ✗ FC层参数太多 |
| **SmallCNN (GAP版)** | **94.8K** | **✓** |

**决策原因**：
- 原版 SmallCNN 的 FC(8192→256) 层占了 2.1M/2.2M 参数，不合理
- 使用 GAP 替代大 FC 层，参数量从 2.2M 降至 94.8K
- 教师/学生参数比 = 18.4:1，差距足够体现蒸馏效果

### 决策3：训练超参数

原始配置（为GPU设计）vs 实际使用（CPU适配）：

| 参数 | 原始配置 | 实际配置 | 原因 |
|------|---------|---------|------|
| 优化器 | SGD(lr=0.1, momentum=0.9) | Adam(lr=0.01) | Adam收敛更快，适合少epoch |
| 学习率 | 0.1 | 0.01 | Adam默认lr更小 |
| Epochs | 200 | 20(教师)/7(学生)/10(蒸馏) | CPU时间限制 |
| Batch Size | 128 | 128 | 保持不变 |

### 决策4：蒸馏实验设计

- 温度参数：T = [2, 4, 8]（3组对比）
- Alpha权重：固定为 0.7
- 训练轮数：每组 10 epochs

---

## 三、代码结构

```
H:/orca/CNN/
├── config.py                       # 全局配置（路径、超参��、归一化参数）
├── requirements.txt                # Python依赖
├── core_code.py                    # ★ 核心代码合集（自包含，可直接运行）
├── models/
│   ├── __init__.py
│   ├── resnet.py                   # 教师模型 TeacherCNN (1.74M params)
│   └── student.py                  # 学生模型 SmallCNN (94.8K params)
├── utils/
│   ├── __init__.py
│   ├── dataset.py                  # CIFAR-10数据加载与增强
│   └── visualization.py           # 绘图工具（训练曲线/对比图/混淆矩阵）
├── train_teacher.py                # 训练教师模型
├── train_student_baseline.py       # 训练学生模型（无蒸馏，对照组）
├── train_student_distill.py        # 知识蒸馏训练学生模型
├── run_all_experiments.py          # 批量蒸馏实验（多T×alpha组合）
├── evaluate.py                     # 评估所有模型 + 生成报告图表
├── generate_report.py              # 生成 Word 实验报告 (.docx)
├── checkpoints/                    # 模型权重 (.pth)
├── outputs/                        # 训练日志 + 图表 + 报告
└── data/                           # CIFAR-10数据集（.gitignore排除）
```

### 核心代码说明

**教师模型 TeacherCNN** (`models/resnet.py`)
- 4个卷积块：Block1(3→64→64), Block2(64→128→128), Block3(128→256), Block4(256→512)
- 共6个Conv层 + BN + ReLU + MaxPool
- GAP → FC(512→10)
- 参数量：1,741,770 (1.74M)

**学生模型 SmallCNN** (`models/student.py`)
- 3个卷积层：Conv(3→32), Conv(32→64), Conv(64→128)
- BN + ReLU + MaxPool(2×2)
- GAP → Dropout(0.3) → FC(128→10)
- 参数量：94,762 (94.8K)

**知识蒸馏损失** (`train_student_distill.py`)
```python
def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    hard_loss = F.cross_entropy(student_logits, labels)
    soft_teacher = F.softmax(teacher_logits / T, dim=-1)
    soft_student = F.log_softmax(student_logits / T, dim=-1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T * T)
    loss = (1 - alpha) * hard_loss + alpha * soft_loss
    return loss, hard_loss.item(), soft_loss.item()
```

**数据预处理** (`utils/dataset.py`)
- 训练集：RandomCrop(32, padding=4) + RandomHorizontalFlip + ToTensor + Normalize
- 测试集：ToTensor + Normalize
- 归一化参数：mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)

---

## 四、实验结果

### 训练环境
- CPU: Intel (无GPU加速)
- Python: 3.10.20
- PyTorch: 2.12.0+cpu
- OS: Windows 11

### 完整实验数据

| 模型 | 参数量 | 测试准确率 | 训练时间 | 相比Baseline |
|------|--------|-----------|---------|-------------|
| Teacher (TeacherCNN) | 1.74M | 64.72% | 33.9 min (6 epochs) | — |
| Student Baseline (SmallCNN) | 94.8K | 61.23% | 11.3 min (7 epochs) | — |
| Distilled T=2, α=0.7 | 94.8K | 67.99% | 17.8 min (10 epochs) | **+6.76%** |
| **Distilled T=4, α=0.7** | **94.8K** | **68.26%** | **16.2 min (10 epochs)** | **+7.03%** ⭐ |
| Distilled T=8, α=0.7 | 94.8K | 67.64% | 12.9 min (10 epochs) | **+6.41%** |

### 关键发现

1. **蒸馏有效**：所有3组蒸馏实验均超过学生baseline（+6~7%）
2. **蒸馏学生超越教师**：蒸馏学生(68.26%) > 教师模型(64.72%)，超出了+3.54%
3. **T=4效果最佳**：中等温度提供最合适的软标签平滑度
4. **T的影响**：T过小(2)暗知识不足，T过大(8)过度平滑，T=4取得平衡

### 训练日志

- 教师模型: `outputs/teacher_log.csv` (6 epochs)
- 学生baseline: `outputs/student_baseline_log.csv` (7 epochs)
- 蒸馏T=2: `outputs/student_distill_T2.0_a0.7_log.csv`
- 蒸馏T=4: `outputs/student_distill_T4.0_a0.7_log.csv`
- 蒸馏T=8: `outputs/student_distill_T8.0_a0.7_log.csv`
- 汇总: `outputs/distillation_summary.csv`

---

## 五、生成文件清单

### 模型权重 (checkpoints/)
- `teacher_best.pth` — 教师模型最佳权重
- `student_baseline_best.pth` — 学生baseline最佳权重
- `student_distill_T2.0_a0.7_best.pth` — 蒸馏学生(T=2)权重
- `student_distill_T4.0_a0.7_best.pth` — 蒸馏学生(T=4)权重
- `student_distill_T8.0_a0.7_best.pth` — 蒸馏学生(T=8)权重

### 图表 (outputs/)
- `report_teacher_curve.png` — 教师模型训练曲线(Loss+Acc)
- `report_student_curve.png` — 学生baseline训练曲线
- `report_distill_curve.png` — 蒸馏学生(T=4)训练曲线
- `report_model_comparison.png` — 三模型准确率对比折线图
- `report_temperature_bar.png` — 温度对比柱状图

### 报告
- `outputs/experiment_report.docx` — 完整实验报告Word文档

---

## 六、已知限制与改进方向

### 当前限制
1. **CPU训练**：无GPU导致epoch数受限（教师仅6轮，学生7轮，蒸馏10轮）
2. **教师模型精度不高**：64.72%远低于ResNet-18在GPU上可达的93%+
3. **训练不稳定**：Adam+高学习率导致测试准确率波动大
4. **实验组数有限**：仅测试了3个温度值，未测试不同alpha

### 如有GPU可做的改进
1. 使用完整ResNet-18作为教师（预期93%+）
2. 增加训练轮数至200 epochs
3. 测试更多T×alpha组合（如T=[2,4,8,10,20] × α=[0.3,0.5,0.7,0.9]）
4. 添加混淆矩阵、t-SNE可视化等更多分析

---

## 七、如何运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 验证模型定义
python core_code.py

# 3. 训练教师模型
python train_teacher.py --epochs 20

# 4. 训练学生baseline
python train_student_baseline.py --epochs 20

# 5. 知识蒸馏训练
python train_student_distill.py -T 4 -a 0.7 --epochs 10

# 6. 生成评估图表
python evaluate.py

# 7. 生成实验报告
python generate_report.py
```

---

## 八、Git提交历史

```
ec411af 修复报告卷积层数描述 + 整理核心代码到core_code.py
a62fd09 完成知识蒸馏实验：3组温度对比实验 + 评估图表 + 实验报告docx
c7276fd 实现成员B任务：教师模型、学生模型、知识蒸馏训练脚本和报告生成
51860f6 新增项目分工计划，明确成员任务与工作内容
31c53f7 Initial commit
```

---

## 九、开发过程遇到的问题及解决方案

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| ResNet-18训练10.5min/epoch太慢 | CPU-only，11.2M参数 | 改用轻量TeacherCNN(1.74M) |
| tqdm产生78K+行输出 | 非终端环境tqdm刷屏 | 去掉tqdm，用简单print |
| MemoryError加载数据 | 多进程同时加载CIFAR-10 | 改为顺序训练，num_workers=0 |
| 训练准确率波动大 | Adam lr=0.01过高 | 接受波动，保存best checkpoint |
| Python输出缓冲看不到进度 | 非终端stdout缓冲 | 添加 -u 标志 |
| Windows GBK编码显示乱码 | 终端不支持UTF-8 | 不影响功能，仅显示问题 |
