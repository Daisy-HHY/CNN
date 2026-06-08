# 成员B：神经网络模型实现与知识蒸馏 — 实现计划

## Context

这是一个课程作业项目，需要完成 **教师模型CNN训练 + 学生模型训练 + 知识蒸馏实验**。
项目目录目前为空（仅有 plan.md 和 README.md），所有代码从零创建。

**环境约束**: CPU-only（无GPU）

---

## 项目文件结构

```
H:/orca/CNN/
├── requirements.txt                # 依赖列表
├── config.py                       # 全局超参数配置
├── models/
│   ├── __init__.py
│   ├── resnet.py                   # 教师模型: ResNet-18 (CIFAR-10适配版)
│   └── student.py                  # 学生模型: 轻量CNN (SmallCNN)
├── utils/
│   ├── __init__.py
│   ├── dataset.py                  # CIFAR-10数据加载与预处理
│   └── visualization.py           # 绘图工具（训练曲线、混淆矩阵、对比图）
├── train_teacher.py                # 脚本1: 训练教师模型
├── train_student_baseline.py       # 脚本2: 训练学生模型（无蒸馏，对照组）
├── train_student_distill.py        # 脚本3: 训练学生模型（带蒸馏）
├── run_all_experiments.py          # 脚本4: 一键运行所有蒸馏实验
├── evaluate.py                     # 脚本5: 评估所有模型，生成报告图表
├── checkpoints/                    # 保存的模型权重
└── outputs/                        # 生成的图表和日志
```

---

## 模型架构

### 教师模型: ResNet-18 (CIFAR-10适配版)
- stem: Conv2d(3→64, 3×3, stride=1, padding=1) + BN + ReLU
- 4个stage: 各含2个BasicBlock，通道 64→128→256→512
- 分类头: AdaptiveAvgPool2d(1) + Linear(512→10)
- **参数量**: ~11.2M | **预期准确率**: 90-93%

### 学生模型: SmallCNN
- conv1(3→32) + BN + ReLU + MaxPool
- conv2(32→64) + BN + ReLU + MaxPool
- conv3(64→128) + BN + ReLU
- FC(8192→256) + Dropout(0.3) + FC(256→10)
- **参数量**: ~2.2M | **预期准确率**: 78-82%(直接) / 82-86%(蒸馏)

---

## 知识蒸馏核心公式

```
L = (1 - α) × CE(student_logits, labels) + α × T² × KL(softmax(s/T) || softmax(t/T))
```

- T (温度): [2, 4, 8, 10, 20]
- α (软标签权重): [0.3, 0.5, 0.7, 0.9]

---

## 训练超参数

| 参数 | 教师模型 | 学生模型 |
|------|---------|---------|
| 优化器 | SGD(lr=0.1, momentum=0.9) | SGD(lr=0.05, momentum=0.9) |
| 学习率策略 | CosineAnnealingLR | CosineAnnealingLR |
| 权重衰减 | 5e-4 | 5e-4 |
| Batch Size | 128 | 128 |
| Epochs | 200 | 200 |

---

## CPU训练时间预估

- 教师模型 ResNet-18 (200 epochs): ~5-8小时
- 学生模型 SmallCNN (200 epochs): ~2-4小时
- 单次蒸馏实验 (200 epochs): ~3-5小时

**建议**: 先跑2个epoch快速验证，再跑完整训练。

---

## 报告产出物 (outputs/)

1. 训练曲线图 (教师/学生baseline/蒸馏学生)
2. 三模型准确率对比折线图
3. 不同T值的蒸馏效果柱状图
4. 混淆矩阵热力图
5. 实验结果汇总表格 (CSV + 文本)
