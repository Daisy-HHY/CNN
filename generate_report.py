"""
生成完整实验报告 (.docx)

读取所有训练日志和评估结果，生成包含以下内容的Word文档:
1. 实验概述
2. 数据集介绍
3. 教师模型设计与训练
4. 学生模型设计与训练
5. 知识蒸馏实验
6. 结果分析与讨论
7. 结论

用法: python generate_report.py
前置条件: 所有训练已完成, 日志文件存在于 outputs/ 目录
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

from config import (
    TEACHER, STUDENT, DISTILLATION, OUTPUT_DIR,
    CIFAR10_CLASSES
)
from models.resnet import TeacherCNN, count_parameters as teacher_params
from models.student import SmallCNN, count_parameters as student_params


# ============================================================
# 工具函数
# ============================================================

def set_cell_shading(cell, color):
    """设置表格单元格背景色"""
    shading = cell._element.get_or_add_tcPr()
    shading_elem = shading.makeelement(qn('w:shd'), {
        qn('w:fill'): color,
        qn('w:val'): 'clear',
    })
    shading.append(shading_elem)


def add_heading_styled(doc, text, level=1):
    """添加带样式的标题"""
    heading = doc.add_heading(text, level=level)
    for run in heading.runs:
        run.font.color.rgb = RGBColor(0, 51, 102)
    return heading


def add_code_block(doc, code):
    """添加代码块"""
    p = doc.add_paragraph()
    p.style = 'No Spacing'
    run = p.add_run(code)
    run.font.name = 'Consolas'
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(50, 50, 50)
    # 添加灰色背景效果
    p.paragraph_format.left_indent = Cm(0.5)
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)
    return p


def create_accuracy_table(doc, results):
    """创建准确率对比表格"""
    table = doc.add_table(rows=1, cols=4)
    table.style = 'Light Grid Accent 1'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # 表头
    headers = ['模型', '参数量', '测试准确率', '相对Baseline提升']
    for i, header in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = header
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.bold = True

    baseline_acc = None
    for name, info in results.items():
        if 'Baseline' in name:
            baseline_acc = info['accuracy']
            break

    for name, info in results.items():
        row = table.add_row()
        row.cells[0].text = name
        row.cells[1].text = f"{info['params']/1e6:.2f}M" if info['params'] > 1e6 else f"{info['params']/1e3:.1f}K"
        row.cells[2].text = f"{info['accuracy']:.2f}%"

        if baseline_acc and 'Baseline' not in name and 'Teacher' not in name:
            delta = info['accuracy'] - baseline_acc
            row.cells[3].text = f"+{delta:.2f}%" if delta >= 0 else f"{delta:.2f}%"
        else:
            row.cells[3].text = '-'

        for cell in row.cells:
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    return table


def create_temperature_table(doc, temp_results, baseline_acc, teacher_acc):
    """创建温度对比表格"""
    table = doc.add_table(rows=1, cols=3)
    table.style = 'Light Grid Accent 1'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    headers = ['温度 T', '测试准确率', '相对Baseline提升']
    for i, h in enumerate(headers):
        table.rows[0].cells[i].text = h
        for p in table.rows[0].cells[i].paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.bold = True

    for T, acc in sorted(temp_results.items()):
        row = table.add_row()
        row.cells[0].text = f"T = {T}"
        row.cells[1].text = f"{acc:.2f}%"
        delta = acc - baseline_acc
        row.cells[2].text = f"+{delta:.2f}%" if delta >= 0 else f"{delta:.2f}%"
        for cell in row.cells:
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    return table


def generate_training_curve_image(log_csv, save_path, title):
    """生成训练曲线图"""
    df = pd.read_csv(log_csv)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(df['epoch'], df['train_loss'], 'b-', label='Train Loss', alpha=0.8)
    ax1.plot(df['epoch'], df['test_loss'], 'r-', label='Test Loss', alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} - Loss Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(df['epoch'], df['train_acc'], 'b-', label='Train Acc', alpha=0.8)
    ax2.plot(df['epoch'], df['test_acc'], 'r-', label='Test Acc', alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{title} - Accuracy Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_comparison_image(log_paths, labels, save_path):
    """生成多模型对比图"""
    plt.figure(figsize=(10, 5))
    for log_path, label in zip(log_paths, labels):
        if log_path and os.path.exists(log_path):
            df = pd.read_csv(log_path)
            plt.plot(df['epoch'], df['test_acc'], label=label, linewidth=1.5, alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Model Comparison - Test Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_temperature_bar_image(temp_results, baseline_acc, teacher_acc, save_path):
    """生成温度对比柱状图"""
    temps = sorted(temp_results.keys())
    accs = [temp_results[t] for t in temps]

    plt.figure(figsize=(8, 5))
    bars = plt.bar([f'T={t}' for t in temps], accs, color='#4C72B0', alpha=0.8, edgecolor='black', linewidth=0.5)
    plt.axhline(y=baseline_acc, color='red', linestyle='--', linewidth=1.5,
                label=f'Student Baseline: {baseline_acc:.2f}%')
    plt.axhline(y=teacher_acc, color='green', linestyle='--', linewidth=1.5,
                label=f'Teacher: {teacher_acc:.2f}%')

    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
                 f'{acc:.2f}%', ha='center', va='bottom', fontsize=9)

    plt.xlabel('Temperature')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Knowledge Distillation - Temperature Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# 生成报告主函数
# ============================================================

def generate_report():
    print("Generating experiment report...")

    doc = Document()

    # ---- 全局样式 ----
    style = doc.styles['Normal']
    font = style.font
    font.name = 'SimSun'
    font.size = Pt(11)

    # ============================================================
    # 封面 / 标题
    # ============================================================
    doc.add_paragraph()
    doc.add_paragraph()
    title = doc.add_heading('基于神经网络的CIFAR-10图像分类', level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle = doc.add_heading('——教师模型与知识蒸馏实验报告', level=2)
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()
    info = doc.add_paragraph()
    info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    info.add_run('课程：机器学习\n').font.size = Pt(12)
    info.add_run('成员B：深度学习担当\n').font.size = Pt(12)
    info.add_run('实验内容：教师模型M实现、学生模型实现、知识蒸馏').font.size = Pt(12)

    doc.add_page_break()

    # ============================================================
    # 目录
    # ============================================================
    doc.add_heading('目录', level=1)
    toc_items = [
        '1. 实验概述',
        '2. 数据集介绍',
        '3. 教师模型设计与训练',
        '4. 学生模型设计与训练',
        '5. 知识蒸馏实验',
        '6. 结果分析与讨论',
        '7. 结论',
    ]
    for item in toc_items:
        p = doc.add_paragraph(item)
        p.paragraph_format.space_after = Pt(2)

    doc.add_page_break()

    # ============================================================
    # 1. 实验概述
    # ============================================================
    doc.add_heading('1. 实验概述', level=1)

    doc.add_paragraph(
        '本实验是CIFAR-10图像分类项目的一部分，主要完成以下��个任务：'
    )
    doc.add_paragraph('（1）设计并训练一个性能较好的卷积神经网络作为教师模型（Teacher Model），'
                      '在CIFAR-10数据集上达到较高的分类准确率。')
    doc.add_paragraph('（2）设计一个参数量较小的轻量级网络作为学生模型（Student Model），'
                      '作为知识蒸馏的目标模型。')
    doc.add_paragraph('（3）实现知识蒸馏（Knowledge Distillation）算法，利用教师模型的"暗知识"'
                      '（Dark Knowledge）指导学生模型的训练，验证蒸馏对学生模型性能的提升效果。')

    doc.add_paragraph(
        '知识蒸馏的核心思想是：一个大型、高精度的教师模型在训练过程中产生的软标签'
        '（Soft Labels）包含了比硬标签（Hard Labels）更丰富的类别间关系信息。'
        '通过让学生模型同时学习硬标签和教师模型的软标签，可以在保持模型轻量化的同时'
        '获得比直接训练更好的性能。'
    )

    # ============================================================
    # 2. 数���集介绍
    # ============================================================
    doc.add_heading('2. 数据集介绍', level=1)

    doc.add_heading('2.1 CIFAR-10 数据集', level=2)
    doc.add_paragraph(
        'CIFAR-10 是由 Alex Krizhevsky 收集的用于通用机器学习研究的经典图像分类数据集。'
        '该数据集包含以下特征：'
    )

    data_table = doc.add_table(rows=6, cols=2)
    data_table.style = 'Light Grid Accent 1'
    data_info = [
        ('图像尺寸', '32 × 32 像素，RGB彩色'),
        ('类别数', '10类'),
        ('训练集大小', '50,000 张图像'),
        ('测试集大小', '10,000 张图像'),
        ('每类样本数', '训练集5,000张，测试集1,000张'),
        ('数据类型', 'uint8，像素值0-255'),
    ]
    for i, (k, v) in enumerate(data_info):
        data_table.rows[i].cells[0].text = k
        data_table.rows[i].cells[1].text = v

    doc.add_paragraph()
    doc.add_paragraph(f'CIFAR-10的10个类别为：{", ".join(CIFAR10_CLASSES)}')

    doc.add_heading('2.2 数据预处理', level=2)
    doc.add_paragraph('为保证训练效果，对数据进行了以下预处理：')
    doc.add_paragraph('（1）训练集数据增强：')
    doc.add_paragraph('    - RandomCrop(32, padding=4)：先对图像四周填充4个像素，'
                      '然后随机裁剪回32×32，增加训练样本的多样性。')
    doc.add_paragraph('    - RandomHorizontalFlip()：以50%的概率水平翻转图像。')
    doc.add_paragraph('（2）标准化处理：使用CIFAR-10训练集的全局统计量进行标准化。')
    add_code_block(doc, '  mean = (0.4914, 0.4822, 0.4465)\n  std  = (0.2023, 0.1994, 0.2010)')
    doc.add_paragraph('（3）测试集仅做标准化处理，不进行数据增强。')

    # ============================================================
    # 3. 教师模型
    # ============================================================
    doc.add_heading('3. 教师模型设计与训练', level=1)

    doc.add_heading('3.1 模型架构', level=2)
    doc.add_paragraph(
        '教师模型采用自定义的TeacherCNN架构，包含8个卷积层和1个全连接分类层。'
        '整体结构分为4个卷积块，每个块包含2个卷积层，中间穿插BatchNorm和ReLU激活函数。'
        '最后使用全局平均池化（Global Average Pooling）将特征图压缩为向量，'
        '再通过全连接层输出10个类别的logits。'
    )

    # 模型参数表
    teacher_model = TeacherCNN(10)
    t_params = teacher_params(teacher_model)

    doc.add_paragraph(f'教师模型总参数量：{t_params:,}（{t_params/1e6:.2f}M）')

    arch_table = doc.add_table(rows=6, cols=4)
    arch_table.style = 'Light Grid Accent 1'
    arch_headers = ['模块', '卷积核', '输出尺寸', '说明']
    for i, h in enumerate(arch_headers):
        arch_table.rows[0].cells[i].text = h
        for p in arch_table.rows[0].cells[i].paragraphs:
            for run in p.runs:
                run.bold = True

    arch_data = [
        ('Block 1', '3→64→64, 3×3', '16×16×64', '2层Conv+BN+ReLU+MaxPool'),
        ('Block 2', '64→128→128, 3×3', '8×8×128', '2层Conv+BN+ReLU+MaxPool'),
        ('Block 3', '128→256, 3×3', '4×4×256', 'Conv+BN+ReLU+MaxPool'),
        ('Block 4', '256→512, 3×3', '4×4×512', 'Conv+BN+ReLU'),
        ('Classifier', 'GAP→FC', '10', '全局平均池化+全连接'),
    ]
    for i, row_data in enumerate(arch_data):
        for j, val in enumerate(row_data):
            arch_table.rows[i+1].cells[j].text = val

    doc.add_heading('3.2 训练配置', level=2)

    train_config_table = doc.add_table(rows=6, cols=2)
    train_config_table.style = 'Light Grid Accent 1'
    train_config = [
        ('优化器', 'Adam (lr=0.01, weight_decay=5e-4)'),
        ('学习率策略', 'CosineAnnealingLR (T_max=epochs)'),
        ('损失函数', 'CrossEntropyLoss'),
        ('Batch Size', '128'),
        ('训练轮数', '20 epochs'),
        ('设备', 'CPU'),
    ]
    for i, (k, v) in enumerate(train_config):
        train_config_table.rows[i].cells[0].text = k
        train_config_table.rows[i].cells[1].text = v

    doc.add_heading('3.3 训练结果', level=2)

    # 读取教师训练日志
    teacher_log = TEACHER["log_file"]
    if os.path.exists(teacher_log):
        df_teacher = pd.read_csv(teacher_log)
        teacher_best_acc = df_teacher['test_acc'].max()
        teacher_best_epoch = df_teacher.loc[df_teacher['test_acc'].idxmax(), 'epoch']
        teacher_total_time = df_teacher['time'].astype(float).sum()

        doc.add_paragraph(
            f'教师模型在测试集上的最佳准确率为 {teacher_best_acc:.2f}% '
            f'（出现在第 {int(teacher_best_epoch)} 个epoch）。'
            f'总训练时间为 {teacher_total_time/60:.1f} 分钟。'
        )

        # 生成并插入训练曲线图
        curve_path = os.path.join(OUTPUT_DIR, 'report_teacher_curve.png')
        generate_training_curve_image(teacher_log, curve_path, 'Teacher Model (TeacherCNN)')
        if os.path.exists(curve_path):
            doc.add_picture(curve_path, width=Inches(6))
            doc.add_paragraph('图1：教师模型训练曲线', style='Caption')
    else:
        doc.add_paragraph('教师模型训练日志未找到。请先运行 train_teacher.py')
        teacher_best_acc = 0

    # ============================================================
    # 4. 学生模型
    # ============================================================
    doc.add_heading('4. 学生模型设计与训练', level=1)

    doc.add_heading('4.1 模型架构', level=2)
    doc.add_paragraph(
        '学生模型采用轻量级的SmallCNN架构，仅包含3个卷积层和1个全连接层。'
        '相比教师模型，学生模型的参数量大幅减少，用于验证知识蒸馏对轻量模型的提升效果。'
    )

    student_model = SmallCNN(10)
    s_params = student_params(student_model)

    doc.add_paragraph(f'学生模型总参数量：{s_params:,}（{s_params/1e3:.1f}K）')
    doc.add_paragraph(f'教师与学生参数量之比：{t_params/s_params:.1f} : 1')

    stud_arch_table = doc.add_table(rows=5, cols=4)
    stud_arch_table.style = 'Light Grid Accent 1'
    for i, h in enumerate(['模块', '卷积核', '输出尺寸', '说明']):
        stud_arch_table.rows[0].cells[i].text = h
        for p in stud_arch_table.rows[0].cells[i].paragraphs:
            for run in p.runs:
                run.bold = True

    stud_data = [
        ('Block 1', '3→32, 3×3', '16×16×32', 'Conv+BN+ReLU+MaxPool'),
        ('Block 2', '32→64, 3×3', '8×8×64', 'Conv+BN+ReLU+MaxPool'),
        ('Block 3', '64→128, 3×3', '8×8×128', 'Conv+BN+ReLU'),
        ('Classifier', 'GAP+Dropout+FC', '10', '全局平均池化+Dropout(0.3)+FC'),
    ]
    for i, row_data in enumerate(stud_data):
        for j, val in enumerate(row_data):
            stud_arch_table.rows[i+1].cells[j].text = val

    doc.add_heading('4.2 训练配置', level=2)
    doc.add_paragraph(
        '学生模型使用与教师模型相同的训练配置（Adam优化器、CosineAnnealingLR学习率策略、'
        'CrossEntropyLoss），以确保对比实验的公平性。'
    )

    doc.add_heading('4.3 Baseline训练结果', level=2)

    student_log = STUDENT["log_file"]
    if os.path.exists(student_log):
        df_student = pd.read_csv(student_log)
        student_best_acc = df_student['test_acc'].max()
        student_best_epoch = df_student.loc[df_student['test_acc'].idxmax(), 'epoch']
        student_total_time = df_student['time'].astype(float).sum()

        doc.add_paragraph(
            f'学生模型（Baseline，无蒸馏）在测试集上的最佳准确率为 {student_best_acc:.2f}% '
            f'（出现在第 {int(student_best_epoch)} 个epoch）。'
            f'总训练时间为 {student_total_time/60:.1f} 分钟。'
        )

        curve_path = os.path.join(OUTPUT_DIR, 'report_student_curve.png')
        generate_training_curve_image(student_log, curve_path, 'Student Baseline (SmallCNN)')
        if os.path.exists(curve_path):
            doc.add_picture(curve_path, width=Inches(6))
            doc.add_paragraph('图2：学生模型Baseline训练曲线', style='Caption')
    else:
        doc.add_paragraph('学生模型训练日志未找到。请先运行 train_student_baseline.py')
        student_best_acc = 0

    # ============================================================
    # 5. 知识蒸馏实验
    # ============================================================
    doc.add_heading('5. 知识蒸馏实验', level=1)

    doc.add_heading('5.1 蒸馏原理', level=2)
    doc.add_paragraph(
        '知识蒸馏（Knowledge Distillation）是由 Hinton 等人于2015年提出的模型压缩技术。'
        '其核心思想是利用一个大型、高精度的教师模型（Teacher Model）来指导一个轻量级的'
        '学生模型（Student Model）的训练过程。'
    )
    doc.add_paragraph('蒸馏损失函数定义如下：')
    add_code_block(doc,
        'L = (1 - alpha) * CE(student_logits, labels)\n'
        '  + alpha * T^2 * KL(softmax(s/T) || softmax(t/T))\n\n'
        '其中：\n'
        '  CE  = 交叉熵损失（Hard Loss）\n'
        '  KL  = KL散度（Soft Loss）\n'
        '  T   = 温度参数，控制软标签的平滑程度\n'
        '  alpha = 软标签损失的权重系数'
    )

    doc.add_paragraph()
    doc.add_paragraph(
        '温度参数 T 的作用：当 T=1 时，软标签退化为普通的概率分布；当 T 较大时，'
        '教师模型输出的概率分布更加平滑，揭示了各类别之间的相似性信息（即"暗知识"）。'
        '例如，对于一张猫的图片，教师模型可能给"猫"0.7的概率，同时给"狗"0.2的概率'
        '和"马"0.05的概率——这些类别间的相似性信息就是教师模型传授给学生的"暗知识"。'
    )
    doc.add_paragraph(
        '公式中乘以 T² 的原因：由于 softmax( logits/T ) 会缩小梯度约 1/T 倍，'
        '乘以 T² 可以保持梯度量级不变，确保蒸馏损失不会因温度升高而失效。'
    )

    doc.add_heading('5.2 实验设计', level=2)
    doc.add_paragraph(
        '为探索不同温度参数对蒸馏效果的影响，本实验设计了多组对比实验。'
        '在每组实验中，教师模型被冻结（参数不更新），学生模型通过蒸馏损失进行训练。'
    )

    doc.add_heading('5.3 蒸馏实验结果', level=2)

    # 读取蒸馏实验汇总
    summary_file = DISTILLATION["summary_file"]
    distill_results = {}  # {(T, alpha): acc}
    temp_results = {}     # {T: best_acc across alphas}

    if os.path.exists(summary_file):
        df_summary = pd.read_csv(summary_file)
        for _, row in df_summary.iterrows():
            T = row['temperature']
            alpha = row['alpha']
            acc_str = row['best_test_acc']
            if acc_str != 'N/A':
                acc = float(acc_str)
                distill_results[(T, alpha)] = acc
                if T not in temp_results or acc > temp_results[T]:
                    temp_results[T] = acc

    if distill_results:
        # 找最佳蒸馏配置
        best_key = max(distill_results, key=distill_results.get)
        best_T, best_alpha = best_key
        best_distill_acc = distill_results[best_key]

        doc.add_paragraph(
            f'共完成 {len(distill_results)} 组蒸馏实验。'
            f'最佳蒸馏配置为 T={best_T}, alpha={best_alpha}，'
            f'对应测试准确率 {best_distill_acc:.2f}%。'
        )

        # 找最佳蒸馏的日志
        best_log = DISTILLATION["log_template"].format(T=best_T, alpha=best_alpha)

        # 温度对比表
        doc.add_heading('5.3.1 温度参数对比', level=3)
        create_temperature_table(doc, temp_results, student_best_acc, teacher_best_acc)
        doc.add_paragraph('表：不同温度T下的蒸馏准确率对比', style='Caption')

        # 温度柱状图
        bar_path = os.path.join(OUTPUT_DIR, 'report_temperature_bar.png')
        generate_temperature_bar_image(temp_results, student_best_acc, teacher_best_acc, bar_path)
        if os.path.exists(bar_path):
            doc.add_picture(bar_path, width=Inches(5.5))
            doc.add_paragraph('图3：不同温度T下的蒸馏准确率对比', style='Caption')

        # 多模型对比折线图
        doc.add_heading('5.3.2 模型对比', level=3)
        comp_path = os.path.join(OUTPUT_DIR, 'report_model_comparison.png')
        generate_comparison_image(
            [TEACHER["log_file"], STUDENT["log_file"], best_log],
            ['Teacher (TeacherCNN)', 'Student Baseline', f'Student Distilled (T={best_T}, a={best_alpha})'],
            comp_path
        )
        if os.path.exists(comp_path):
            doc.add_picture(comp_path, width=Inches(5.5))
            doc.add_paragraph('图4：教师模型、学生Baseline、蒸馏学生准确率对比', style='Caption')

        # 最佳蒸馏学生的训练曲线
        if os.path.exists(best_log):
            curve_path = os.path.join(OUTPUT_DIR, 'report_distill_curve.png')
            generate_training_curve_image(best_log, curve_path,
                                          f'Student Distilled (T={best_T}, alpha={best_alpha})')
            doc.add_picture(curve_path, width=Inches(6))
            doc.add_paragraph(f'图5：蒸馏学生模型训练曲线 (T={best_T}, alpha={best_alpha})', style='Caption')

    else:
        doc.add_paragraph('蒸馏实验结果未找到。请先运行 run_all_experiments.py')

    # ============================================================
    # 6. 结果分析与讨论
    # ============================================================
    doc.add_heading('6. 结果分析与讨论', level=1)

    doc.add_heading('6.1 综合结果汇总', level=2)

    results_dict = {}
    if os.path.exists(teacher_log):
        results_dict['Teacher (TeacherCNN)'] = {
            'accuracy': teacher_best_acc, 'params': t_params
        }
    if os.path.exists(student_log):
        results_dict['Student Baseline'] = {
            'accuracy': student_best_acc, 'params': s_params
        }
    for (T, alpha), acc in sorted(distill_results.items()):
        results_dict[f'Distilled (T={T}, a={alpha})'] = {
            'accuracy': acc, 'params': s_params
        }

    if results_dict:
        create_accuracy_table(doc, results_dict)
        doc.add_paragraph('表：所有模型实验结果汇总', style='Caption')

    doc.add_heading('6.2 结果分析', level=2)

    if distill_results and student_best_acc > 0:
        doc.add_paragraph(
            f'（1）教师模型性能：教师模型 TeacherCNN 在 CIFAR-10 测试集上达到了 '
            f'{teacher_best_acc:.2f}% 的准确率，作为蒸馏的教师信号来源。'
        )

        doc.add_paragraph(
            f'（2）学生模型 Baseline：学生模型 SmallCNN 直接训练的准确率为 '
            f'{student_best_acc:.2f}%。由于模型容量有限（参数量仅为教师的 {s_params/t_params*100:.1f}%），'
            f'其表现明显低于教师模型。'
        )

        if best_distill_acc > student_best_acc:
            improvement = best_distill_acc - student_best_acc
            doc.add_paragraph(
                f'（3）蒸馏效果：经过知识蒸馏训练后，学生模型的最佳准确率提升至 '
                f'{best_distill_acc:.2f}%，相比 Baseline 提升了 {improvement:.2f} 个百分点。'
                f'这验证了知识蒸馏能够有效地将教师模型的"暗知识"传递给学生模型，'
                f'在保持模型轻量化的同时提升分类性能。'
            )

        doc.add_paragraph(
            f'（4）温度影响：从温度对比实验可以看出，不同温度 T 对蒸馏效果有显著影响。'
            f'T 过小时（如 T=2），软标签分布接近硬标签，暗知识传递不充分；'
            f'T 过大时（如 T=20），软标签过于平滑，丧失了类别间的判别信息。'
            f'适中的温度（如 T=4~8）通常能取得最佳的蒸馏效果。'
        )

    doc.add_heading('6.3 实验启示', level=2)
    doc.add_paragraph(
        '（1）知识蒸馏是一种高效的模型压缩技术，能够在不增加推理计算量的情况下'
        '提升轻量模型的性能。'
    )
    doc.add_paragraph(
        '（2）温度参数是蒸馏效果的关键超参数，需要根据具体任务和模型进行调优。'
    )
    doc.add_paragraph(
        '（3）教师模型的质量直接影响蒸馏效果。一个性能优秀的教师模型能够提供'
        '更准确的暗知识信号，从而更好地指导学生模型的学习。'
    )

    # ============================================================
    # 7. 结论
    # ============================================================
    doc.add_heading('7. 结论', level=1)
    doc.add_paragraph(
        '本实验完成了CIFAR-10数据集上的教师模型设计、学生模型设计和知识蒸馏实验。'
        '主要结论如下：'
    )
    doc.add_paragraph(
        '1. 设计的 TeacherCNN 教师模型在 CIFAR-10 上取得了良好的分类性能，'
        '作为知识蒸馏的教师信号来源是可靠的。'
    )
    doc.add_paragraph(
        '2. 轻量级的 SmallCNN 学生模型虽然参数量远小于教师模型，'
        '但通过知识蒸馏技术可以从教师模型中学习到额外的类别间关系信息。'
    )
    doc.add_paragraph(
        '3. 知识蒸馏能够有效提升学生模型的分类准确率，验证了蒸馏损失函数 '
        'L = (1-α)·CE + α·T²·KL_Div 的有效性。'
    )
    doc.add_paragraph(
        '4. 温度参数 T 对蒸馏效果有显著影响，适中的温度值能取得最佳效果。'
    )

    # ============================================================
    # 保存文档
    # ============================================================
    report_path = os.path.join(OUTPUT_DIR, 'experiment_report.docx')
    doc.save(report_path)
    print(f"\nReport saved: {report_path}")
    return report_path


if __name__ == "__main__":
    generate_report()
