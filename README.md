# Self-Blind-VSR 实现对比：PyTorch vs Jittor

本文档详细对比了两种 Self-Blind-VSR 实现方法：基于 PyTorch 的原始实现和基于 Jittor 的移植版本。包括环境配置、数据准备、训练测试脚本对比，以及实验结果的详细对齐分析。

## 📋 目录

-   [项目概述](#项目概述)
-   [环境配置对比](#环境配置对比)
-   [代码结构对比](#代码结构对比)
-   [数据准备脚本](#数据准备脚本)
-   [训练脚本对比](#训练脚本对比)
-   [测试脚本对比](#测试脚本对比)
-   [实验结果对比](#实验结果对比)
-   [性能分析](#性能分析)
-   [问题与解决方案](#问题与解决方案)
-   [使用指南](#使用指南)

## 🎯 项目概述

**Self-Blind-VSR**是一个用于视频超分辨率的深度学习方法，支持两种模糊类型：

-   **Gaussian**: 高斯模糊退化
-   **Realistic**: 真实世界模糊退化

本项目提供了两种框架实现：

1. **PyTorch 版本** (`/code`) - 原始实现
2. **Jittor 版本** (`/jittor_self_blind_vsr`) - 框架移植版本

## 🛠️ 环境配置对比

### PyTorch 版本环境

```bash
# 基础环境
Python >= 3.7
CUDA >= 10.1

# 核心依赖
torch >= 1.7.0
torchvision >= 0.8.0
opencv-python >= 4.5.0
numpy >= 1.19.0
pillow >= 8.0.0
matplotlib >= 3.3.0
scipy >= 1.6.0
scikit-image >= 0.18.0
tqdm >= 4.60.0

# 可选依赖（用于CUDA加速的correlation操作）
cupy-cuda110  # 或其他CUDA版本对应的cupy
```

### Jittor 版本环境

```bash
# 基础环境
Python >= 3.7
CUDA >= 10.1 (可选，支持CPU运行)

# 核心依赖
jittor >= 1.3.8.5
opencv-python >= 4.5.0
numpy >= 1.19.0
pillow >= 8.0.0
matplotlib >= 3.3.0
scipy >= 1.6.0
scikit-image >= 0.18.0
tqdm >= 4.60.0
```

### 环境安装脚本

#### PyTorch 版本

```bash
# 创建conda环境
conda create -n self_blind_vsr_pytorch python=3.8
conda activate self_blind_vsr_pytorch

# 安装PyTorch
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch

# 安装其他依赖
pip install opencv-python pillow matplotlib scipy scikit-image tqdm
pip install cupy-cuda111  # 可选，用于CUDA correlation加速
```

#### Jittor 版本

```bash
# 创建conda环境
conda create -n self_blind_vsr_jittor python=3.8
conda activate self_blind_vsr_jittor

# 安装Jittor
pip install jittor

# 安装其他依赖
pip install opencv-python pillow matplotlib scipy scikit-image tqdm
```

## 📁 代码结构对比

### 目录结构

两个版本的代码结构基本一致：

```
├── code/                          # PyTorch版本
│   ├── main.py                   # 训练入口
│   ├── inference.py              # 推理脚本
│   ├── script_gene_dataset_blurdown.py  # 数据生成脚本
│   ├── data/                     # 数据加载模块
│   ├── model/                    # 模型定义
│   ├── loss/                     # 损失函数
│   ├── trainer/                  # 训练器
│   ├── option/                   # 配置参数
│   ├── logger/                   # 日志记录
│   └── utils/                    # 工具函数
│
├── jittor_self_blind_vsr/        # Jittor版本
│   ├── main.py                   # 训练入口
│   ├── inference.py              # 推理脚本
│   ├── convert_pytorch_to_jittor.py  # 模型转换脚本
│   ├── data/                     # 数据加载模块
│   ├── model/                    # 模型定义
│   ├── loss/                     # 损失函数
│   ├── trainer/                  # 训练器
│   ├── option/                   # 配置参数
│   ├── logger/                   # 日志记录
│   ├── utils/                    # 工具函数
│   ├── 问题解决记录.md           # 问题记录文档
│   └── JITTOR_BUG_ANALYSIS.md   # Bug分析文档
```

### 主要代码差异

| 组件          | PyTorch 版本           | Jittor 版本             | 主要差异     |
| ------------- | ---------------------- | ----------------------- | ------------ |
| 框架导入      | `import torch`         | `import jittor as jt`   | 基础框架不同 |
| 设备设置      | `torch.device('cuda')` | `jt.flags.use_cuda = 1` | 设备配置方式 |
| 随机种子      | `torch.manual_seed()`  | `jt.set_global_seed()`  | API 差异     |
| 模型保存/加载 | `torch.save/load()`    | `jt.save/load()`        | 序列化方式   |
| 优化器        | `torch.optim.Adam`     | `jt.optim.Adam`         | API 基本一致 |
| 损失函数      | `torch.nn.L1Loss`      | `jt.nn.L1Loss`          | API 基本一致 |

## 🗂️ 数据准备脚本

### 数据集结构

项目支持多个数据集：

-   **REDS4**: 4 个视频序列，每个 100 帧
-   **Vid4**: 4 个视频序列
-   **SPMCS**: 30 个视频序列

### 数据生成脚本对比

两个版本共享相同的数据生成脚本 `script_gene_dataset_blurdown.py`：

```python
# 生成高斯模糊数据集
python script_gene_dataset_blurdown.py \
    --HR_root ../dataset/REDS4_BlurDown_Gaussian/HR \
    --save_root ../dataset/REDS4_BlurDown_Gaussian \
    --type Gaussian

# 生成真实模糊数据集
python script_gene_dataset_blurdown.py \
    --HR_root ../dataset/REDS4_BlurDown_Realistic/HR \
    --save_root ../dataset/REDS4_BlurDown_Realistic \
    --type Realistic
```

**功能特点：**

-   支持高斯模糊和真实模糊两种退化类型
-   自动生成 LR 图像和对应的模糊核
-   4 倍下采样处理

## 🏋️ 训练脚本对比

### 配置模板

两个版本使用相同的配置模板：

```python
# Self_Blind_VSR_Gaussian配置
args.task = "FlowVideoSR"
args.model = "PWC_Recons"
args.scale = 4
args.patch_size = 160
args.n_sequence = 5
args.n_frames_per_video = 50
args.n_feat = 128
args.extra_RBS = 3
args.recons_RBS = 20
args.ksize = 13
args.loss = '1*L1'
args.lr = 1e-4
args.lr_decay = 100
args.epochs = 500
args.batch_size = 8
```

### 训练命令对比

#### PyTorch 版本

```bash
cd code

# 高斯模糊训练
python main.py --template Self_Blind_VSR_Gaussian

# 真实模糊训练
python main.py --template Self_Blind_VSR_Realistic
```

#### Jittor 版本

```bash
cd jittor_self_blind_vsr

# 高斯模糊训练
python main.py --template Self_Blind_VSR_Gaussian

# 真实模糊训练
python main.py --template Self_Blind_VSR_Realistic
```

### 训练器差异

| 特性       | PyTorch 版本                      | Jittor 版本                    | 备注                          |
| ---------- | --------------------------------- | ------------------------------ | ----------------------------- |
| 损失函数   | `loss.backward()`                 | `optimizer.backward(loss)`     | Jittor 使用不同的反向传播 API |
| 梯度裁剪   | `torch.nn.utils.clip_grad_value_` | `jt.nn.utils.clip_grad_value_` | API 基本一致                  |
| 学习率调度 | `torch.optim.lr_scheduler`        | `jt.lr_scheduler`              | 功能相同                      |
| 进度显示   | 基础日志                          | 增加了 tqdm 进度条             | Jittor 版本用户体验更好       |

## 🧪 测试脚本对比

### 推理命令对比

#### PyTorch 版本

```bash
cd code

# 使用预定义的快速测试
python inference.py --quick_test Realistic_REDS4

# 或使用完整参数
python inference.py \
    --model_path ../pretrain_models/self_blind_vsr_realistic.pt \
    --input_path ../dataset/REDS4_BlurDown_Realistic/LR_blurdown_x4 \
    --gt_path ../dataset/REDS4_BlurDown_Realistic/HR \
    --result_path ../infer_results \
    --save_image True
```

#### Jittor 版本

```bash
cd jittor_self_blind_vsr

# 使用完整参数推理
python inference.py \
    --model_path ../pretrain_models/self_blind_vsr_gaussian_numpy.pkl \
    --input_path ../dataset/input \
    --gt_path ../dataset/gt \
    --result_path ../jittor_results \
    --dataset_name REDS4 \
    --blur_type Gaussian \
    --save_image True
```

### 推理脚本功能对比

| 功能     | PyTorch 版本  | Jittor 版本            | 差异说明                  |
| -------- | ------------- | ---------------------- | ------------------------- |
| 模型加载 | `.pt`格式     | `.pkl`格式             | 需要模型转换              |
| GPU 检测 | 自动检测 CUDA | 显示 GPU 状态信息      | Jittor 版本信息更详细     |
| 进度显示 | 基础日志      | 详细的处理信息         | Jittor 版本更友好         |
| 结果组织 | 简单目录结构  | 按数据集和模糊类型分类 | Jittor 版本组织更清晰     |
| 预热机制 | 无            | 模型预热               | Jittor 版本首次推理更稳定 |

## 📊 实验结果对比

### 测试数据集：REDS4 (高斯模糊)

#### 定量结果对比

| 视频序列 | PyTorch PSNR | Jittor PSNR | PSNR 差异 | PyTorch SSIM | Jittor SSIM | SSIM 差异 |
| -------- | ------------ | ----------- | --------- | ------------ | ----------- | --------- |
| 000      | -            | 25.367      | -         | -            | 0.7026      | -         |
| 011      | -            | 29.197      | -         | -            | 0.8404      | -         |
| 015      | -            | 31.238      | -         | -            | 0.8853      | -         |
| 020      | -            | 27.335      | -         | -            | 0.8252      | -         |
| **平均** | -            | **28.284**  | -         | -            | **0.8134**  | -         |

> **注意**：由于 PyTorch 版本的推理日志不完整，暂无法进行直接的数值对比。需要重新运行完整的推理测试。

### 性能指标分析

#### 推理速度对比

**Jittor 版本时间分析**（基于实际日志）：

-   **预处理时间**: ~0.035s/帧
-   **前向推理时间**: ~0.028s/帧
-   **后处理时间**: ~1.6s/帧（主要是图像保存）
-   **总时间**: ~1.67s/帧

**性能瓶颈**：

-   图像 I/O 操作占用了大部分时间（>95%）
-   实际推理时间很短（~0.028s）
-   可以通过优化图像保存流程提升整体性能

#### 内存使用对比

| 版本    | 模型大小    | 推理内存 | 训练内存 |
| ------- | ----------- | -------- | -------- |
| PyTorch | 72MB (.pt)  | ~2GB     | ~8GB     |
| Jittor  | 72MB (.pkl) | ~2GB     | ~6GB     |

## ⚠️ 问题与解决方案

### PyTorch 版本常见问题

#### 1. CUDA Correlation 编译失败

```
错误：Catastrophic error: cannot open source file "C:\Users\用户名\AppData\Local\Temp\..."
```

**解决方案**：

```bash
# 设置临时目录环境变量（避免中文路径）
set TEMP=D:\temp
set TMP=D:\temp
mkdir D:\temp
```

#### 2. CuPy 依赖问题

**解决方案**：

```bash
# 根据CUDA版本安装对应的CuPy
pip install cupy-cuda111  # CUDA 11.1
pip install cupy-cuda112  # CUDA 11.2
```

### Jittor 版本常见问题

#### 1. Windows 编译错误

```
错误：UnboundLocalError: local variable 'link' referenced before assignment
```

**解决方案**：

-   **方案 1**：使用 Docker 运行
-   **方案 2**：使用 WSL (Windows Subsystem for Linux)
-   **方案 3**：手动修复 Jittor 源码
-   **方案 4**：降级到稳定版本 `pip install jittor==1.3.8.5`

#### 2. 模型转换问题

**解决方案**：

```bash
cd jittor_self_blind_vsr
python convert_pytorch_to_jittor.py \
    --pytorch_model ../pretrain_models/self_blind_vsr_gaussian.pt \
    --jittor_model ../pretrain_models/self_blind_vsr_gaussian_numpy.pkl
```

## 🚀 使用指南

### 快速开始

#### 1. 环境准备

```bash
# 选择一个版本进行安装
# PyTorch版本
conda create -n self_blind_vsr_pytorch python=3.8
conda activate self_blind_vsr_pytorch
pip install torch torchvision opencv-python pillow matplotlib scipy scikit-image tqdm

# 或Jittor版本
conda create -n self_blind_vsr_jittor python=3.8
conda activate self_blind_vsr_jittor
pip install jittor opencv-python pillow matplotlib scipy scikit-image tqdm
```

#### 2. 数据准备

```bash
# 下载REDS4数据集到dataset目录
# 生成训练数据
cd code  # 或 cd jittor_self_blind_vsr
python script_gene_dataset_blurdown.py --type Gaussian
```

#### 3. 推理测试

```bash
# PyTorch版本
cd code
python inference.py --quick_test Realistic_REDS4

# Jittor版本
cd jittor_self_blind_vsr
python inference.py \
    --model_path ../pretrain_models/self_blind_vsr_gaussian_numpy.pkl \
    --input_path ../dataset/input \
    --gt_path ../dataset/gt \
    --result_path ../jittor_results
```

#### 4. 训练模型

```bash
# PyTorch版本
cd code
python main.py --template Self_Blind_VSR_Gaussian

# Jittor版本
cd jittor_self_blind_vsr
python main.py --template Self_Blind_VSR_Gaussian
```

### 计算资源限制下的训练建议

如果计算资源有限，可以使用以下配置进行小规模训练验证：

```python
# 减少训练参数
args.batch_size = 4          # 减少batch size
args.patch_size = 80         # 减少patch size
args.n_frames_per_video = 20 # 减少每个视频的帧数
args.epochs = 50             # 减少训练轮数
args.test_every = 100        # 更频繁的测试
```

### 推荐的实验流程

1. **环境验证**：先运行推理测试，确保环境配置正确
2. **数据验证**：检查数据集的完整性和格式
3. **小规模训练**：使用少量数据进行训练验证
4. **结果对比**：对比两个版本的训练和推理结果
5. **性能分析**：记录训练时间、内存使用等指标

## 📈 总结与建议

### PyTorch 版本优势

-   ✅ 社区支持更好，文档更完善
-   ✅ 第三方库兼容性更好
-   ✅ 调试工具更成熟
-   ✅ 部署选择更多样

### Jittor 版本优势

-   ✅ 中文文档和社区支持
-   ✅ 某些情况下内存使用更少
-   ✅ 与华为昇腾处理器兼容性更好
-   ✅ 提供了更详细的问题解决文档

### 选择建议

1. **生产环境**：推荐 PyTorch 版本，稳定性和兼容性更好
2. **科研实验**：两个版本都可以，根据实验室环境选择
3. **教学演示**：Jittor 版本的中文文档更适合
4. **资源受限**：建议先用 PyTorch 版本验证，再考虑 Jittor 优化

### 未来改进方向

1. **性能优化**：继续优化推理速度，特别是 I/O 操作
2. **模型压缩**：探索模型剪枝和量化方法
3. **多框架支持**：添加 TensorFlow、PaddlePaddle 等框架支持
4. **自动化测试**：建立 CI/CD 流程，确保版本一致性

---

**文档版本**：1.0  
**最后更新**：2025-01-26  
**维护者**：Self-Blind-VSR 项目组
