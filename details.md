# VQ-VAE + PatchCore 工业缺陷检测 使用文档

## 项目概述

本项目是一个基于 **VQ-VAE（Vector Quantized Variational Autoencoder）** 与 **PatchCore** 思想融合的工业图像异常检测系统。与原版 PatchCore 使用预训练骨干网络提取特征 + 记忆库不同，本项目将 VQ-VAE 作为核心特征提取与重建模块，通过训练阶段学习正常样本的分布，在推理阶段通过**重建误差**和**结构相似度差异**定位异常区域。

支持的数据集：**MVTec AD**、**MPDD**、**CarDD**

---

## 整体流程

```
输入图像
   │
   ▼
┌──────────────────────────────────┐
│         VQ-VAE 训练阶段          │
│  编码器(Encoder) → 量化器        │
│  (VQ / EMA-VQ / FSQ) → 解码器   │
│  损失：MSE重建损失 + VQ损失      │
└──────────────────────────────────┘
   │（仅用正常训练图）
   ▼
┌──────────────────────────────────┐
│         VQ-VAE 推理阶段          │
│  编码 → 量化 → 解码 → 重建图    │
│  计算：MAD图 × (1 - SSIM图)     │
│  → 异常热力图 + 图像级分数       │
└──────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────┐
│         评估指标                 │
│  Image-AUROC / Pixel-AUROC       │
└──────────────────────────────────┘
```

---

## 核心方法详解

### 1. 编码器（Encoder）

位于 `src/patchcore/vqvae_model.py`

编码器支持两种模式：

#### 模式一：自定义 CNN 编码器（默认）

采用三个卷积 Block，总下采样倍率为 **4x**：

| 层 | 操作 | 下采样 |
|---|---|---|
| Block 1 | Conv2d(in→H/2, k=4, s=2) + BN + ReLU | /2 |
| Block 2 | Conv2d(H/2→H, k=4, s=2) + BN + ReLU | /2 |
| Block 3 | Conv2d(H→H, k=3, s=1) + BN + ReLU | 无 |
| conv_out | Conv2d(H→H, k=3, s=1) | 无 |

其中 `H = vq_num_hiddens`（主干通道数）。

支持通过 `--vq_layers_to_extract_from` 指定提取中间层特征后拼接（多尺度融合，双线性插值对齐尺寸）。

#### 模式二：预训练骨干网络编码器

通过 `--vq_backbone_name` 指定，使用 Hook 机制提取指定层特征，支持的骨干网络（`src/patchcore/backbones.py`）包括：

- ResNet 系列：`resnet50`, `resnet101`, `wideresnet50`, `wideresnet101`
- EfficientNet 系列：`efficientnet_b1` ~ `efficientnet_b7`, `efficientnetv2_m/l`
- ViT 系列：`vit_small`, `vit_base`, `vit_large`, `vit_swin_base/large`
- DenseNet 系列：`densenet121`, `densenet201`
- 等多种 timm / torchvision 模型

---

### 2. 量化器（Quantizer）

本项目提供三种量化方案，可通过参数切换：

#### 2.1 标准向量量化器 `VectorQuantizer`（默认）

实现了 VQ-VAE 原论文的离散码本：

- 码本大小：`K = vq_num_embeddings`，向量维度：`D = vq_embedding_dim`
- 距离计算：欧氏距离 `||z_e - e_i||²`，找最近邻
- **损失函数**（两项）：
  - **码本损失（Codebook Loss）**：`||sg[z_e] - e||²`，将码本向量拉向编码器输出
  - **承诺损失（Commitment Loss）**：`β × ||z_e - sg[e]||²`，将编码器输出拉向码本向量
  - 合计：`VQ_loss = q_latent_loss + β × e_latent_loss`
- **直通估计器（STE）**：反向传播时梯度绕过量化操作直接传给编码器

#### 2.2 EMA 向量量化器 `EMAVectorQuantizer`（推荐，`--vq_use_ema_codebook`）

使用**指数移动平均（EMA）**更新码本，解决码本塌缩（codebook collapse）问题：

- 码本不参与梯度更新，通过 EMA 跟踪每个码本向量的使用情况：
  ```
  ema_cluster_size = decay × ema_cluster_size + (1 - decay) × cluster_size
  ema_embedding   = decay × ema_embedding   + (1 - decay) × Σ(z_e)
  embedding = ema_embedding / normalized_cluster_size
  ```
- **损失函数**（仅有承诺损失）：`VQ_loss = β × ||sg[quantized] - z_e||²`
- 参数：`--vq_ema_decay`（默认 0.99）、`--vq_ema_eps`（Laplace平滑，默认 1e-5）

#### 2.3 有限标量量化器 `ScalarQuantizer (FSQ)`（`--vq_use_fsq`）

基于 CLIP-FSQAE 论文，无显式码本，解决死码问题：

- 通过 **Tanh + 缩放 + 取整** 实现隐式量化：
  ```
  z_tanh = tanh(z)
  scale_i = (level_i - 1) / 2
  z_q_i = round(z_tanh_i × scale_i)
  ```
- **无码本损失**，`VQ_loss = 0`
- 隐式码本大小 = `∏ levels`（例如 `[8,6,4,4,4]` → 6144 个组合）
- 反向传播使用 STE，梯度流过 tanh
- 参数：`--vq_fsq_levels 8,6,4,4,4`

**对比总结**：

| 量化器 | 码本更新方式 | VQ损失项 | 死码风险 |
|---|---|---|---|
| VectorQuantizer | 梯度下降 | 码本损失 + 承诺损失 | 高 |
| EMAVectorQuantizer | 指数移动平均 | 仅承诺损失 | 中 |
| ScalarQuantizer (FSQ) | 无码本 | 无 | 无 |

---

### 3. 解码器（Decoder）

使用 **PixelShuffle 亚像素卷积**进行上采样，总放大 4x（与编码器下采样对应）：

| 层 | 操作 |
|---|---|
| conv_in | Conv2d(H→H) + ReLU |
| ps1 | Conv2d(H→H×4) + PixelShuffle(2) + ReLU（×2 放大） |
| ps2 | Conv2d(H→H/2×4) + PixelShuffle(2) + ReLU（×2 放大） |
| conv_out | Conv2d(H/2→out_channels) |

若重建图尺寸与输入不一致（使用骨干网络时可能出现），会通过双线性插值自动对齐到输入尺寸。

---

### 4. 总损失函数

训练阶段的总损失为：

```
L_total = L_recon + L_vq

L_recon = MSE(x_recon, x)           # 像素级均方误差重建损失
L_vq    = VQ量化损失（依量化器类型而定）
```

---

### 5. 异常图计算（推理阶段）

推理时综合**潜空间量化误差**与**图像空间结构相似度**计算异常热力图：

#### 5.1 MAD 图（Mean Absolute Difference，潜空间）

在**隐空间**计算编码器输出 `z_e` 与量化后向量 `z_q` 的逐像素绝对差均值：

```
MAD_map = mean(|z_e - z_q|, dim=channel)   # 形状: [B, H_lat, W_lat]
```

正常区域量化误差小，异常区域因超出码本覆盖范围导致较大偏差。

#### 5.2 SSIM 图（结构相似度，图像空间）

在**像素空间**计算原始输入与重建图之间的结构相似性：

```
SSIM_map = ssim(x, x_recon)   # skimage 逐像素 SSIM，形状: [B, H, W]
```

正常区域重建质量高（SSIM 接近 1），异常区域重建失真（SSIM 偏低）。

#### 5.3 融合异常图

将 MAD 图（上采样对齐至图像尺寸）与 SSIM 差异图相乘，获得最终异常热力图：

```
anomaly_map = MAD_map × (1 - SSIM_map)
```

两者均高时才判定为强异常，有效抑制假阳性。

#### 5.4 图像级异常分数

取异常图中**最高 0.1%（Top-K）像素**的均值作为图像级异常分数：

```
k = max(1, int(H × W × 0.001))
image_score = mean(topk(anomaly_map.flatten(), k))
```

#### 5.5 分割图后处理

通过 `RescaleSegmentor` 对异常图进行：
1. 双线性插值上采样至输入图像尺寸
2. 高斯平滑（σ = 4）去噪

---

### 6. 评估指标

| 指标 | 说明 |
|---|---|
| `instance_auroc` | 图像级 AUROC（正常/异常分类能力） |
| `full_pixel_auroc` | 全部测试图的像素级 AUROC |
| `anomaly_pixel_auroc` | 仅含异常图像的像素级 AUROC |

额外保存：
- 每个 Epoch 的 **Perplexity**（码本使用熵，越高表示码本利用越均匀）到 `perplexity_epoch.csv`
- 若开启 `--save_segmentation_images`：保存四联图（原图 / GT Mask / 异常热力图 / 重建图）及 PSNR/SSIM 指标 CSV

---

## 文件结构

```
patchcore-inspection/
├── bin/
│   ├── run_patchcore.py            # 训练+评估入口
│   └── load_and_evaluate_patchcore.py  # 加载已训练模型评估
├── src/patchcore/
│   ├── patchcore.py               # PatchCore主类（训练、推理、保存/加载）
│   ├── vqvae_model.py             # VQ-VAE模型（编码器、量化器、解码器）
│   ├── backbones.py               # 预训练骨干网络列表
│   ├── common.py                  # 公共模块（FAISS、特征聚合、分割器）
│   ├── metrics.py                 # 评估指标（AUROC等）
│   ├── utils.py                   # 工具函数
│   └── datasets/
│       ├── mvtec.py               # MVTec AD 数据集
│       ├── mpdd.py                # MPDD 数据集
│       └── cardd.py               # CarDD 数据集
├── sample_training.sh             # 训练示例脚本
├── sample_evaluation.sh           # 评估示例脚本
└── 使用文档.md                    # 本文档
```

---

## 快速开始

### 环境准备

```bash
export PYTHONPATH=src
```

### 训练（以 MPDD 数据集为例）

```bash
python bin/run_patchcore.py \
    --gpu 0 \
    --seed 123 \
    --save_segmentation_images \
    --log_group <实验名称> \
    --log_project <项目名称> \
    results \
patch_core \
    --vq_in_channels 3 \
    --vq_out_channels 3 \
    --vq_num_hiddens 128 \
    --vq_num_embeddings 128 \
    --vq_embedding_dim 128 \
    --vq_commitment_cost 0.25 \
    --vq_lr 0.001 \
    --vq_epochs 50 \
    --vq_use_ema_codebook \
    --vq_ema_decay 0.99 \
    --vq_ema_eps 1e-5 \
    --patchsize 3 \
dataset \
    --resize 256 \
    --imagesize 224 \
    -d bracket_black \
    -d bracket_brown \
    -d metal_plate \
    mpdd MPDD
```

或直接使用示例脚本：

```bash
bash sample_training.sh
```

### 加载已保存模型评估

```bash
python bin/load_and_evaluate_patchcore.py \
    --gpu -1 \
    --seed 0 \
    evaluated_results/my_model \
patch_core_loader \
    -p /models/my_model/models/mvtec_bottle \
    --no-faiss_on_gpu \
dataset \
    --resize 366 \
    --imagesize 320 \
    -d bottle \
    mvtec mvtec
```

或直接使用示例脚本：

```bash
bash sample_evaluation.sh
```

---

## 主要参数说明

### 全局参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--gpu` | `0` | 使用的 GPU 编号，`-1` 使用 CPU |
| `--seed` | `0` | 随机种子 |
| `--log_group` | `group` | 实验组名，用于区分结果目录 |
| `--log_project` | `project` | 项目名，结果保存于 `results/<log_project>/<log_group>/` |
| `--save_segmentation_images` | 否 | 保存重建图、异常热力图四联图及 PSNR/SSIM 指标 |
| `--save_patchcore_model` | 否 | 保存训练好的 VQ-VAE 模型权重 |
| `--save_recon` | 否 | 单独开启重建图保存和 PSNR/SSIM 计算 |

### VQ-VAE 模型参数（`patch_core` 子命令）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--vq_in_channels` | `3` | 输入图像通道数（RGB=3） |
| `--vq_out_channels` | `3` | 输出重建图通道数 |
| `--vq_num_hiddens` | `128` | 编码器/解码器主干通道数 |
| `--vq_num_embeddings` | `512` | 码本大小 K（码本向量数） |
| `--vq_embedding_dim` | `64` | 码本向量维度 D |
| `--vq_commitment_cost` | `0.25` | 承诺损失权重 β |
| `--vq_lr` | `1e-4` | 训练学习率 |
| `--vq_epochs` | `10` | 训练轮数 |
| `--patchsize` | `3` | PatchMaker 的 Patch 大小（卷积展开核尺寸） |
| `--vq_use_ema_codebook` | 否 | 启用 EMA 码本更新（推荐） |
| `--vq_ema_decay` | `0.99` | EMA 衰减系数 |
| `--vq_ema_eps` | `1e-5` | EMA Laplace 平滑项 |
| `--vq_use_fsq` | 否 | 启用有限标量量化（FSQ） |
| `--vq_fsq_levels` | `5,5,5,5,5` | FSQ 每维量化等级，逗号分隔 |
| `--vq_backbone_name` | `None` | 预训练骨干网络名称（如 `resnet50`） |
| `--vq_layers_to_extract_from` | `None` | 骨干网络特征提取层（可多个，如 `layer2`） |

### 数据集参数（`dataset` 子命令）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `name` | 必填 | 数据集类型：`mvtec` / `mpdd` / `cardd` |
| `data_path` | 必填 | 数据集根目录路径 |
| `-d` / `--subdatasets` | 必填 | 子类别名（可多次指定） |
| `--resize` | `256` | 图像先缩放到此尺寸 |
| `--imagesize` | `224` | 再中心裁剪到此尺寸 |
| `--batch_size` | `2` | 批大小 |
| `--num_workers` | `8` | DataLoader 工作进程数 |
| `--train_val_split` | `1.0` | 训练集比例（1.0 = 全部用于训练） |

---

## 数据集目录结构

### MVTec AD / MPDD（相同结构）

```
数据集根目录/
└── <类别名>/
    ├── train/
    │   └── good/
    │       ├── 000.png
    │       └── ...
    ├── test/
    │   ├── good/
    │   │   └── ...
    │   └── <缺陷类型>/
    │       └── ...
    └── ground_truth/
        └── <缺陷类型>/
            └── <对应掩码>.png
```

### 支持的 MPDD 子类别

`bracket_black`, `bracket_brown`, `bracket_white`, `connector`, `metal_plate`, `tubes`

---

## 输出结果说明

训练完成后，结果保存于 `results/<log_project>/<log_group>/`：

```
results/
└── <log_project>/
    └── <log_group>/
        ├── results.csv                          # 各子类别评估指标汇总
        ├── perplexity_epoch.csv                 # 每 Epoch 的码本 Perplexity 曲线
        ├── models/
        │   └── <dataset_name>/
        │       ├── vqvae_model.pth              # 模型权重（--save_patchcore_model）
        │       └── patchcore_params.pkl         # 模型超参数
        └── recon_segmentation_images/
            └── <dataset_name>/
                ├── reconstruction_metrics.csv   # 每张图的 PSNR/SSIM
                └── <图像名>.png                 # 四联图（原图/GT/热力图/重建）
```

`results.csv` 包含以下列：

| 列名 | 说明 |
|---|---|
| `instance_auroc` | 图像级 AUROC |
| `full_pixel_auroc` | 全像素级 AUROC |
| `anomaly_pixel_auroc` | 异常样本像素级 AUROC |

---

## 典型实验配置参考

### 配置一：MPDD 标准训练（EMA 码本，50 Epoch）

```bash
python bin/run_patchcore.py --gpu 0 --seed 123 \
    --save_segmentation_images \
    --log_group mpdd_ema_e50 --log_project MPDD results \
patch_core \
    --vq_in_channels 3 --vq_out_channels 3 \
    --vq_num_hiddens 128 --vq_num_embeddings 128 \
    --vq_embedding_dim 128 --vq_commitment_cost 0.25 \
    --vq_lr 0.001 --vq_epochs 50 \
    --vq_use_ema_codebook --vq_ema_decay 0.99 \
    --patchsize 3 \
dataset --resize 256 --imagesize 224 \
    -d bracket_black -d connector -d metal_plate \
    mpdd MPDD
```

### 配置二：使用 FSQ 替代 VQ

```bash
python bin/run_patchcore.py --gpu 0 --seed 123 \
    --log_group mpdd_fsq --log_project MPDD results \
patch_core \
    --vq_num_hiddens 128 --vq_num_embeddings 128 \
    --vq_embedding_dim 128 --vq_lr 0.001 --vq_epochs 50 \
    --vq_use_fsq --vq_fsq_levels 8,6,4,4,4 \
    --patchsize 3 \
dataset --resize 256 --imagesize 224 \
    -d bracket_black mpdd MPDD
```

### 配置三：使用预训练 ResNet50 作为编码器骨干

```bash
python bin/run_patchcore.py --gpu 0 --seed 123 \
    --log_group mpdd_resnet50 --log_project MPDD results \
patch_core \
    --vq_num_hiddens 128 --vq_num_embeddings 256 \
    --vq_embedding_dim 128 --vq_lr 0.0001 --vq_epochs 20 \
    --vq_backbone_name resnet50 \
    --vq_layers_to_extract_from layer2 \
    --vq_layers_to_extract_from layer3 \
    --patchsize 3 \
dataset --resize 256 --imagesize 224 \
    -d bracket_black mpdd MPDD
```

---

## 关键模块索引

| 功能 | 文件 | 类/函数 |
|---|---|---|
| 训练主流程 | `src/patchcore/patchcore.py` | `PatchCore.fit()` → `_train_vqvae()` |
| 推理异常图 | `src/patchcore/patchcore.py` | `PatchCore._predict()` |
| MAD 图计算 | `src/patchcore/patchcore.py` | `calculate_mad_map()` |
| SSIM 图计算 | `src/patchcore/patchcore.py` | `calculate_ssim_map()` |
| VQ-VAE 前向 | `src/patchcore/vqvae_model.py` | `VQVAE.forward()` |
| 标准 VQ 量化 | `src/patchcore/vqvae_model.py` | `VectorQuantizer` |
| EMA VQ 量化 | `src/patchcore/vqvae_model.py` | `EMAVectorQuantizer` |
| FSQ 量化 | `src/patchcore/vqvae_model.py` | `ScalarQuantizer` |
| 分割图后处理 | `src/patchcore/common.py` | `RescaleSegmentor` |
| AUROC 计算 | `src/patchcore/metrics.py` | `compute_imagewise_retrieval_metrics()` |
| 结果可视化 | `bin/run_patchcore.py` | `run()` 函数中四联图保存逻辑 |
