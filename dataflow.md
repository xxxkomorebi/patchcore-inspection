# 数据流

## 训练阶段 (`PatchCore.fit` → `_train_vqvae`)

```
输入图像 x [B, 3, 224, 224]
│
├─────────────────────────────────────────────────────┐
│                                                     │
▼                                                     ▼
┌─────────────── VQVAE Forward ───────────────┐   ┌──────────────────┐
│                                              │   │  PerceptualLoss  │
│  Encoder (3 conv blocks, stride-2 下采样)    │   │  (冻结 VGG16)    │
│  x → block1(/2) → block2(/2) → block3 →     │   │                  │
│      conv_out                                │   │  x ──→ relu1_2   │
│         │                                    │   │    ──→ relu2_2   │
│         ▼                                    │   │    ──→ relu3_3   │
│  z_e [B, 128, 56, 56]                       │   │                  │
│         │                                    │   │  x_recon ──→ ...│
│         ▼                                    │   │                  │
│  pre_quantization_conv (1×1)                 │   │  L1 距离求和     │
│  z_e [B, 128, 56, 56] → [B, D, 56, 56]      │   │         │        │
│         │                                    │   │         ▼        │
│         ▼                                    │   │  perceptual_loss │
│  ┌─── Quantizer (EMA/VQ/FSQ) ───┐           │   └────────┬─────────┘
│  │  z_e → 找最近码本向量 → z_q   │           │            │
│  │  vq_loss (commitment loss)    │           │            │
│  │  perplexity (码本使用率)       │           │            │
│  └───────────────────────────────┘           │            │
│         │                                    │            │
│         ▼                                    │            │
│  post_quantization_conv (1×1)                │            │
│  z_q [B, D, 56, 56] → [B, 128, 56, 56]      │            │
│         │                                    │            │
│         ▼                                    │            │
│  ┌─── Decoder ───────────────────────┐       │            │
│  │  conv_in (3×3)                    │       │            │
│  │       │                           │       │            │
│  │       ▼                           │       │            │
│  │  ResidualBlock × N                │       │            │
│  │  (Conv3x3→BN→ReLU→Conv3x3→BN     │       │            │
│  │   + skip connection)              │       │            │
│  │       │                           │       │            │
│  │       ▼                           │       │            │
│  │  PixelShuffle ×2 (↑2x)           │       │            │
│  │       │                           │       │            │
│  │       ▼                           │       │            │
│  │  PixelShuffle ×2 (↑2x)           │       │            │
│  │       │                           │       │            │
│  │       ▼                           │       │            │
│  │  conv_out (3×3)                   │       │            │
│  └───────────────────────────────────┘       │            │
│         │                                    │            │
│         ▼                                    │            │
│  x_recon [B, 3, 224, 224]                    │            │
│                                              │            │
└──────────────────────────────────────────────┘            │
         │                                                  │
         ▼                                                  │
┌─────── Loss 计算 ────────────────────────────────────────┐│
│                                                          ││
│  recon_loss = MSE(x_recon, x)                            ││
│                          +                               ││
│  vq_loss (来自 Quantizer)                                ││
│                          +                               ││
│  λ × perceptual_loss  ←─────────────────────────────────┘│
│                                                          │
│  total_loss = recon_loss + vq_loss + λ × perceptual_loss │
│                                                          │
└──────────────────────────────────────────────────────────┘
         │
         ▼
   Adam optimizer.step()
```

## 推理阶段 (`PatchCore._predict`)

```
输入图像 x [B, 3, 224, 224]
         │
         ▼
┌─── VQVAE Forward (eval, no_grad) ───┐
│                                      │
│  Encoder → z_e [B, D, H, W]         │
│       │                              │
│       ▼                              │
│  Quantizer → z_q [B, D, H, W]       │
│       │                              │
│       ▼                              │
│  Decoder (含 ResidualBlock)          │
│       │                              │
│       ▼                              │
│  x_recon [B, 3, 224, 224]           │
│                                      │
└──────────────────────────────────────┘
         │
    ┌────┴──────────────┐
    ▼                   ▼
┌─ 信号1: MAD ─┐   ┌─ 信号2: SSIM ──────────┐
│ (潜在空间)    │   │ (图像空间)              │
│              │   │                         │
│ mean(|z_e -  │   │ SSIM(x, x_recon)       │
│       z_q|)  │   │ 高斯加权结构相似度       │
│              │   │                         │
│ mad_map      │   │ ssim_map               │
│ [B, H, W]    │   │ [B, H, W]             │
└──────┬───────┘   └────────┬────────────────┘
       │                    │
       │  (双线性上采样对齐)  │
       ▼                    ▼
┌──────────────────────────────────────┐
│  融合: anomaly_map = MAD × (1-SSIM)  │
│  [B, H, W]                           │
│                                      │
│  两个信号都高 → 强异常响应            │
└──────────────────┬───────────────────┘
                   │
          ┌────────┴────────┐
          ▼                 ▼
  ┌─ 图像级评分 ─┐   ┌─ 像素级分割 ───────┐
  │ top-0.5% 像素 │   │ RescaleSegmentor   │
  │ 均值          │   │ 双线性上采样        │
  │              │   │ + 高斯模糊 (σ=4)   │
  │ score (标量) │   │                    │
  └──────────────┘   │ mask [H, W]       │
                     └────────────────────┘
```

## 模块职责

| 文件 | 职责 |
|---|---|
| `src/patchcore/vqvae_model.py` | `VQVAE`, `Encoder`, `Decoder`, `ResidualBlock`, 三种量化器 (`VectorQuantizer`, `EMAVectorQuantizer`, `ScalarQuantizer`) |
| `src/patchcore/patchcore.py` | `PatchCore` (训练/推理/保存/加载), `PerceptualLoss`, `PatchMaker`, `calculate_mad_map`, `calculate_ssim_map` |
| `src/patchcore/common.py` | `RescaleSegmentor` (上采样+高斯模糊), `NetworkFeatureAggregator` (hook 特征提取) |
| `src/patchcore/backbones.py` | 预训练骨干网络注册表 (ResNet, EfficientNet, ViT, DenseNet) |
| `src/patchcore/metrics.py` | 图像级和像素级 AUROC |
| `src/patchcore/datasets/` | `MVTecDataset`, `MPDDDataset`, `CarDDDataset` |
| `bin/run_patchcore.py` | 训练+评估入口, `click` 命令链 |
| `bin/load_and_evaluate_patchcore.py` | 加载已保存模型并评估 |

## 关键参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--vq_num_hiddens` | 128 | Encoder/Decoder 通道数 |
| `--vq_num_embeddings` | 1024 | 码本大小 K |
| `--vq_embedding_dim` | 128 | 码本向量维度 D |
| `--vq_num_residual_layers` | 0 | Decoder 残差块数量 (推荐 2) |
| `--vq_num_residual_hiddens` | 0 | 残差块隐藏通道数 (推荐 128) |
| `--vq_commitment_cost` | 0.25 | commitment loss 权重 β |
| `--perceptual_loss_weight` | 0.1 | VGG 感知损失权重 λ (设 0 禁用) |
| `--vq_use_ema_codebook` | False | 使用 EMA 更新码本 (推荐开启) |
| `--vq_lr` | 1e-4 | 学习率 |
| `--vq_epochs` | 10 | 训练轮数 |
