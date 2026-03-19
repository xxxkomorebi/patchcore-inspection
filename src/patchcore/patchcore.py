"""PatchCore and PatchCore detection methods."""
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import tqdm
import torch.utils.data

import patchcore
import patchcore.common
import patchcore.vqvae_model


class PerceptualLoss(nn.Module):
    """L1 perceptual loss using frozen VGG16 intermediate features."""

    # ImageNet normalisation expected by VGG
    _MEAN = [0.485, 0.456, 0.406]
    _STD = [0.229, 0.224, 0.225]

    def __init__(self, device="cpu"):
        super().__init__()
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT).features
        # Extract up to relu1_2 (idx 4), relu2_2 (idx 9), relu3_3 (idx 16)
        self.slices = nn.ModuleList([
            nn.Sequential(*list(vgg.children())[:4]),   # relu1_2
            nn.Sequential(*list(vgg.children())[4:9]),  # relu2_2
            nn.Sequential(*list(vgg.children())[9:16]), # relu3_3
        ])
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer(
            "mean", torch.tensor(self._MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(self._STD).view(1, 3, 1, 1)
        )
        self.to(device)

    def _normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, x, x_recon):
        x = self._normalize(x)
        x_recon = self._normalize(x_recon)
        loss = 0.0
        for s in self.slices:
            x = s(x)
            x_recon = s(x_recon)
            loss += F.l1_loss(x_recon, x)
        return loss

# MAD calculation: quantization error in latent space
def calculate_mad_map(z_e, z_q):
    """Quantization error map: mean(|z_e - z_q|, dim=C).

    Measures how far the encoder output strays from the nearest
    codebook vector at each spatial position.

    Args:
        z_e: [B, C, H, W] encoder output (pre-quantization).
        z_q: [B, C, H, W] quantized latent (post-quantization).

    Returns:
        mad_map: [B, H, W] per-pixel anomaly score.
    """
    mad_map = torch.abs(z_e - z_q).mean(dim=1)
    return mad_map


def _gaussian_kernel_2d(kernel_size=11, sigma=1.5, channels=1, device=None):
    """Create a 2D Gaussian kernel for SSIM computation."""
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = torch.outer(g, g)
    g = g / g.sum()
    return g.expand(channels, 1, kernel_size, kernel_size).contiguous()


def calculate_ssim_map(original_image, reconstructed_image, c1=0.01, c2=0.03,
                       kernel_size=11, sigma=1.5):
    """Compute per-pixel SM (Similarity Map) following the VAE-GRF paper.

    SM(x_i) = SSIM(p_i, q_i)
            = (2 mu_p mu_q + c1)(2 sigma_pq + c2)
              / ((mu_p^2 + mu_q^2 + c1)(sigma_p^2 + sigma_q^2 + c2))

    where c1 = 0.01, c2 = 0.03 as stated in the paper (NOT the squared
    constants used in the standard SSIM implementation).

    Args:
        original_image:      [B, C, H, W] tensor in [0, 1].
        reconstructed_image: [B, C, H, W] tensor in [0, 1].
        c1, c2: stabilisation constants (paper defaults).
        kernel_size: Gaussian window size.
        sigma: Gaussian window std.

    Returns:
        ssim_map: [B, H, W] tensor — per-pixel similarity.
    """
    device = original_image.device
    channels = original_image.shape[1]
    kernel = _gaussian_kernel_2d(kernel_size, sigma, channels, device)
    pad = kernel_size // 2

    # Means
    mu_p = F.conv2d(original_image, kernel, padding=pad, groups=channels)
    mu_q = F.conv2d(reconstructed_image, kernel, padding=pad, groups=channels)

    mu_p_sq = mu_p * mu_p
    mu_q_sq = mu_q * mu_q
    mu_pq = mu_p * mu_q

    # Variances / covariance
    sigma_p_sq = F.conv2d(original_image * original_image, kernel, padding=pad, groups=channels) - mu_p_sq
    sigma_q_sq = F.conv2d(reconstructed_image * reconstructed_image, kernel, padding=pad, groups=channels) - mu_q_sq
    sigma_pq = F.conv2d(original_image * reconstructed_image, kernel, padding=pad, groups=channels) - mu_pq

    # SSIM map per channel
    numerator = (2 * mu_pq + c1) * (2 * sigma_pq + c2)
    denominator = (mu_p_sq + mu_q_sq + c1) * (sigma_p_sq + sigma_q_sq + c2)
    ssim_map = numerator / denominator          # [B, C, H, W]

    # Average across channels → [B, H, W]
    ssim_map = ssim_map.mean(dim=1)
    return ssim_map

# 获取日志记录器实例
LOGGER = logging.getLogger(__name__)

class PatchCore(torch.nn.Module):
    # 构造函数，接受device参数
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        # 调用父类的构造函数，进行初始化
        super(PatchCore, self).__init__()
        self.device = device
    
    # 初始化和配置PatchCore模型的所有组件
    def load(
        self,
        device,
        input_shape,
        # VQ-VAE Parameters
        vq_in_channels,
        vq_out_channels,
        vq_num_hiddens,
        vq_num_residual_layers,
        vq_num_residual_hiddens,
        vq_num_embeddings,
        vq_embedding_dim,
        vq_commitment_cost,
        vq_use_ema_codebook=False,
        vq_ema_decay=0.99,
        vq_ema_eps=1e-5,
        vq_backbone_name=None,
        vq_layers_to_extract_from=None,
        vq_use_fsq=False,
        vq_fsq_levels=None,
        # PatchCore Parameters
        patchsize=3,
        patchstride=1,
        vq_lr=1e-4,
        vq_epochs=10,
        perceptual_loss_weight=0.0,
        **kwargs,
    ):
        self.device = device
        self.input_shape = input_shape
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)
        self.forward_modules = torch.nn.ModuleDict({})

        # 1. VQ-VAE Model (Replaces Backbone and Feature Aggregator)
        self.vqvae_model = patchcore.vqvae_model.VQVAE(
            in_channels=vq_in_channels,
            out_channels=vq_out_channels,
            num_hiddens=vq_num_hiddens,
            num_residual_layers=vq_num_residual_layers,
            num_residual_hiddens=vq_num_residual_hiddens,
            num_embeddings=vq_num_embeddings,
            embedding_dim=vq_embedding_dim,
            commitment_cost=vq_commitment_cost,
            use_ema_codebook=vq_use_ema_codebook,
            ema_decay=vq_ema_decay,
            ema_eps=vq_ema_eps,
            backbone_name=vq_backbone_name,
            layers_to_extract_from=vq_layers_to_extract_from,
            device=device,
            original_image_size=input_shape[-2:],
            use_fsq=vq_use_fsq,
            fsq_levels=vq_fsq_levels,
        ).to(device)
        self.forward_modules["vqvae_model"] = self.vqvae_model

        # 2. VQ-VAE Training Configuration
        self.vq_lr = vq_lr
        self.vq_epochs = vq_epochs
        self.vq_optimizer = torch.optim.Adam(self.vqvae_model.parameters(), lr=self.vq_lr)
        self.vq_loss_fn = F.mse_loss # Reconstruction loss

        # Perceptual loss (VGG-based)
        self.perceptual_loss_weight = perceptual_loss_weight
        if perceptual_loss_weight > 0:
            self.perceptual_loss_fn = PerceptualLoss(device=device)
        else:
            self.perceptual_loss_fn = None

        self.anomaly_segmentor = patchcore.common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

    # 从输入数据提取特征嵌入
    def embed(self, data):
        # 如果输入数据是DataLoader类型
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for item in data:
                image = item
                if isinstance(item, dict): # 如果输入是字典，提取图像张量和辅助数据
                    image = item["image"]
                        
                with torch.no_grad():   # 禁用梯度计算，节省内存和计算资源
                    # 将图像移动到指定设备并转换为浮点类型
                    input_image = image.to(torch.float).to(self.device)
                    # 调用私有方法 _embed 提取特征，并添加到特征列表中
                    # _embed returns quantized features (NumPy array)
                    quantized_features = self._embed(input_image)
                    features.append(quantized_features)
            return features
        
        # 如果输入不是DataLoader，直接调用 _embed 方法提取特征
        quantized_features = self._embed(data)
        return quantized_features
    
    # _embed 方法实现了从输入图像中提取特征嵌入的具体逻辑
    
    def _embed(self, images, detach=True, provide_patch_shapes=False):

        _ = self.vqvae_model.eval()

        # 现在 vqvae_model 额外返回 perplexity、量化距离图与 active code 比例
        x_recon, vq_loss, perplexity, z_e, quantized_features, encoding_indices, quantization_error, active_code_ratio = self.vqvae_model(images)


        return x_recon, vq_loss, perplexity, z_e, quantized_features, encoding_indices, quantization_error, active_code_ratio

    def fit(self, training_data):
        # Train VQ-VAE
        self._train_vqvae(training_data)
        # Calibrate normalization using normal training samples
        self._calibrate_normalization(training_data)

    def _train_vqvae(self, input_data):
        """Trains the VQ-VAE model using reconstruction and VQ losses."""
        self.vqvae_model.train()
        
        self.perplexity_history = []  # 记录每个 epoch 的平均 perplexity，便于后续画曲线

        for epoch in range(self.vq_epochs):
            total_loss = 0
            perplexities = []
            active_ratios = []
            with tqdm.tqdm(
                input_data, desc=f"VQ-VAE Training Epoch {epoch+1}/{self.vq_epochs}", leave=False
            ) as data_iterator:
                for item in data_iterator:
                    image = item
                    if isinstance(item, dict):
                        image = item["image"]
                    elif isinstance(item, (list, tuple)):
                        image = item[0]

                    input_image = image.to(torch.float).to(self.device)
                    
                    # Forward pass
                    x_recon, vq_loss, perplexity, _, _, _, _, active_code_ratio = self.vqvae_model(input_image)
                    
                    # Calculate Reconstruction Loss (MSE)
                    recon_loss = self.vq_loss_fn(x_recon, input_image)

                    # Total Loss
                    loss = recon_loss + vq_loss

                    # Perceptual loss
                    if self.perceptual_loss_fn is not None:
                        loss = loss + self.perceptual_loss_weight * self.perceptual_loss_fn(input_image, x_recon)
                    
                    # Backward pass and optimization
                    self.vq_optimizer.zero_grad()
                    loss.backward()
                    self.vq_optimizer.step()
                    
                    total_loss += loss.item()
                    perplexities.append(perplexity.item())
                    active_ratios.append(active_code_ratio.item())

            avg_perplexity = sum(perplexities) / max(len(perplexities), 1)
            avg_active_ratio = sum(active_ratios) / max(len(active_ratios), 1)
            self.perplexity_history.append(avg_perplexity)
            LOGGER.info(
                f"VQ-VAE Epoch {epoch+1} finished. "
                f"Avg Loss: {total_loss / len(input_data):.4f}; "
                f"Avg Perplexity: {avg_perplexity:.3f}; "
                f"Active Code Ratio: {avg_active_ratio:.3f}"
            )

    def _calibrate_normalization(self, training_data):
        """Compute per-pixel mean and std of anomaly maps on normal samples."""
        LOGGER.info("Calibrating anomaly map normalization on training data...")
        self.vqvae_model.eval()

        # Welford online algorithm for per-pixel mean/std
        count = 0
        mean = None
        m2 = None

        with torch.no_grad():
            for item in tqdm.tqdm(training_data, desc="Calibrating normalization", leave=False):
                image = item
                if isinstance(item, dict):
                    image = item["image"]
                elif isinstance(item, (list, tuple)):
                    image = item[0]

                input_images = image.to(torch.float).to(self.device)
                x_recon, _, _, z_e, quantized_features, _, _, _ = self._embed(image)

                mad_map = calculate_mad_map(z_e, quantized_features)
                sm_map = calculate_ssim_map(input_images, x_recon)

                if mad_map.shape[-2:] != sm_map.shape[-2:]:
                    mad_map = F.interpolate(
                        mad_map.unsqueeze(1), size=sm_map.shape[-2:],
                        mode='bilinear', align_corners=False
                    ).squeeze(1)

                anomaly_maps = mad_map * (1 - sm_map)  # [B, H, W]

                # Welford update per sample in batch
                for i in range(anomaly_maps.shape[0]):
                    amap = anomaly_maps[i]  # [H, W]
                    count += 1
                    if mean is None:
                        mean = torch.zeros_like(amap)
                        m2 = torch.zeros_like(amap)
                    delta = amap - mean
                    mean = mean + delta / count
                    delta2 = amap - mean
                    m2 = m2 + delta * delta2

        if count > 1:
            std = torch.sqrt(m2 / (count - 1))
        else:
            std = torch.ones_like(mean)

        self.norm_mean = mean.cpu()  # [H, W]
        self.norm_std = std.cpu()    # [H, W]
        LOGGER.info("Normalization calibration done (n=%d samples).", count)

    # patchcore的推理入口
    def predict(self, data, return_recon=False):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data, return_recon=return_recon)

        return self._predict(data, return_recon=return_recon)
    
    # 对整个数据加载器中的图像进行异常检测
    def _predict_dataloader(self, dataloader, return_recon=False):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []         # 异常分数
        masks = []          # 异常掩码
        labels_gt = []      # 真实标签
        masks_gt = []       # 真实掩码
        reconstructions = []
        inputs_raw = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    # 提取真实标签和掩码
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    masks_gt.extend(image["mask"].numpy().tolist())
                    image = image["image"]

                if return_recon:
                    _scores, _masks, _recons, _inputs = self._predict(image, return_recon=return_recon)
                else:
                    _scores, _masks = self._predict(image, return_recon=return_recon)

                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)
                if return_recon: # Corrected from 'if return_recon):'
                    reconstructions.extend(_recons)
                    inputs_raw.extend(_inputs)
        if return_recon:
            return scores, masks, labels_gt, masks_gt, reconstructions, inputs_raw
        return scores, masks, labels_gt, masks_gt

    def _predict(self, images, return_recon=False):
        # 1. 准备数据
        input_images = images.to(torch.float).to(self.device)
        self.vqvae_model.eval()
        
        with torch.no_grad():
            # 2. 获取重建图
            x_recon, _, _, z_e, quantized_features, _, _, _ = self._embed(images)

            # 1. MAD map (latent space): quantization error |z_e - z_q|
            mad_map = calculate_mad_map(z_e, quantized_features)  # [B, H_lat, W_lat]

            # 2. SM map (image space): per-pixel SSIM
            sm_map = calculate_ssim_map(input_images, x_recon)  # [B, H, W]

            # Upsample MAD to image resolution if needed
            if mad_map.shape[-2:] != sm_map.shape[-2:]:
                mad_map = F.interpolate(
                    mad_map.unsqueeze(1), size=sm_map.shape[-2:],
                    mode='bilinear', align_corners=False
                ).squeeze(1)

            # 3. Combined anomaly map: MAD × (1 - SSIM)
            #    MAD high = large quantization error (latent anomaly)
            #    (1 - SSIM) high = poor reconstruction (image anomaly)
            anomaly_maps = mad_map * (1 - sm_map)

            # Adaptive normalization: z-score using training set statistics
            if hasattr(self, 'norm_mean') and self.norm_mean is not None:
                norm_mean = self.norm_mean.to(anomaly_maps.device)
                norm_std = self.norm_std.to(anomaly_maps.device)
                anomaly_maps = (anomaly_maps - norm_mean) / (norm_std + 1e-6)
                anomaly_maps = torch.clamp(anomaly_maps, min=0)

            # ---- Image-level scoring: top-0.5% pixel mean ----
            flat_scores = anomaly_maps.flatten(1)
            k = max(1, int(flat_scores.shape[1] * 0.005))
            image_scores = torch.topk(flat_scores, k, dim=1).values.mean(dim=1)

            anomaly_maps_np = anomaly_maps.cpu().numpy()
            masks = self.anomaly_segmentor.convert_to_segmentation(anomaly_maps_np)

        if return_recon:
            return (
                [score.item() for score in image_scores],
                [mask for mask in masks],
                [x.cpu() for x in x_recon],
                [inp.cpu() for inp in input_images],
            )

        return [score.item() for score in image_scores], [mask for mask in masks]
    
    # 生成保存或加载模型参数的文件路径
    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    # 保存PatchCore模型及其参数到指定路径
    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        LOGGER.info("saving VQ-VAE data.")
        
        # 保存 VQ-VAE 模型参数
        vqvae_path = os.path.join(save_path, prepend + "vqvae_model.pth")
        torch.save(self.vqvae_model.state_dict(), vqvae_path)
        
        # 创建字典，存储VQ-VAE的关键参数
        patchcore_params = {
            "input_shape": self.input_shape,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            # VQ-VAE parameters for reconstruction
            "vq_in_channels": self.vqvae_model.encoder.in_channels,
            "vq_out_channels": self.vqvae_model.decoder.out_channels,
            "vq_num_hiddens": self.vqvae_model.encoder.num_hiddens,
            "vq_num_residual_layers": 0, # Simplified VQ-VAE doesn't use this yet
            "vq_num_residual_hiddens": 0, # Simplified VQ-VAE doesn't use this yet
            "vq_num_embeddings": self.vqvae_model.quantizer.num_embeddings,
            "vq_embedding_dim": self.vqvae_model.quantizer.embedding_dim,
            "vq_commitment_cost": self.vqvae_model.quantizer.commitment_cost,
            "vq_backbone_name": self.vqvae_model.encoder.backbone_name,
            "vq_layers_to_extract_from": self.vqvae_model.encoder.layers_to_extract_from,
            "vq_use_fsq": getattr(self.vqvae_model, 'use_fsq', False),
            "vq_fsq_levels": getattr(self.vqvae_model.quantizer, 'levels', None) if getattr(self.vqvae_model, 'use_fsq', False) else None,
            "vq_lr": self.vq_lr,
            "vq_epochs": self.vq_epochs,
            "perceptual_loss_weight": self.perceptual_loss_weight,
        }
        # Save normalization statistics if available
        if hasattr(self, 'norm_mean') and self.norm_mean is not None:
            patchcore_params["norm_mean"] = self.norm_mean.numpy()
            patchcore_params["norm_std"] = self.norm_std.numpy()
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)

    # 从指定路径加载PatchCore模型及其参数
    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        prepend: str = "",
    ) -> None:
        LOGGER.info("Loading and initializing VQ-VAE Anomaly Detection")
        # 打开参数文件，使用pickle反序列化参数字典
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        
        # Extract normalization stats before passing to load()
        norm_mean_np = patchcore_params.pop("norm_mean", None)
        norm_std_np = patchcore_params.pop("norm_std", None)

        # Load VQ-VAE parameters and initialize model
        self.load(**patchcore_params, device=device)

        # Load VQ-VAE state dict
        vqvae_path = os.path.join(load_path, prepend + "vqvae_model.pth")
        self.vqvae_model.load_state_dict(torch.load(vqvae_path, map_location=device))

        # Restore normalization statistics if saved
        if norm_mean_np is not None:
            self.norm_mean = torch.from_numpy(norm_mean_np)
            self.norm_std = torch.from_numpy(norm_std_np)
        else:
            self.norm_mean = None
            self.norm_std = None


# Image handling classes.
# 图像处理类
class PatchMaker:
    # 接收 patchsize 和 stride 步长
    def __init__(self, patchsize, stride=1):
        self.patchsize = patchsize
        self.stride = stride

    # 将输入的特征图划分为补丁
    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        # 计算填充量
        padding = int((self.patchsize - 1) / 2)
        # 实例化unfold模块，从输入张量中提取重叠的局部区域（补丁），并将其展平
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))

        # 将展平的特征重新整形为补丁形式[batchsize, channels, patchsize*patchsize, N_patches]
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x
