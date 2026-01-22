"""PatchCore and PatchCore detection methods."""
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm # 进度条
import torch.utils.data

import patchcore
import patchcore.common
import patchcore.vqvae_model

# 获取日志记录器实例
LOGGER = logging.getLogger(__name__)

class PatchCore(torch.nn.Module):
    # 构造函数，接受device参数
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        # 调用父类的构造函数，进行初始化
        super(PatchCore, self).__init__()
        # 将device参数赋值给实例变量
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
        # PatchCore Parameters
        patchsize=3,
        patchstride=1,
        vq_lr=1e-4,
        vq_epochs=10,
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
            original_image_size=input_shape[-2:]
        ).to(device)
        self.forward_modules["vqvae_model"] = self.vqvae_model

        # 2. VQ-VAE Training Configuration
        self.vq_lr = vq_lr
        self.vq_epochs = vq_epochs
        self.vq_optimizer = torch.optim.Adam(self.vqvae_model.parameters(), lr=self.vq_lr)
        self.vq_loss_fn = F.mse_loss # Reconstruction loss

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
    '''形状为 [Batch_size, Channels, Height, Width]'''
    def _embed(self, images, detach=True, provide_patch_shapes=False):

        _ = self.vqvae_model.eval()

        # 现在 vqvae_model 额外返回 perplexity、量化距离图与 active code 比例
        x_recon, vq_loss, perplexity, quantized_features, encoding_indices, quantization_error, active_code_ratio = self.vqvae_model(images)

        return x_recon, vq_loss, perplexity, quantized_features, encoding_indices, quantization_error, active_code_ratio

    def fit(self, training_data):
        # Train VQ-VAE
        self._train_vqvae(training_data)

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
                    x_recon, vq_loss, perplexity, _, _, _, active_code_ratio = self.vqvae_model(input_image)
                    
                    # Calculate Reconstruction Loss (MSE)
                    recon_loss = self.vq_loss_fn(x_recon, input_image)
                    
                    # Total Loss
                    loss = recon_loss + vq_loss
                    
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
                if return_recon:
                    reconstructions.extend(_recons)
                    inputs_raw.extend(_inputs)
        if return_recon:
            return scores, masks, labels_gt, masks_gt, reconstructions, inputs_raw
        return scores, masks, labels_gt, masks_gt

    '''这后面的我看不懂'''
    def _predict(self, images, return_recon=False):
        # 1. 准备数据
        input_images = images.to(torch.float).to(self.device)
        self.vqvae_model.eval()
        
        with torch.no_grad():
            # 2. 获取重建图
            x_recon, _, _, _, _, quantization_error, _ = self._embed(images)

            # ==========================================
            # === 核心修改：使用 梯度损失 + L1 损失 ===
            # ==========================================

            # --- A. 像素级误差 (改用 L1，比 MSE 对光照更鲁棒) ---
            # reduction='none' 也就是保留 [B, C, H, W]
            # mean(dim=1) 把通道平均掉 -> [B, H, W]
            pixel_loss = torch.mean(torch.abs(x_recon - input_images), dim=1)

            # --- B. 梯度损失 (Gradient Loss) ---
            # 定义计算梯度的函数 (计算相邻像素的差值，即边缘)
            def compute_gradient(img):
                # 沿 X 轴差分
                gx = img[:, :, :, :-1] - img[:, :, :, 1:]
                # 沿 Y 轴差分
                gy = img[:, :, :-1, :] - img[:, :, 1:, :]
                # 补齐尺寸 (Padding)，保持和原图一样大
                gx = F.pad(gx, (0, 1, 0, 0), mode='replicate')
                gy = F.pad(gy, (0, 0, 0, 1), mode='replicate')
                return gx, gy

            # 计算原图和重建图的梯度
            gx_in, gy_in = compute_gradient(input_images)
            gx_rec, gy_rec = compute_gradient(x_recon)

            # 计算梯度差异 (边缘对不上的程度)
            grad_error_x = torch.mean(torch.abs(gx_in - gx_rec), dim=1)
            grad_error_y = torch.mean(torch.abs(gy_in - gy_rec), dim=1)
            grad_loss = grad_error_x + grad_error_y

            # --- C. 融合异常图 ---
            # 梯度误差通常数值很小，给它加权 (例如 5.0 倍)
            # 这样模型主要关注“边缘有没有对上”，而不是“亮度有没有对上”
            anomaly_maps = pixel_loss + 5.0 * grad_loss
            
            # ==========================================

            # 后续处理逻辑保持不变
            flat_scores = anomaly_maps.flatten(1)
            # 使用 Top-K 平均值作为图片级分数
            k = max(1, int(flat_scores.shape[1] * 0.001))
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
    
    #def _predict(self, images, return_recon=False):
        
    #     input_images = images.to(torch.float).to(self.device)
    #     self.vqvae_model.eval()
    #     with torch.no_grad():
    #         x_recon,_,_,_,_, quantization_error, _ = self._embed(images)

    #         error_maps = F.mse_loss(x_recon, input_images, reduction="none")
    #         anomaly_maps = torch.mean(error_maps, dim=1) # 恢复仅使用重建误差

    #         # 三种聚合方式计算图像级异常分数
    #         # 原始最大值聚合（对噪声敏感）
    #         # image_scores = anomaly_maps.amax(dim=(1,2))
    #         # 分位数聚合（更稳健）
    #         # image_scores = torch.quantile(anomaly_maps.flatten(1), 0.999, dim=1)
    #         # Top-k 均值聚合
    #         flat_scores = anomaly_maps.flatten(1)
    #         k = max(1, int(flat_scores.shape[1] * 0.001))  # top 0.1%
    #         image_scores = torch.topk(flat_scores, k, dim=1).values.mean(dim=1)
    #         anomaly_maps_np = anomaly_maps.cpu().numpy()
    #         masks = self.anomaly_segmentor.convert_to_segmentation(anomaly_maps_np)

    #     if return_recon:
    #         return (
    #             [score.item() for score in image_scores],
    #             [mask for mask in masks],
    #             [x.cpu() for x in x_recon],
    #             [inp.cpu() for inp in input_images],
    #         )

    #     return [score.item()for score in image_scores], [mask for mask in masks]

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
            "vq_lr": self.vq_lr,
            "vq_epochs": self.vq_epochs,
        }
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
        
        # Load VQ-VAE parameters and initialize model
        self.load(**patchcore_params, device=device)
        
        # Load VQ-VAE state dict
        vqvae_path = os.path.join(load_path, prepend + "vqvae_model.pth")
        self.vqvae_model.load_state_dict(torch.load(vqvae_path, map_location=device))


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
            x = x.cpu.numpy()
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
