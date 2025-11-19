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
import patchcore.backbones
import patchcore.common
import patchcore.sampler
import patchcore.vqvae_model

# 获取日志记录器实例
LOGGER = logging.getLogger(__name__)

# 定义PatchCore类，继承module，是一个PyTorch模型
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
        vq_aux_dim=0, # New parameter for multi-modal support
        # PatchCore Parameters
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=patchcore.sampler.IdentitySampler(),
        nn_method=patchcore.common.FaissNN(False, 4),
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
            aux_dim=vq_aux_dim, # Pass aux_dim to VQVAE
        ).to(device)
        self.forward_modules["vqvae_model"] = self.vqvae_model

        # 2. VQ-VAE Training Configuration
        self.vq_lr = vq_lr
        self.vq_epochs = vq_epochs
        self.vq_optimizer = torch.optim.Adam(self.vqvae_model.parameters(), lr=self.vq_lr)
        self.vq_loss_fn = F.mse_loss # Reconstruction loss

        # 3. PatchCore Components (Simplified)
        # The VQ-VAE output (quantized feature map) is the new feature source.
        # We only need Aggregator and Scorer/Segmentor.

        # The feature dimension is now VQ_EMBEDDING_DIM
        feature_dimensions = [vq_embedding_dim]
        
        # We use VQ_EMBEDDING_DIM as the target dimension for the memory bank.
        self.target_embed_dimension = vq_embedding_dim
        
        # We use a simplified Aggregator that just flattens the patches.
        preadapt_aggregator = patchcore.common.Aggregator(
            target_dim=vq_embedding_dim # Target dim is the embedding dim
        )
        _ = preadapt_aggregator.to(self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator
        
        # 4. Anomaly Scorer and Segmentor
        self.anomaly_scorer = patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )
        self.anomaly_segmentor = patchcore.common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )
        self.featuresampler = featuresampler

    # 从输入数据提取特征嵌入
    def embed(self, data):
        # 如果输入数据是DataLoader类型
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for item in data:
                image = item
                aux_data = None
                if isinstance(item, dict): # 如果输入是字典，提取图像张量和辅助数据
                    image = item["image"]
                    aux_data = item.get("aux_data")
                    if aux_data is not None:
                        aux_data = aux_data.to(torch.float).to(self.device)
                        
                with torch.no_grad():   # 禁用梯度计算，节省内存和计算资源
                    # 将图像移动到指定设备并转换为浮点类型
                    input_image = image.to(torch.float).to(self.device)
                    # 调用私有方法 _embed 提取特征，并添加到特征列表中
                    # _embed returns quantized features (NumPy array)
                    quantized_features = self._embed(input_image, aux_data)
                    features.append(quantized_features)
            return features
        
        # 如果输入不是DataLoader，直接调用 _embed 方法提取特征
        # 假设直接输入只包含图像，或者用户在调用时提供了 aux_data
        quantized_features = self._embed(data)
        return quantized_features
    
    # _embed 方法实现了从输入图像中提取特征嵌入的具体逻辑
    '''形状为 [Batch_size, Channels, Height, Width]'''
    def _embed(self, images, aux_data=None, detach=True, provide_patch_shapes=False):
        """
        Returns VQ-VAE outputs: quantized features, reconstruction, vq_loss, encoding_indices.
        The quantized features are used as PatchCore features.
        """

        def _detach(features):
            if detach:
                # features is a list containing one aggregated feature tensor.
                # We return the single NumPy array.
                return features[0].detach().cpu().numpy()
            return features[0]
        
        # VQ-VAE is trained, so we set it to eval mode for feature extraction
        _ = self.vqvae_model.eval()
        
        # 1. VQ-VAE Forward Pass
        x_recon, vq_loss, quantized_features, encoding_indices = self.vqvae_model(images, aux_data)
        
        # 2. Patchify the Quantized Features
        # VQ-VAE output is [B, D, H/4, W/4]. We treat this as the feature map.
        
        # Patchify returns [N_total_patches, D, P, P] and spatial info [H_patches, W_patches]
        features_and_shapes = self.patch_maker.patchify(
            quantized_features, return_spatial_info=True
        )
        
        features = [features_and_shapes[0]]
        patch_shapes = [features_and_shapes[1]]
        
        # 3. Aggregate Patches (Flattening)
        # features is a list containing one element: [N_total_patches, D, P, P]
        # The Aggregator flattens the patch dimensions (P, P) and aggregates the list.
        features = self.forward_modules["preadapt_aggregator"](features)
        
        # features is now [N_total_patches, D * P * P]
        
        # We return the quantized features (for memory bank) and the VQ-VAE outputs (for training/loss)
        
        if provide_patch_shapes:
            # We detach the features for memory bank storage (NumPy array)
            return _detach(features), patch_shapes, x_recon, vq_loss, encoding_indices
        
        # If not providing shapes, we return the features for memory bank construction
        return _detach(features)

    # memory bank的训练入口
    def fit(self, training_data):
        """
        VQ-VAE-PatchCore training.
        1. Train VQ-VAE model (Encoder, Quantizer, Decoder) using reconstruction loss.
        2. Compute quantized embeddings of the training data and fill the memory bank.
        """
        # 1. Train VQ-VAE
        self._train_vqvae(training_data)
        
        # 2. Build Memory Bank
        self._fill_memory_bank(training_data)

    def _train_vqvae(self, input_data):
        """Trains the VQ-VAE model using reconstruction and VQ losses."""
        self.vqvae_model.train()
        
        for epoch in range(self.vq_epochs):
            total_loss = 0
            with tqdm.tqdm(
                input_data, desc=f"VQ-VAE Training Epoch {epoch+1}/{self.vq_epochs}", leave=False
            ) as data_iterator:
                for item in data_iterator:
                    image = item
                    aux_data = None
                    if isinstance(item, dict):
                        image = item["image"]
                        aux_data = item.get("aux_data")
                        if aux_data is not None:
                            aux_data = aux_data.to(torch.float).to(self.device)
                    
                    input_image = image.to(torch.float).to(self.device)
                    
                    # Forward pass
                    x_recon, vq_loss, _, _ = self.vqvae_model(input_image, aux_data)
                    
                    # Calculate Reconstruction Loss (MSE)
                    recon_loss = self.vq_loss_fn(x_recon, input_image)
                    
                    # Total Loss
                    loss = recon_loss + vq_loss
                    
                    # Backward pass and optimization
                    self.vq_optimizer.zero_grad()
                    loss.backward()
                    self.vq_optimizer.step()
                    
                    total_loss += loss.item()
            
            LOGGER.info(f"VQ-VAE Epoch {epoch+1} finished. Avg Loss: {total_loss / len(input_data):.4f}")


    def _fill_memory_bank(self, input_data):
        """Computes and sets the support features for PatchCore memory bank."""
        
        # Set VQ-VAE to evaluation mode for feature extraction
        self.vqvae_model.eval()

        def _image_to_features(input_image):
            # _embed returns quantized features (NumPy array) when provide_patch_shapes=False
            quantized_features = self._embed(input_image)
            return quantized_features

        features = []
        with torch.no_grad():
            with tqdm.tqdm(
                input_data, desc="Computing support features...", position=1, leave=False
            ) as data_iterator:
                for image in data_iterator:
                    if isinstance(image, dict):
                        image = image["image"]
                    
                    features.append(_image_to_features(image))
        
        # 沿着第0维连接所有图像的特征
        features = np.concatenate(features, axis=0)
        # 使用特征采样器对特征进行采样，减少记忆库大小
        features = self.featuresampler.run(features)

        # 确保 features 是 NumPy 数组，以满足 anomaly_scorer.fit 的类型要求
        if not isinstance(features, np.ndarray):
            features = features.cpu().numpy()

        # 使用采样后的特征来拟合异常评分器
        self.anomaly_scorer.fit(detection_features=[features])
    
    # patchcore的推理入口
    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)
    
    # 对整个数据加载器中的图像进行异常检测
    def _predict_dataloader(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []         # 异常分数
        masks = []          # 异常掩码
        labels_gt = []      # 真实标签
        masks_gt = []       # 真实掩码
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    # 提取真实标签和掩码
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    masks_gt.extend(image["mask"].numpy().tolist())
                    image = image["image"]
                # 获取图像的异常分数和掩码
                _scores, _masks = self._predict(image)

                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)
        return scores, masks, labels_gt, masks_gt

    '''这后面的我看不懂'''
    def _predict(self, images, aux_data=None):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        self.vqvae_model.eval() # Ensure VQ-VAE is in eval mode

        batchsize = images.shape[0]
        with torch.no_grad():
            # _embed returns quantized features (for memory bank), patch_shapes, x_recon, vq_loss, encoding_indices
            features, patch_shapes, x_recon, vq_loss, encoding_indices = self._embed(
                images, aux_data=aux_data, provide_patch_shapes=True
            )
            # 将提取的特征转换为numpy数组
            features = np.asarray(features)

            # 使用anomaly_scorer对提取的补丁特征进行预测，计算每个补丁的异常分数
            # [0] 表示只取第一个元素，即补丁分数
            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]

            # 将补丁分数重新组合成图像级别的异常分数和掩码
            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            # 将补丁分数重塑为[B,H,W]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        return [score for score in image_scores], [mask for mask in masks]

    # 生成保存或加载模型参数的文件路径
    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    # 保存PatchCore模型及其参数到指定路径
    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        # 保存异常评分器的内部状态，不保存特征
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        
        # 保存 VQ-VAE 模型参数
        vqvae_path = os.path.join(save_path, prepend + "vqvae_model.pth")
        torch.save(self.vqvae_model.state_dict(), vqvae_path)
        
        # 创建字典，存储PatchCore的关键参数
        patchcore_params = {
            "input_shape": self.input_shape,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
            # VQ-VAE parameters for reconstruction
            "vq_in_channels": self.vqvae_model.encoder.in_channels,
            "vq_out_channels": self.vqvae_model.decoder.out_channels,
            "vq_num_hiddens": self.vqvae_model.encoder.num_hiddens,
            "vq_num_residual_layers": 0, # Simplified VQ-VAE doesn't use this yet
            "vq_num_residual_hiddens": 0, # Simplified VQ-VAE doesn't use this yet
            "vq_num_embeddings": self.vqvae_model.quantizer.num_embeddings,
            "vq_embedding_dim": self.vqvae_model.quantizer.embedding_dim,
            "vq_commitment_cost": self.vqvae_model.quantizer.commitment_cost,
            "vq_aux_dim": self.vqvae_model.encoder.aux_dim, # Save aux_dim
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
        nn_method,
        prepend: str = "",
    ) -> None:
        LOGGER.info("Loading and initializing PatchCore.")
        # 打开参数文件，使用pickle反序列化参数字典
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        
        # Load VQ-VAE parameters and initialize model
        self.load(**patchcore_params, device=device, nn_method=nn_method)
        
        # Load VQ-VAE state dict
        vqvae_path = os.path.join(load_path, prepend + "vqvae_model.pth")
        self.vqvae_model.load_state_dict(torch.load(vqvae_path, map_location=device))

        self.anomaly_scorer.load(load_path, prepend)


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
