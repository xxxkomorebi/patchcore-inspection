import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import load as load_backbone
from .common import NetworkFeatureAggregator

class VectorQuantizer(nn.Module):
    """
    Vector Quantizer (Codebook) module.
    Maps continuous latent vectors to discrete codebook vectors.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.num_embeddings = num_embeddings  # K
        self.embedding_dim = embedding_dim    # D
        self.commitment_cost = commitment_cost # beta

        # Initialize the embeddings (Codebook)
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, inputs):
        # inputs: [B, C, H, W] -> [B, H, W, C]
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        # Flatten input: [B*H*W, C]
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate distances: ||z_e - e_i||^2 = ||z_e||^2 + ||e_i||^2 - 2 * z_e * e_i
        # [B*H*W, K]
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                     + torch.sum(self.embedding.weight.to(flat_input.device)**2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.to(flat_input.device).t()))

        # Find the closest codebook vector index
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        # Convert indices to one-hot vectors
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize the input vector (lookup the codebook vector)
        # [B*H*W, C]
        quantized = torch.matmul(encodings, self.embedding.weight).to(inputs.device).view(input_shape)

        # Compute VQ loss components
        # 1. Codebook loss (move the codebook vector towards the encoder output)
        # This part is handled by the optimizer step on the embedding weights.
        # The loss is calculated as ||sg[z_e] - e||^2
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        
        # 2. Commitment loss (move the encoder output towards the codebook vector)
        # The loss is calculated as ||z_e - sg[e]||^2
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        
        # Total VQ loss
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-Through Estimator: 
        # During backward pass, the gradient of the quantized output is set to be 
        # the gradient of the encoder output.
        quantized = inputs + (quantized - inputs).detach()

        # Perplexity (optional metric for codebook usage)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Reshape back to [B, C, H, W]
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        # Active code ratio (optional diagnostic)
        active_code_ratio = (encodings.sum(dim=0) > 0).float().mean()

        return quantized, loss, perplexity, encoding_indices.view(input_shape[:-1]), active_code_ratio

class EMAVectorQuantizer(nn.Module):
    """
    EMA 版向量量化器，使用移动平均更新码本，避免梯度直接作用于嵌入导致塌缩。
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99, eps=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        embed = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embedding", embed)
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_embedding", embed.clone())

    def forward(self, inputs):
        # [B, C, H, W] -> [B*H*W, C]
        inputs_perm = inputs.permute(0, 2, 3, 1).contiguous()
        flat_input = inputs_perm.view(-1, self.embedding_dim)

        # 距离与索引
        # distances的格式为[N,K], N为输入向量数量，K为码本大小
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            - 2 * flat_input @ self.embedding.t()
            + torch.sum(self.embedding ** 2, dim=1)
        )
        # 对每个输入向量，找到最近的码本索引，得到长度为N的索引
        encoding_indices = torch.argmin(distances, dim=1)
        # 将索引转换为独热编码，格式为[N,K]
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat_input.dtype)

        # 量化
        quantized = encodings @ self.embedding
        quantized = quantized.view(inputs_perm.shape).permute(0, 3, 1, 2).contiguous()

        # EMA 更新码本（无梯度）
        with torch.no_grad():
            cluster_size = encodings.sum(0)
            self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)

            embedding_sum = encodings.t() @ flat_input
            self.ema_embedding.mul_(self.decay).add_(embedding_sum, alpha=1 - self.decay)

            n = torch.sum(self.ema_cluster_size)
            cluster_size = (
                (self.ema_cluster_size + self.eps)
                / (n + self.num_embeddings * self.eps)
                * n
            )
            normalized_embed = self.ema_embedding / cluster_size.unsqueeze(1)
            self.embedding.copy_(normalized_embed)

        # 损失：只对 Encoder 加 commitment，码本由 EMA 更新
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss

        # Straight-Through
        quantized = inputs + (quantized - inputs).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Active code ratio (optional diagnostic)
        active_code_ratio = (encodings.sum(dim=0) > 0).float().mean()

        return quantized, loss, perplexity, encoding_indices.view(inputs.shape[0], *inputs.shape[2:]), active_code_ratio

class Encoder(nn.Module):
    """
    VQ-VAE的Encoder模块. 支持使用自定义卷积网络或预训练骨干网络作为特征提取器。
    当使用预训练骨干网络时，它提取指定层的特征图。
    """
    def __init__(self, in_channels, num_hiddens, backbone_name=None, layers_to_extract_from=None, device='cpu'):
        super().__init__()
        self.in_channels = in_channels
        self.num_hiddens = num_hiddens
        self.backbone_name = backbone_name
        self.layers_to_extract_from = layers_to_extract_from
        self.device = device

        if self.backbone_name:
            # 使用预训练骨干网络
            backbone = load_backbone(backbone_name)
            self.feature_extractor = NetworkFeatureAggregator(
                backbone, layers_to_extract_from, device
            )
            self.out_channels = self.feature_extractor.feature_dimensions([in_channels, 256, 256])[-1]
            self.projection = nn.Identity()
            # 这里的self.out_channels将是backbone输出的通道数，VQVAE的num_hiddens需要与此匹配
        else:
            # 使用卷积网络
            self.conv_in = nn.Conv2d(in_channels, num_hiddens // 2, kernel_size=4, stride=2, padding=1)
            self.conv_mid = nn.Conv2d(num_hiddens // 2, num_hiddens, kernel_size=4, stride=2, padding=1)
            self.conv_out = nn.Conv2d(num_hiddens, num_hiddens, kernel_size=3, stride=1, padding=1)
            self.out_channels = num_hiddens

    def forward(self, x):
        # 始终与模块当前所在设备对齐，避免 self.device 过期导致 CPU/GPU 混用
        device = next(self.parameters()).device
        x = x.to(device)
        if self.backbone_name:
            features = self.feature_extractor(x)
            # 提取最后一个指定层的特征
            # 由于VQ-VAE的encoder期望一个单一的特征图输出，我们取最后一个提取的层
            h = features[self.layers_to_extract_from[-1]]
            h = self.projection(h)
            return h
        else:
            h = F.relu(self.conv_in(x))
            h = F.relu(self.conv_mid(h))
            return self.conv_out(h)

class Decoder(nn.Module):
    """
    Simple Decoder for VQ-VAE. Reconstructs image from feature map.
    """
    def __init__(self, out_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        self.out_channels = out_channels
        self.num_hiddens = num_hiddens

        # Initial feature mixing
        self.conv_in = nn.Conv2d(num_hiddens, num_hiddens, kernel_size=3, stride=1, padding=1)

        # PixelShuffle 上采样链，总放大 4x（适配 /4 下采样特征）
        self.ps1 = nn.Sequential(
            nn.Conv2d(num_hiddens, num_hiddens * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )
        self.ps2 = nn.Sequential(
            nn.Conv2d(num_hiddens, (num_hiddens // 2) * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )
        self.conv_out = nn.Conv2d(num_hiddens // 2, out_channels, kernel_size=3, stride=1, padding=1)
        self.output_act = nn.Identity()

    def forward(self, x):
        h = F.relu(self.conv_in(x))
        h = self.ps1(h)
        h = self.ps2(h)
        out = self.conv_out(h)
        return self.output_act(out) # Output reconstruction [B, out_channels, H, W]

class VQVAE(nn.Module):

    def __init__(self, in_channels, out_channels, num_hiddens, num_residual_layers, num_residual_hiddens,
                 num_embeddings, embedding_dim, commitment_cost, backbone_name=None, layers_to_extract_from=None, device='cpu', original_image_size=(224, 224),
                 use_ema_codebook=False, ema_decay=0.99, ema_eps=1e-5):
        super().__init__()
        self.device = device
        self.original_image_size = original_image_size

        # Encoder支持自定义卷积网络或预训练骨干网络
        if backbone_name:
            # 对于预训练模型，in_channels通常为3
            self.encoder = Encoder(
            in_channels,
                num_hiddens, # num_hiddens现在应该与backbone的输出通道数匹配
                backbone_name=backbone_name,
                layers_to_extract_from=layers_to_extract_from,
                device=device
            )
            # Encoder的输出通道数由backbone决定，这里直接使用Encoder的out_channels
            encoder_out_channels = self.encoder.out_channels
            # 强制VQVAE的num_hiddens与encoder_out_channels匹配
            num_hiddens = encoder_out_channels
        else:
            self.encoder = Encoder(in_channels, num_hiddens)
            encoder_out_channels = num_hiddens

        self.pre_quantization_conv = nn.Conv2d(encoder_out_channels, embedding_dim, kernel_size=1, stride=1)

        # 允许切换 EMA 版码本
        if use_ema_codebook:
            self.quantizer = EMAVectorQuantizer(num_embeddings, embedding_dim, commitment_cost, decay=ema_decay, eps=ema_eps)
        else:
            self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        self.decoder = Decoder(out_channels, num_hiddens, num_residual_layers, num_residual_hiddens)
        self.post_quantization_conv = nn.Conv2d(embedding_dim, num_hiddens, kernel_size=1, stride=1)

    def forward(self, x):
        # 1. Encode
        z_e = self.encoder(x)
        
        # 2. Pre-quantization convolution (map hidden channels to embedding dimension)
        z_e = self.pre_quantization_conv(z_e)
        
        # 3. Quantize
        quantized, vq_loss, perplexity, encoding_indices, active_code_ratio = self.quantizer(z_e)
        
        # 4. Post-quantization convolution (map embedding dimension back to hidden channels)
        quantized_h = self.post_quantization_conv(quantized)
        
        # 5. Decode
        # Decoder 总上采样 16x；若使用更浅的骨干（如 layer1 /4），会产生尺寸偏大/偏小，这里统一对齐到输入尺寸
        x_recon = self.decoder(quantized_h)
        if x_recon.shape[-2:] != x.shape[-2:]:
            x_recon = F.interpolate(
                x_recon, size=x.shape[-2:], mode="bilinear", align_corners=False
            )

        # 量化距离（用于 anomaly map）
        quantization_error = F.mse_loss(quantized, z_e, reduction="none").mean(dim=1)

        # 返回 perplexity 便于监控 codebook 使用情况
        return x_recon, vq_loss, perplexity, quantized, encoding_indices, quantization_error, active_code_ratio
