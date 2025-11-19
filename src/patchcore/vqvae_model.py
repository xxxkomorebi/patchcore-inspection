import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# 1. Vector Quantizer (Codebook)
# -----------------------------------------------------------------------------

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
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        # Find the closest codebook vector index
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        # Convert indices to one-hot vectors
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize the input vector (lookup the codebook vector)
        # [B*H*W, C]
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)

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

        return quantized, loss, perplexity, encoding_indices.view(input_shape[:-1])

# -----------------------------------------------------------------------------
# 2. Encoder and Decoder (Simple ConvNet for feature map output)
# -----------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    Simple Encoder for VQ-VAE. Designed to output a feature map.
    Supports image input (3 channels).
    """
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, aux_dim=0):
        super().__init__()
        self.in_channels = in_channels
        self.num_hiddens = num_hiddens
        self.aux_dim = aux_dim

        # Initial convolution layer (to handle potential multi-modal fusion later)
        self.conv_in = nn.Conv2d(in_channels, num_hiddens // 2, kernel_size=4, stride=2, padding=1)
        self.conv_mid = nn.Conv2d(num_hiddens // 2, num_hiddens, kernel_size=4, stride=2, padding=1)
        
        # Residual blocks (omitted for simplicity, using basic convs)
        self.conv_out = nn.Conv2d(num_hiddens, num_hiddens, kernel_size=3, stride=1, padding=1)
        
        if self.aux_dim > 0:
            # Fusion layer: maps auxiliary dimension to the channel dimension of the first conv layer
            self.aux_fusion = nn.Linear(self.aux_dim, num_hiddens // 2)

    def forward(self, x, aux=None):
        # x is expected to be [B, C, H, W] (e.g., image)
        h = self.conv_in(x)
        
        if self.aux_dim > 0 and aux is not None:
            # Aux fusion: [B, aux_dim] -> [B, num_hiddens // 2]
            aux_h = self.aux_fusion(aux)
            
            # Reshape aux_h to match spatial dimensions of h: [B, C_h, 1, 1]
            aux_h = aux_h.unsqueeze(-1).unsqueeze(-1)
            
            # Add aux feature to image feature map (broadcast addition)
            h = h + aux_h
            
        h = F.relu(h)
        h = F.relu(self.conv_mid(h))
        return self.conv_out(h) # Output feature map [B, num_hiddens, H/4, W/4]

class Decoder(nn.Module):
    """
    Simple Decoder for VQ-VAE. Reconstructs image from feature map.
    """
    def __init__(self, out_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        self.out_channels = out_channels
        self.num_hiddens = num_hiddens

        # Initial convolution layer
        self.conv_in = nn.Conv2d(num_hiddens, num_hiddens, kernel_size=3, stride=1, padding=1)
        
        # Transposed convolutions for upsampling
        self.conv_t1 = nn.ConvTranspose2d(num_hiddens, num_hiddens // 2, kernel_size=4, stride=2, padding=1)
        self.conv_t2 = nn.ConvTranspose2d(num_hiddens // 2, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        h = F.relu(self.conv_in(x))
        h = F.relu(self.conv_t1(h))
        return self.conv_t2(h) # Output reconstruction [B, out_channels, H, W]

# -----------------------------------------------------------------------------
# 3. VQVAE Model
# -----------------------------------------------------------------------------

class VQVAE(nn.Module):
    """
    VQ-VAE Model integrating Encoder, Quantizer, and Decoder.
    """
    def __init__(self, in_channels, out_channels, num_hiddens, num_residual_layers, num_residual_hiddens,
                 num_embeddings, embedding_dim, commitment_cost, aux_dim=0):
        super().__init__()
        self.aux_dim = aux_dim

        self.encoder = Encoder(in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, aux_dim=aux_dim)
        self.pre_quantization_conv = nn.Conv2d(num_hiddens, embedding_dim, kernel_size=1, stride=1)
        
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        
        self.decoder = Decoder(out_channels, num_hiddens, num_residual_layers, num_residual_hiddens)
        self.post_quantization_conv = nn.Conv2d(embedding_dim, num_hiddens, kernel_size=1, stride=1)

    def forward(self, x, aux=None):
        # 1. Encode (handles multi-modal fusion internally)
        z_e = self.encoder(x, aux)
        
        # 2. Pre-quantization convolution (map hidden channels to embedding dimension)
        z_e = self.pre_quantization_conv(z_e)
        
        # 3. Quantize
        quantized, vq_loss, perplexity, encoding_indices = self.quantizer(z_e)
        
        # 4. Post-quantization convolution (map embedding dimension back to hidden channels)
        quantized_h = self.post_quantization_conv(quantized)
        
        # 5. Decode
        x_recon = self.decoder(quantized_h)

        return x_recon, vq_loss, quantized, encoding_indices