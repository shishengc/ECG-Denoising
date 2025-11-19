import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        super(PositionalEncoding1D, self).__init__()
        self.channels = int(np.ceil(channels / 2) * 2)
        self.register_buffer(
            "inv_freq", 
            1.0 / (10000 ** (torch.arange(0, self.channels, 2).float() / self.channels))
        )
        self.cached_penc = None
        
    def forward(self, x):
        if len(x.shape) != 3:
            raise RuntimeError("The input tensor must be 3D!")
            
        if self.cached_penc is not None and self.cached_penc.shape == x.shape:
            return self.cached_penc
            
        self.cached_penc = None
        _, seq_len, _ = x.shape
        
        pos = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        sin_inp_x = torch.einsum("i,j->ij", pos, self.inv_freq)
        emb = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        emb = emb[:, :x.shape[-1]]
        self.cached_penc = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        
        return self.cached_penc


class TransformerEncoder(nn.Module):
    def __init__(self, dim, head_size, num_heads, ff_dim, dropout=0):
        super(TransformerEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.ff = nn.Sequential(
            nn.Conv1d(dim, ff_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(ff_dim, dim, kernel_size=1)
        )
        
    def forward(self, x):
        norm_x = self.norm1(x)
        attn_output, _ = self.attn(norm_x, norm_x, norm_x)
        x1 = x + self.dropout1(attn_output)
        
        norm_x1 = self.norm2(x1)
        norm_x1_perm = norm_x1.permute(0, 2, 1)
        ff_output = self.ff(norm_x1_perm).permute(0, 2, 1)
        
        return x1 + ff_output


class AddGatedNoise(nn.Module):
    def __init__(self):
        super(AddGatedNoise, self).__init__()
        
    def forward(self, x):
        if self.training:
            noise = torch.rand_like(x) * 2 - 1  # [-1, 1]
            return x * (1 + noise)
        return x


class TCDAE(nn.Module):
    def __init__(self, signal_size=512, head_size=64, num_heads=8, ff_dim=64, num_transformer_layers=6, dropout=0):
        super(TCDAE, self).__init__()
        self.ks = 13
        
        # Encoder layers
        self.conv0 = nn.Conv1d(1, 16, kernel_size=self.ks, stride=2, padding=self.ks//2)
        self.gated_noise0 = AddGatedNoise()
        self.conv0_ = nn.Conv1d(1, 16, kernel_size=self.ks, stride=2, padding=self.ks//2)
        self.bn0 = nn.BatchNorm1d(16)
        
        self.conv1 = nn.Conv1d(16, 32, kernel_size=self.ks, stride=2, padding=self.ks//2)
        self.gated_noise1 = AddGatedNoise()
        self.conv1_ = nn.Conv1d(16, 32, kernel_size=self.ks, stride=2, padding=self.ks//2)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=self.ks, stride=2, padding=self.ks//2)
        self.gated_noise2 = AddGatedNoise()
        self.conv2_ = nn.Conv1d(32, 64, kernel_size=self.ks, stride=2, padding=self.ks//2)
        self.bn2 = nn.BatchNorm1d(64)
        

        reduced_size = signal_size // 8 
        self.pos_encoding = PositionalEncoding1D(reduced_size)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoder(64, head_size, num_heads, ff_dim, dropout)
            for _ in range(num_transformer_layers)
        ])
        
        # Decoder layers
        self.deconv1 = nn.Sequential(nn.ConvTranspose1d(64, 64, kernel_size=self.ks, stride=1, padding=self.ks//2),
                                     nn.ELU())
        self.bn3 = nn.BatchNorm1d(64)
        
        self.deconv2 = nn.Sequential(nn.ConvTranspose1d(64, 32, kernel_size=self.ks, stride=2, padding=self.ks//2, output_padding=1),
                                     nn.ELU())
        self.bn4 = nn.BatchNorm1d(32)
        
        self.deconv3 = nn.Sequential(nn.ConvTranspose1d(32, 16, kernel_size=self.ks, stride=2, padding=self.ks//2, output_padding=1),
                                     nn.ELU())
        self.bn5 = nn.BatchNorm1d(16)
        
        self.deconv4 = nn.ConvTranspose1d(16, 1, kernel_size=self.ks, stride=2, padding=self.ks//2, output_padding=1)
        
    def forward(self, x):
            
        # Encoder path
        x0 = self.conv0(x)
        x0 = self.gated_noise0(x0)
        x0 = torch.sigmoid(x0)
        
        x0_ = self.conv0_(x)
        xmul0 = x0 * x0_
        xmul0 = self.bn0(xmul0)
        
        x1 = self.conv1(xmul0)
        x1 = self.gated_noise1(x1)
        x1 = torch.sigmoid(x1)
        
        x1_ = self.conv1_(xmul0)
        xmul1 = x1 * x1_
        xmul1 = self.bn1(xmul1)
        
        x2 = self.conv2(xmul1)
        x2 = self.gated_noise2(x2)
        x2 = torch.sigmoid(x2)
        
        x2_ = self.conv2_(xmul1)
        xmul2 = x2 * x2_
        xmul2 = self.bn2(xmul2)

        x3 = xmul2.permute(0, 2, 1)
        pos_embed = self.pos_encoding(x3)
        x3 = x3 + pos_embed
        
        for transformer_block in self.transformer_blocks:
            x3 = transformer_block(x3)

        x4 = x3.permute(0, 2, 1)
        
        # Decoder path with residual connections
        x5 = self.deconv1(x4)
        x5 = x5 + xmul2
        x5 = self.bn3(x5)
        
        x6 = self.deconv2(x5)
        x6 = x6 + xmul1
        x6 = self.bn4(x6)
        
        x7 = self.deconv3(x6)
        x7 = x7 + xmul0
        x7 = self.bn5(x7)
        
        x8 = self.deconv4(x7)
        
        return x8
    
    
if __name__ == "__main__":
    model = TCDAE(signal_size=512, head_size=64, num_heads=8, ff_dim=64, num_transformer_layers=6, dropout=0.1)
    x = torch.randn(1, 1, 512)  # Example input
    output = model(x)
    print(output.shape) 