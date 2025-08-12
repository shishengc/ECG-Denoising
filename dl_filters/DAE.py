import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialGate(nn.Module):
    def __init__(self, filters, kernel_size, input_shape=None, activation='sigmoid', transpose=False):
        super(SpatialGate, self).__init__()
        self.transpose = transpose
        self.conv = nn.Conv1d(2, filters, kernel_size, padding=kernel_size//2)
        self.activation = nn.Sigmoid() if activation == 'sigmoid' else None
        
    def forward(self, x):
        if self.transpose:
            x = x.permute(0, 2, 1)
        
        avg_ = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_, max_], dim=1)
        out = self.conv(x)
        
        if self.activation:
            out = self.activation(out)
            
        if self.transpose:
            out = out.permute(0, 2, 1)
        return out

class ChannelGate(nn.Module):
    def __init__(self, filters, kernel_size, input_shape=None, activation='sigmoid', transpose=False):
        super(ChannelGate, self).__init__()
        self.transpose = transpose
        self.conv = nn.Conv1d(1, filters, kernel_size, padding=kernel_size//2)
        self.activation = nn.Sigmoid() if activation == 'sigmoid' else None
        
    def forward(self, x):
        if self.transpose:
            x = x.permute(0, 2, 1)
        
        x = torch.mean(x, dim=2, keepdim=True)
        x = x.permute(0, 2, 1)
        out = self.conv(x)
        out = out.permute(0, 2, 1)
        
        if self.activation:
            out = self.activation(out)
            
        if self.transpose:
            out = out.permute(0, 2, 1)
        return out

class CBAM(nn.Module):
    def __init__(self, c_filters, c_kernel, c_input, c_transpose,
                 s_filters, s_kernel, s_input, s_transpose, spatial=True):
        super(CBAM, self).__init__()
        self.spatial = spatial
        self.channel_attention = ChannelGate(c_filters, c_kernel, input_shape=c_input, transpose=c_transpose)
        self.spatial_attention = SpatialGate(s_filters, s_kernel, input_shape=s_input, transpose=s_transpose)
        
    def forward(self, x):
        channel_mask = self.channel_attention(x)
        x = channel_mask * x
        if self.spatial:
            spatial_mask = self.spatial_attention(x)
            x = spatial_mask * x
        return x

class ConvCBAM(nn.Module):
    def __init__(self, signal_size, channels, input_chanel=None, kernel_size=3, dwonsample=True):
        super(ConvCBAM, self).__init__()
        self.conv0 = nn.Conv1d(
            input_chanel if input_chanel else channels,
            channels,
            kernel_size,
            padding=kernel_size//2
        )
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=kernel_size//2
        )
        
        self.bn0 = nn.BatchNorm1d(channels)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act = nn.LeakyReLU()
        self.attention = CBAM(
            1,
            3,
            (channels, 1),
            False,
            1,
            7,
            (signal_size, 1),
            False
        )
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2) if dwonsample else nn.Identity()
        
    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        
        x = self.attention(x)
        x = self.maxpool(x)
        return x
    
class DeConvCBAM(nn.Module):
    def __init__(self, signal_size, channels, input_chanel=None, kernel_size=3, upsample=True):
        super(DeConvCBAM, self).__init__()
        self.conv0 = nn.ConvTranspose1d(
            input_chanel if input_chanel else channels,
            channels,
            kernel_size,
            stride=2,
            padding=kernel_size//2,
            output_padding=1
        ) if upsample else nn.ConvTranspose1d(
            input_chanel if input_chanel else channels,
            channels,
            kernel_size,
            padding=kernel_size//2
        )
        self.conv1 = nn.ConvTranspose1d(
            channels,
            channels,
            kernel_size,
            padding=kernel_size//2
        )
        
        self.bn0 = nn.BatchNorm1d(channels)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act = nn.LeakyReLU()
        
        self.attention = CBAM(
            1,
            3,
            (channels, 1),
            False,
            1,
            7,
            (signal_size, 1),
            False
        )
        
    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        
        x = self.attention(x)
        return x


class AttentionDAE(nn.Module):    
    def __init__(self, signal_size=512, filters=16):
        super(AttentionDAE, self).__init__()
        self.b1 = ConvCBAM(signal_size, filters, 1, dwonsample=False)
        self.b2 = ConvCBAM(signal_size, filters*2, filters)
        self.b3 = ConvCBAM(signal_size//2, filters*4, filters*2)
        self.b4 = ConvCBAM(signal_size//4, filters*8, filters*4)
        self.d4 = DeConvCBAM(signal_size//8, filters*4, filters*8)
        self.d3 = DeConvCBAM(signal_size//4, filters*2, filters*4)
        self.d2 = DeConvCBAM(signal_size//2, filters, filters*2)
        self.d1 = DeConvCBAM(signal_size, 1, filters, upsample=False)
        
    def encode(self, x):
        encoded = self.b1(x)
        encoded = self.b2(encoded)
        encoded = self.b3(encoded)
        encoded = self.b4(encoded)
        return encoded
    
    def decode(self, x):
        decoded = self.d4(x)
        decoded = self.d3(decoded)
        decoded = self.d2(decoded)
        decoded = self.d1(decoded)
        return decoded
    
    def forward(self, x):
        enc1 = self.b1(x)
        enc2 = self.b2(enc1)
        enc3 = self.b3(enc2)
        enc4 = self.b4(enc3)
        dec4 = self.d4(enc4)
        dec3 = self.d3(dec4)
        dec2 = self.d2(dec3)
        dec1 = self.d1(dec2)
        return dec1