import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricPadding1d(nn.Module):
    def __init__(self, padding_left, padding_right):
        super(AsymmetricPadding1d, self).__init__()
        self.padding_left = padding_left
        self.padding_right = padding_right
    
    def forward(self, x):
        return F.pad(x, (self.padding_left, self.padding_right))

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

class AttentionBlockBN(nn.Module):
    def __init__(self, signal_size, channels, input_chanel=None, kernel_size=16):
        super(AttentionBlockBN, self).__init__()
        self.pad = AsymmetricPadding1d(kernel_size//2 - 1, kernel_size//2)
        self.conv = nn.Conv1d(
            input_chanel if input_chanel else channels,
            channels,
            kernel_size
        )
        self.activation = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(channels)
        self.dp = nn.Dropout(0.1) # dropout rate = 0.1 for qtdb
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
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
    def forward(self, x):
        x = self.pad(x)
        output = self.conv(x)
        output = self.activation(self.bn(output))
        output = self.dp(output)
        output = self.attention(output)
        output = self.maxpool(output)
        return output

class AttentionDeconvBN(nn.Module):
    def __init__(self, signal_size, channels, input_chanel=None,
                 kernel_size=16, activation='LeakyReLU',
                 strides=2, padding='same'):
        super(AttentionDeconvBN, self).__init__()
        self.deconv = nn.ConvTranspose1d(
            input_chanel if input_chanel else channels,
            channels,
            kernel_size,
            stride=strides,
            padding=kernel_size//2 - 1
        )
        self.bn = nn.BatchNorm1d(channels)
        self.activation = nn.LeakyReLU() if activation == 'LeakyReLU' else None
        self.dp = nn.Dropout(p=0.1) # dropout rate = 0.1 for qtdb
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
        output = self.deconv(x)
        output = self.bn(output)
        if self.activation is not None:
            output = self.activation(output)
        output = self.dp(output)
        output = self.attention(output)
        return output

class AttentionSkipDAE2(nn.Module):
    # Implementation of CBAM_DAE approach by PyTorch presented in
    # W. Chorney et al. (2023).
    # Convolutional block attention autoencoder for denoising electrocardiograms.
    # Biomed. Signal Process. Control., vol. 86, 2023, Art. no. 105242.
    
    def __init__(self, signal_size=512, filters=16):
        super(AttentionSkipDAE2, self).__init__()
        self.b1 = AttentionBlockBN(signal_size, filters, 1)
        self.b2 = AttentionBlockBN(signal_size//2, filters*2, filters)
        self.b3 = AttentionBlockBN(signal_size//4, filters*4, filters*2)
        self.b4 = AttentionBlockBN(signal_size//8, filters*4, filters*4)
        self.b5 = AttentionBlockBN(signal_size//16, 1, filters*4)
        self.d5 = AttentionDeconvBN(signal_size//16, filters*4, 1)
        self.d4 = AttentionDeconvBN(signal_size//8, filters*4, filters*4)
        self.d3 = AttentionDeconvBN(signal_size//4, filters*2, filters*4)
        self.d2 = AttentionDeconvBN(signal_size//2, filters, filters*2)
        self.d1 = AttentionDeconvBN(signal_size, 1, filters, activation='linear')
        
    def encode(self, x):
        encoded = self.b1(x)
        encoded = self.b2(encoded)
        encoded = self.b3(encoded)
        encoded = self.b4(encoded)
        encoded = self.b5(encoded)
        return encoded
    
    def decode(self, x):
        decoded = self.d5(x)
        decoded = self.d4(decoded)
        decoded = self.d3(decoded)
        decoded = self.d2(decoded)
        decoded = self.d1(decoded)
        return decoded
    
    def forward(self, x):
        enc1 = self.b1(x)
        enc2 = self.b2(enc1)
        enc3 = self.b3(enc2)
        enc4 = self.b4(enc3)
        enc5 = self.b5(enc4)
        dec5 = self.d5(enc5)
        dec4 = self.d4(dec5 + enc4)
        dec3 = self.d3(dec4 + enc3)
        dec2 = self.d2(dec3 + enc2)
        dec1 = self.d1(dec2 + enc1)
        return dec1