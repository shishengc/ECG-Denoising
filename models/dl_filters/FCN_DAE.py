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

class Conv1DTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='elu'):
        super(Conv1DTranspose, self).__init__()
        self.conv_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)
        if activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'linear':
            self.activation = nn.Identity()
    
    def forward(self, x):
        x = self.conv_transpose(x)
        return self.activation(x)

class FCN_DAE(nn.Module):
    # Implementation of FCN_DAE approach by PyTorch presented in
    # Chiang, H. T., Hsieh, Y. Y., Fu, S. W., Hung, K. H., Tsao, Y., & Chien, S. Y. (2019).
    # Noise reduction in ECG signals using fully convolutional denoising autoencoders.
    # IEEE Access, 7, 60806-60813.
        
    def __init__(self, signal_size=512, filters=16):
        super(FCN_DAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 40, kernel_size=filters, stride=2, padding=7),
            nn.ELU(),
            nn.BatchNorm1d(40),
            
            nn.Conv1d(40, 20, kernel_size=filters, stride=2, padding=7),
            nn.ELU(),
            nn.BatchNorm1d(20),
            
            nn.Conv1d(20, 20, kernel_size=filters, stride=2, padding=7),
            nn.ELU(),
            nn.BatchNorm1d(20),
            
            nn.Conv1d(20, 20, kernel_size=filters, stride=2, padding=7),
            nn.ELU(),
            nn.BatchNorm1d(20),
            
            nn.Conv1d(20, 40, kernel_size=filters, stride=2, padding=7),
            nn.ELU(),
            nn.BatchNorm1d(40),
            
            AsymmetricPadding1d(7, 8),
            nn.Conv1d(40, 1, kernel_size=filters, stride=1),
            nn.ELU(),
            nn.BatchNorm1d(1)
        )
        
        self.decoder = nn.Sequential(
            AsymmetricPadding1d(7, 8),
            nn.Conv1d(1, 1, kernel_size=filters, stride=1),
            nn.ELU(),
            nn.BatchNorm1d(1),
            
            Conv1DTranspose(1, 40, kernel_size=filters, stride=2, padding=7, activation='elu'),
            nn.BatchNorm1d(40),
            
            Conv1DTranspose(40, 20, kernel_size=filters, stride=2, padding=7, activation='elu'),
            nn.BatchNorm1d(20),
            
            Conv1DTranspose(20, 20, kernel_size=filters, stride=2, padding=7, activation='elu'),
            nn.BatchNorm1d(20),
            
            Conv1DTranspose(20, 20, kernel_size=filters, stride=2, padding=7, activation='elu'),
            nn.BatchNorm1d(20),
            
            Conv1DTranspose(20, 40, kernel_size=filters, stride=2, padding=7, activation='elu'),
            nn.BatchNorm1d(40),
            
            AsymmetricPadding1d(7, 8),
            nn.Conv1d(40, 1, kernel_size=filters, stride=1)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        decoded = self.decoder(x)
        return decoded