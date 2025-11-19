import torch
import torch.nn as nn
import torch.nn.functional as F

class ECAModule(nn.Module):
    def __init__(self, channels, k_size=3):
        super(ECAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x.transpose(-1, -2))
        y = self.conv(y.transpose(-1, -2))
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class ACDAE(nn.Module):
    def __init__(self, in_channels=1):
        super(ACDAE, self).__init__()
        
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, 16, kernel_size=13, padding=6),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                nn.Conv1d(16, 32, kernel_size=7, padding=3),
                nn.MaxPool1d(2),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=7, padding=3),
                nn.MaxPool1d(2),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=7, padding=3),
                nn.MaxPool1d(2),
                nn.LeakyReLU(0.1)
            )
        ])
        
        self.decoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose1d(128, 128, kernel_size=7, stride=1, padding=3),
                nn.LeakyReLU(0.1),
                ECAModule(128)
            ),
            nn.Sequential(
                nn.ConvTranspose1d(128, 64, kernel_size=7, stride=2, padding=3, output_padding=1),
                nn.LeakyReLU(0.1),
                ECAModule(64)
            ),
            nn.Sequential(
                nn.ConvTranspose1d(64, 32, kernel_size=7, stride=2, padding=3, output_padding=1),
                nn.LeakyReLU(0.1),
                ECAModule(32)
            ),
            nn.Sequential(
                nn.ConvTranspose1d(32, 16, kernel_size=13, stride=2, padding=6, output_padding=1),
                nn.LeakyReLU(0.1)
            )
        ])
        
        self.output_conv = nn.Conv1d(16, 1, kernel_size=1)

    def forward(self, x):
        skips = []
        
        for layer in self.encoder_layers:
            x = layer(x)
            skips.append(x)
        
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)
            if i < 3:
                skip_connection = skips[-(i+1)]
                x = x + skip_connection
        
        x = self.output_conv(x)
        return x