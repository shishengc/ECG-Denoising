import torch
import torch.nn as nn


##### Generator Class #####

class Generator(nn.Module):
    def __init__(
        self, 
        input_channels=1,
        latent_dim=512,
        encoder_channels=[16, 32, 64, 128, 256, 512],
        decoder_channels=[256, 128, 64, 32, 16, 1],
        kernel_size=31,
        stride=2,
        padding_mode='reflect'
    ):
        super(Generator, self).__init__()
        
        # Encoder layers
        self.encoder = nn.ModuleList()
        in_channels = input_channels
        for out_channels in encoder_channels:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels, 
                        out_channels, 
                        kernel_size=kernel_size, 
                        stride=stride, 
                        padding=kernel_size // 2,
                        padding_mode=padding_mode
                    ),
                    nn.PReLU()
                )
            )
            in_channels = out_channels
        
        # Decoder layers
        self.decoder = nn.ModuleList()
        for i, out_channels in enumerate(decoder_channels):
            if i == 0:
                in_channels = encoder_channels[-1] + latent_dim
            else:
                in_channels = decoder_channels[i-1] + encoder_channels[-(i+1)]
            
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        in_channels, 
                        out_channels, 
                        kernel_size=kernel_size, 
                        stride=stride, 
                        padding=kernel_size // 2,
                        output_padding=stride - 1 
                    ),
                    nn.PReLU()
                )
            )
        
    def forward(self, x, z):
        # Encoder
        skips = []
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)
        
        # Concatenate latent vector z
        x = torch.cat([x, z], dim=1)
        
        # Decoder with skip connections
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            if i < len(skips) - 1:
                x = torch.cat([x, skips[-(i+2)]], dim=1)
        
        return x
    


##### Discriminator Class #####

class Discriminator(nn.Module):
    def __init__(
        self, 
        input_channels=2,  
        hidden_channels=[16, 32, 64, 128, 256, 512],
        kernel_size=31,
        stride=2,
        padding_mode='reflect'
    ):
        super(Discriminator, self).__init__()
        
        self.layers = nn.ModuleList()
        in_channels = input_channels
        for out_channels in hidden_channels:
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels, 
                        out_channels, 
                        kernel_size=kernel_size, 
                        stride=stride, 
                        padding=kernel_size // 2,
                        padding_mode=padding_mode
                    ),
                    nn.LeakyReLU(0.2)
                )
            )
            in_channels = out_channels
        
        self.final_conv = nn.Conv1d(in_channels, 1, kernel_size=1)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final_conv(x)
        return torch.sigmoid(x.mean(dim=(1, 2)))