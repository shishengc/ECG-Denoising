import numpy as np
import torch
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class AGs(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AGs, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = LayerNorm()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        x = x * psi
        return x

    def forward_attention(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        x = x * psi
        return x, psi


class LayerNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data):
        ans = nn.functional.layer_norm(data, (data.shape[-1],)).to(device=data.device)
        return ans


class Encoder(nn.Module):
    def __init__(self, in_channels=1):
        super(Encoder, self).__init__()
        
        self.Maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.Conv1 = conv_block(ch_in=in_channels, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)
        
    def forward(self, x):
        x1 = self.Conv1(x)
        
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        
        return [x1, x2, x3, x4, x5]


class Decoder(nn.Module):
    def __init__(self, out_channels=1):
        super(Decoder, self).__init__()
        
        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = AGs(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)
        
        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = AGs(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = AGs(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = AGs(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)
        
        self.Conv_out = nn.Conv1d(64, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        
    def forward(self, emb):
        """
        Args:
            encoder_features: [x1, x2, x3, x4, x5]
        Returns:
            output: 
            attn_maps:
        """
        x1, x2, x3, x4, x5 = emb
        d5 = self.Up5(x5)
        x4_att, _ = self.Att5.forward_attention(g=d5, x=x4)
        d5 = torch.cat((x4_att, d5), dim=1)
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3_att, _ = self.Att4.forward_attention(g=d4, x=x3)
        d4 = torch.cat((x3_att, d4), dim=1)
        d4 = self.Up_conv4(d4)
        
        d3 = self.Up3(d4)
        x2_att, _ = self.Att3.forward_attention(g=d3, x=x2)
        d3 = torch.cat((x2_att, d3), dim=1)
        d3 = self.Up_conv3(d3)
        
        d2 = self.Up2(d3)
        x1_att, _ = self.Att2.forward_attention(g=d2, x=x1)
        d2 = torch.cat((x1_att, d2), dim=1)
        d2 = self.Up_conv2(d2)
        
        output = self.Conv_out(d2)
        
        return output
        
    def forward_attention(self, emb):
        """
        Args:
            encoder_features: [x1, x2, x3, x4, x5]
        Returns:
            output: 
            attn_maps:
        """
        x1, x2, x3, x4, x5 = emb
        attn_maps = []

        d5 = self.Up5(x5)
        x4_att, att5 = self.Att5.forward_attention(g=d5, x=x4)
        attn_maps.append(att5)
        d5 = torch.cat((x4_att, d5), dim=1)
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3_att, att4 = self.Att4.forward_attention(g=d4, x=x3)
        attn_maps.append(att4)
        d4 = torch.cat((x3_att, d4), dim=1)
        d4 = self.Up_conv4(d4)
        
        d3 = self.Up3(d4)
        x2_att, att3 = self.Att3.forward_attention(g=d3, x=x2)
        attn_maps.append(att3)
        d3 = torch.cat((x2_att, d3), dim=1)
        d3 = self.Up_conv3(d3)
        
        d2 = self.Up2(d3)
        x1_att, att2 = self.Att2.forward_attention(g=d2, x=x1)
        attn_maps.append(att2)
        d2 = torch.cat((x1_att, d2), dim=1)
        d2 = self.Up_conv2(d2)
        
        output = self.Conv_out(d2)
        
        return output, attn_maps


class SemiAE(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(SemiAE, self).__init__()

        self.encoder_c = Encoder(in_channels)
        self.encoder_n = Encoder(in_channels)

        self.decoder = Decoder(out_channels)
        
        self.loss_func = nn.MSELoss(reduction='mean')
        self._lambda = 10
        
    def forward(self, x_n):
        noise_embed = self.encoder_n(x_n)
        out_n = self.decoder(noise_embed)
        
        return out_n
    
    def _make_ae(self):
        class _Wrapper(nn.Module):
            def __init__(self, encoder, decoder):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder
                
            def forward(self, x):
                features = self.encoder(x)
                output = self.decoder(features)
                return output
    
        return _Wrapper(self.encoder_n, self.decoder)
        
    def forward_attention(self, x_c, x_n):
        clean_embed = self.encoder_c(x_c)
        noise_embed = self.encoder_n(x_n)

        out_c, attn_maps_c = self.decoder.forward_attention(clean_embed)
        out_n, attn_maps_n = self.decoder.forward_attention(noise_embed)
        
        return out_c, out_n, attn_maps_c, attn_maps_n
    
    def get_loss(self, x_c, x_n):
        out_c, out_n, attn_maps_c, attn_maps_n = self.forward_attention(x_c, x_n)
        
        loss_recon_c = self.loss_func(out_c, x_c)
        loss_recon_n = self.loss_func(out_n, x_c)
        
        loss_attn = 0
        for i in range(len(attn_maps_c)):
            loss_attn += self.loss_func(attn_maps_c[i], attn_maps_n[i])
        
        loss = loss_recon_c + loss_recon_n + self._lambda * loss_attn
        return loss