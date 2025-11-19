import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class base_Encoder(nn.Module):
    def __init__(self, kernel_size=5, bias=False):
        super(base_Encoder, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size, padding=(kernel_size - 1) // 2, bias=bias),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=1)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size, padding=(kernel_size - 1) // 2, bias=bias),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size, padding=(kernel_size - 1) // 2, bias=bias),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=1)
        )
        
        self.conv_block4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size, padding=(kernel_size - 1) // 2, bias=bias),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        return x


class base_Decoder(nn.Module):
    def __init__(self, kernel_size=5, bias=False):
        super(base_Decoder, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose1d(256, 128, kernel_size, padding=(kernel_size - 1) // 2, bias=bias),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.conv_block2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose1d(128, 64, kernel_size, padding=(kernel_size - 1) // 2, bias=bias),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.conv_block3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose1d(64, 32, kernel_size, padding=(kernel_size - 1) // 2, bias=bias),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        
        self.conv_block4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose1d(32, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=bias),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        return x
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0., max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1)].permute(1, 0, 2)
        return self.dropout(x)
    

class ARTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=4, dim_feedforward=512, dropout=0.):
        super(ARTransformer, self).__init__()

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.d_model = d_model
        
        
    def forward(self, x, target=None, training=True):
        batch_size, T, _ = x.shape

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_with_cls = torch.cat([cls_tokens, x], dim=1)  # [B, T+1, D]
        x_with_cls = self.pos_encoder(x_with_cls)

        mask = self._generate_autoregressive_mask(T + 1, x.device)

        output = self.transformer_encoder(x_with_cls, mask)  # [B, T+1, D]

        seq_output = output[:, 1:, :]  # [B, T, D]

        if training:
            ar_loss = F.mse_loss(seq_output, target)
            return seq_output, ar_loss
        else:
            return seq_output
        
    def _generate_autoregressive_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

class AR(nn.Module):
    def __init__(self, kernel_size=5, bias=False):
        super(AR, self).__init__()
        self.encoder = base_Encoder(kernel_size, bias)
        self.decoder = base_Decoder(kernel_size, bias)
        self.transformer = ARTransformer(num_layers=4)
        
    def wavelet(self, ecg_signal, wavelet='db6', level=8):
        import pywt
        import warnings
        warnings.filterwarnings("ignore")
        device = ecg_signal.device
        signal_np = ecg_signal.cpu().numpy()
        B, C, L = signal_np.shape
        flat = signal_np.reshape(-1, L)
        coeffs = pywt.wavedec(flat, wavelet, level=level, axis=-1)
        coeffs[0][:] = 0.0

        for k in range(1, 4):
            idx = -k
            arr = coeffs[idx]
            med = np.median(np.abs(arr), axis=1)
            sigma = med / 0.6745
            thresh = sigma * np.sqrt(2 * np.log(L))
            thresh = thresh[:, None]
            coeffs[idx] = np.sign(arr) * np.maximum(np.abs(arr) - thresh, 0.0)

        rec = pywt.waverec(coeffs, wavelet, axis=-1)
        rec = rec[:, :L]
        filtered_np = rec.reshape(B, C, L)
        return torch.FloatTensor(filtered_np).to(device)
    
    def forward(self, x_n, x_c):
        x_c = self.wavelet(x_n)
        # x_c = x_n
        
        # f_n = self.encoder(x_n)
        # rec_n = self.decoder(f_n)
        
        f_c = self.encoder(x_c)
        rec_c = self.decoder(f_c)
        
        # f_n = f_n.permute(0, 2, 1)
        f_c = f_c.permute(0, 2, 1)
        
        _, ar_loss = self.transformer(f_c, f_c, training=True)
        
        t_loss_c = F.mse_loss(rec_c, x_c)
        
        tc_loss = 20 * ar_loss + \
                  10 * t_loss_c
        
        return tc_loss, t_loss_c, ar_loss

    def inference(self, x):
        x = self.wavelet(x)
        features = self.encoder(x)
        rec = self.decoder(features)
        return rec


if __name__ == "__main__":
    model = AR()
    model = model.to('cuda:0')
    x = torch.randn(64, 1, 512).to('cuda:0')
    tc_loss = model(x)
    reconstructed = model.inference(x)
    
    print("Shape of reconstructed:", reconstructed.shape)