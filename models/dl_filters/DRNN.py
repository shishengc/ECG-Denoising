import torch
import torch.nn as nn

class DRDNN(nn.Module):
    # Implementation of DRNN approach by PyTorch presented in
    # Antczak, K. (2018). Deep recurrent neural networks for ECG signal denoising.
    # arXiv preprint arXiv:1807.11551.
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super(DRDNN, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, input_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Input shape: (B, C, L) -> reshape to (B, L, C) for LSTM
        x = x.permute(0, 2, 1)
        
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Fully connected layers
        out = self.relu(self.fc1(lstm_out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        
        # Reshape back to (B, C, L)
        out = out.permute(0, 2, 1)
        return out