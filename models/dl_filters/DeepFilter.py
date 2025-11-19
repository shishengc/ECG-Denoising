import torch
import torch.nn as nn

class LANLFilter_module(nn.Module):
    def __init__(self, in_channels, channels):
        super(LANLFilter_module, self).__init__()
        self.LB0 = nn.Conv1d(in_channels, int(channels/8), kernel_size=3, stride=1, padding=1)
        self.LB1 = nn.Conv1d(in_channels, int(channels/8), kernel_size=5, stride=1, padding=2)
        self.LB2 = nn.Conv1d(in_channels, int(channels/8), kernel_size=9, stride=1, padding=4)
        self.LB3 = nn.Conv1d(in_channels, int(channels/8), kernel_size=15, stride=1, padding=7)
        
        self.NLB0 = nn.Sequential(
            nn.Conv1d(in_channels, int(channels/8), kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.NLB1 = nn.Sequential(
            nn.Conv1d(in_channels, int(channels/8), kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )
        self.NLB2 = nn.Sequential(
            nn.Conv1d(in_channels, int(channels/8), kernel_size=9, stride=1, padding=4),
            nn.ReLU()
        )
        self.NLB3 = nn.Sequential(
            nn.Conv1d(in_channels, int(channels/8), kernel_size=15, stride=1, padding=7),
            nn.ReLU()
        )
        
    def forward(self, x):
        lb0 = self.LB0(x)
        lb1 = self.LB1(x)
        lb2 = self.LB2(x)
        lb3 = self.LB3(x)
        
        nlb0 = self.NLB0(x)
        nlb1 = self.NLB1(x)
        nlb2 = self.NLB2(x)
        nlb3 = self.NLB3(x)
        
        return torch.cat([lb0, lb1, lb2, lb3, nlb0, nlb1, nlb2, nlb3], dim=1)

class LANLFilter_module_dilated(nn.Module):
    def __init__(self, in_channels, channels):
        super(LANLFilter_module_dilated, self).__init__()
        self.LB1 = nn.Conv1d(in_channels, int(channels/6), kernel_size=5, dilation=3, padding=6)
        self.LB2 = nn.Conv1d(in_channels, int(channels/6), kernel_size=9, dilation=3, padding=12)
        self.LB3 = nn.Conv1d(in_channels, int(channels/6), kernel_size=15, dilation=3, padding=21)
        
        self.NLB1 = nn.Sequential(
            nn.Conv1d(in_channels, int(channels/6), kernel_size=5, dilation=3, padding=6),
            nn.ReLU()
        )
        self.NLB2 = nn.Sequential(
            nn.Conv1d(in_channels, int(channels/6), kernel_size=9, dilation=3, padding=12),
            nn.ReLU()
        )
        self.NLB3 = nn.Sequential(
            nn.Conv1d(in_channels, int(channels/6), kernel_size=15, dilation=3, padding=21),
            nn.ReLU()
        )
        
    def forward(self, x):
        lb1 = self.LB1(x)
        lb2 = self.LB2(x)
        lb3 = self.LB3(x)
        
        nlb1 = self.NLB1(x)
        nlb2 = self.NLB2(x)
        nlb3 = self.NLB3(x)
        
        return torch.cat([lb1, lb2, lb3, nlb1, nlb2, nlb3], dim=1)

class DeepFilterModelLANLDilated(nn.Module):
    def __init__(self, signal_size=512):
        super(DeepFilterModelLANLDilated, self).__init__()
        
        self.layer1 = LANLFilter_module(1, 64)
        self.dropout1 = nn.Dropout(0.4)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.layer2 = LANLFilter_module_dilated(64, 64)
        self.dropout2 = nn.Dropout(0.4)
        self.bn2 = nn.BatchNorm1d(int(64/6)*6)
        
        self.layer3 = LANLFilter_module(int(64/6)*6, 32)
        self.dropout3 = nn.Dropout(0.4)
        self.bn3 = nn.BatchNorm1d(32)
        
        self.layer4 = LANLFilter_module_dilated(32, 32)
        self.dropout4 = nn.Dropout(0.4)
        self.bn4 = nn.BatchNorm1d(int(32/6)*6)
        
        self.layer5 = LANLFilter_module(int(32/6)*6, 16)
        self.dropout5 = nn.Dropout(0.4)
        self.bn5 = nn.BatchNorm1d(16)
        
        self.layer6 = LANLFilter_module_dilated(16, 16)
        self.dropout6 = nn.Dropout(0.4)
        self.bn6 = nn.BatchNorm1d(int(16/6)*6)
        
        self.output_conv = nn.Conv1d(int(16/6)*6, 1, kernel_size=9, stride=1, padding=4)
        
    def forward(self, x):
        
        x = self.layer1(x)
        x = self.dropout1(x)
        x = self.bn1(x)
        
        x = self.layer2(x)
        x = self.dropout2(x)
        x = self.bn2(x)
        
        x = self.layer3(x)
        x = self.dropout3(x)
        x = self.bn3(x)
        
        x = self.layer4(x)
        x = self.dropout4(x)
        x = self.bn4(x)
        
        x = self.layer5(x)
        x = self.dropout5(x)
        x = self.bn5(x)
        
        x = self.layer6(x)
        x = self.dropout6(x)
        x = self.bn6(x)
        
        x = self.output_conv(x)
        
        return x
    
    
class DeepFilterModelLANL(nn.Module):
    def __init__(self, signal_size=512):
        super(DeepFilterModelLANL, self).__init__()
        
        self.layer1 = LANLFilter_module(1, 64)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.layer2 = LANLFilter_module(64, 64)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.layer3 = LANLFilter_module(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        
        self.layer4 = LANLFilter_module(32, 32)
        self.bn4 = nn.BatchNorm1d(32)
        
        self.layer5 = LANLFilter_module(32, 16)
        self.bn5 = nn.BatchNorm1d(16)
        
        self.layer6 = LANLFilter_module(16, 16)
        self.bn6 = nn.BatchNorm1d(16)
        
        self.output_conv = nn.Conv1d(16, 1, kernel_size=9, stride=1, padding=4)
        
    def forward(self, x):
            
        x = self.layer1(x)
        x = self.bn1(x)
        
        x = self.layer2(x)
        x = self.bn2(x)
        
        x = self.layer3(x)
        x = self.bn3(x)
        
        x = self.layer4(x)
        x = self.bn4(x)
        
        x = self.layer5(x)
        x = self.bn5(x)
        
        x = self.layer6(x)
        x = self.bn6(x)
        
        x = self.output_conv(x)
        
        return x