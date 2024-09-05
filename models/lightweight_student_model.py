import torch
import torch.nn as nn
from torch.nn import Transformer

class LightweightStudentModel(nn.Module):
    def __init__(self):
        super(LightweightStudentModel, self).__init__()
        self.backbone = Transformer(d_model=128, nhead=4, num_encoder_layers=2)  
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.backbone(x)
        x = self.pooling(x)
        return x
