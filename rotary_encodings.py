import torch
import torch.nn as nn
# Positional encoding definition

class RotaryEncodings(nn.Module):
    def __init__(self, seq_len, head_dim, dropout=0.1, base=10_000,):
        super().__init__()      
        self.base = base
        self.head_dim = head_dim
        self.sequence_length = seq_len

        theta = 1. / (self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))  # [head_dim/2]
        index = torch.arange(seq_len).float()
        angles = index.unsqueeze(-1) * theta                     
        angles = torch.cat([angles, angles], dim=-1)             
        self.sin = angles.sin()[None, None, :, :]
        self.cos = angles.cos()[None, None, :, :]

    def neg(self, x):
        return torch.cat([-x[:, :, :, self.head_dim//2:], x[:, :, :, :self.head_dim//2]], dim=-1)
    
    def forward(self, x):
        seq_len = x.shape[2]
        cos = self.cos[:, :, :seq_len, :].to(x.device)
        sin = self.sin[:, :, :seq_len, :].to(x.device)
        neg_x = self.neg(x)
        return x * cos + neg_x * sin