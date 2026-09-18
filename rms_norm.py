import torch.nn as nn
import torch

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, weights, weight_name, eps=1e-15):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(weights[weight_name])
    
    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        output = output * self.weight
        return output
