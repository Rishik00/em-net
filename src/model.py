import torch
import torch.nn.functional as F
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class ModelConfig:
    input_dim: int = 3
    hidden_dim: int = 256
    output_dim: int = 1
    lr: float = 0.001
    num_epochs: int = 100
    batch_size: int = 24

class DenseNetwork(nn.Module):
    def __init__(self, config: ModelConfig):
        super(DenseNetwork, self).__init__()
        
        self.l1 = nn.Linear(config.input_dim, config.hidden_dim)
        self.l2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.l3 = nn.Linear(config.hidden_dim, config.output_dim)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)

        x = self.l2(x)
        x = F.relu(x)

        out = self.l3(x)
        return out
