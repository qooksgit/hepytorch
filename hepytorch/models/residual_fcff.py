import torch.nn as nn
from torch import Tensor

__all__ = ("ResidualFCFF",)


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels: int, hidden_channels: int, out_channels: int
    ) -> None:
        super().__init__()
        self.relu = nn.ReLU()

        self.layer1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, in_channels),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.layer1(x)
        x = self.relu(x + identity)
        x = self.layer2(x)
        return x


# https://wandb.ai/amanarora/Written-Reports/reports/Understanding-ResNets-A-Deep-Dive-into-Residual-Networks-with-PyTorch--Vmlldzo1MDAxMTk5
# 2305.08217v1.pdf: Neural Network predictions of inclusive electron-nucleus cross sections
# https://medium.com/@chen-yu/building-a-customized-residual-cnn-with-pytorch-471810e894ed
class ResidualFCFF(nn.Module):
    def __init__(self, **kwargs):
        input_features = kwargs.pop("input_features")
        hidden_features = kwargs.pop("hidden_features")
        output_features = kwargs.pop("output_features")
        super(ResidualFCFF, self).__init__(**kwargs)
        self.network = nn.Sequential(
            ResidualBlock(input_features, hidden_features, hidden_features),
            ResidualBlock(hidden_features, hidden_features, hidden_features),
            ResidualBlock(hidden_features, hidden_features, hidden_features),
            ResidualBlock(hidden_features, hidden_features, hidden_features),
            ResidualBlock(hidden_features, hidden_features, hidden_features),
            ResidualBlock(hidden_features, hidden_features, hidden_features),
            ResidualBlock(hidden_features, hidden_features, hidden_features),
            ResidualBlock(hidden_features, hidden_features, hidden_features),
            ResidualBlock(hidden_features, hidden_features, hidden_features),
            ResidualBlock(hidden_features, hidden_features, hidden_features),
            ResidualBlock(hidden_features, hidden_features, hidden_features),
            nn.Linear(hidden_features, output_features),
        )

    def forward(self, x):
        x = self.network(x)
        return x
