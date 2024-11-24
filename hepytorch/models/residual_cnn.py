import torch.nn as nn
from torch import Tensor
import logging

__all__ = ("ResidualCNN",)


class ResidualBlock(nn.Module):
    def __init__(
        self, id: int, in_channels: int, hidden_channels: int, out_channels: int
    ) -> None:
        super().__init__()
        self.logger = logging.getLogger(
            __name__ + "." + self.__class__.__name__ + f"_{id}"
        )
        self.relu = nn.ReLU()
        self.layer1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.MaxPool1d(2),
            nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.MaxPool1d(2),
            nn.Conv1d(
                in_channels=hidden_channels * 2,
                out_channels=hidden_channels * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.MaxPool1d(2),
            nn.Conv1d(
                in_channels=hidden_channels * 4,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.MaxPool1d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        identity = x
        self.logger.debug(f"Identity shape: {identity.shape}")
        x = self.layer1(x)
        self.logger.debug(f"Layer1 shape: {x.shape}")
        # add identity to x but keep the same shape
        identity = identity[
            :, :, : x.shape[2]
        ]  # adjust the shape of identity to match x
        x = x + identity
        self.logger.debug(f"Add shape: {x.shape}")
        x = x.flatten(1)
        self.logger.debug(f"flatten : {x.shape}")
        x = self.relu(x)
        self.logger.debug(f"ReLU shape: {x.shape}")
        x = self.layer2(x)
        self.logger.debug(f"Layer2 shape: {x.shape}")
        return x


class ResidualCNN(nn.Module):
    def __init__(self, **kwargs):
        input_features = kwargs.pop("input_features")
        hidden_features = kwargs.pop("hidden_features")
        output_features = kwargs.pop("output_features")
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        super(ResidualCNN, self).__init__(**kwargs)
        self.network = nn.Sequential(
            ResidualBlock(0, input_features, hidden_features, hidden_features),
            ResidualBlock(1, hidden_features, hidden_features, hidden_features),
            ResidualBlock(2, hidden_features, hidden_features, hidden_features),
            ResidualBlock(3, hidden_features, hidden_features, hidden_features),
        )
        self.linear = nn.Sequential(
            nn.Linear(hidden_features, output_features),
        )

    def forward(self, x):
        self.logger.debug(f"Input shape: {x.shape}")
        x = self.network(x)
        self.logger.debug(f"Network shape: {x.shape}")
        x = x.squeeze(1)
        self.logger.debug(f"Squeeze shape: {x.shape}")
        x = x.flatten(1)
        self.logger.debug(f"View shape: {x.shape}")
        x = self.linear(x)
        self.logger.debug(f"Linear shape: {x.shape}")
        return x
