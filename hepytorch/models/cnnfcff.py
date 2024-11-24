import torch.nn as nn
import logging

__all__ = ("ConvolutionalNeuralNetworkFullyConnectedFeedForward",)
# 2305.01852v1
# Σ Resonances from a Neural Network-based Partial Wave Analysis on K−p Scattering

# input [nSamples]*1*8
# conv1d 1*16
# pooling (2)
# conv1d 16*64
# pooling (2)
# conv1d 64*128
# polling (2)
# flatten -> 128
# fcff 128 -> 64
# relu
# fcff 64 -> 32
# relu
# fcff 32 -> 16
# relu
# fcff 16 -> 1


class ConvolutionalNeuralNetworkFullyConnectedFeedForward(nn.Module):
    def __init__(self, **kwargs):
        input_features = kwargs.pop("input_features")
        batch_size = kwargs.pop("batch_size")
        hidden_features = kwargs.pop("hidden_features")
        output_features = kwargs.pop("output_features")
        super(ConvolutionalNeuralNetworkFullyConnectedFeedForward, self).__init__(
            **kwargs
        )
        self.logger = logging.getLogger("model")
        self.network = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.MaxPool1d(2),
            nn.Conv1d(
                in_channels=16,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.MaxPool1d(2),
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.MaxPool1d(2),
        )
        self.linear = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):

        self.logger.debug(f"Input shape: {x.shape}")
        x = x.unsqueeze(1)
        self.logger.debug(f"Input shape after unsqueeze: {x.shape}")
        x = self.network(x)
        self.logger.debug(f"Output shape after network: {x.shape}")
        self.logger.debug(f"Output shape view: {x.view(x.size(0), -1).shape}")
        x = x.view(x.size(0), -1)
        self.logger.debug(f"Output shape after view: {x.shape}")
        x = self.linear(x)
        self.logger.debug(f"Output shape after linear: {x.shape}")

        return x
