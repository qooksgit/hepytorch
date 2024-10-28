import torch.nn as nn

__all__ = ("SimpleNeuralNetwork",)


class SimpleNeuralNetwork(nn.Module):
    def __init__(self, **kwargs):
        input_features = kwargs.pop("input_features")
        hidden_features = kwargs.pop("hidden_features")
        output_features = kwargs.pop("output_features")
        dropout = kwargs.pop("dropout")
        super(SimpleNeuralNetwork, self).__init__(**kwargs)
        self.network = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_features),
            nn.ReLU(),
            nn.Linear(in_features=hidden_features, out_features=hidden_features),
            nn.ReLU(),
            nn.Linear(in_features=hidden_features, out_features=hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_features, out_features=output_features),
        )

    def forward(self, x):
        x = self.network(x)
        return x
