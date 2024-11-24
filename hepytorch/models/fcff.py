import torch.nn as nn

__all__ = ("FullyConnectedFeedForward",)
# 2301.05707v2.model
# Machine Learning Assisted Vector Atomic Magnetometry


class FullyConnectedFeedForward(nn.Module):
    def __init__(self, **kwargs):
        input_features = kwargs.pop("input_features")
        hidden_features = kwargs.pop("hidden_features")
        output_features = kwargs.pop("output_features")
        super(FullyConnectedFeedForward, self).__init__(**kwargs)
        self.network = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_features),
            nn.Sigmoid(),
            nn.Linear(in_features=hidden_features, out_features=hidden_features),
            nn.ReLU(),
            nn.Linear(in_features=hidden_features, out_features=hidden_features),
            nn.ReLU(),
            nn.Linear(in_features=hidden_features, out_features=hidden_features),
            nn.ReLU(),
            nn.Linear(in_features=hidden_features, out_features=hidden_features),
            nn.ReLU(),
            nn.Linear(in_features=hidden_features, out_features=hidden_features),
            nn.ReLU(),
            nn.Linear(in_features=hidden_features, out_features=hidden_features),
            nn.ReLU(),
            nn.Linear(in_features=hidden_features, out_features=hidden_features),
            nn.ReLU(),

            nn.Linear(in_features=hidden_features, out_features=output_features),
        )

    def forward(self, x):
        x = self.network(x)
        return x
