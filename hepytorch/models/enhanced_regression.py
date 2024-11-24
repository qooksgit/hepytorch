import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ("EnhancedRegression",)


class EnhancedRegression(nn.Module):
    def __init__(self, **kwargs):
        input_features = kwargs.pop("input_features")
        output_features = kwargs.pop("output_features")
        super(EnhancedRegression, self).__init__(**kwargs)
        self.fc1 = nn.Linear(input_features, 1024)
        self.fc1_residual = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc3_residual = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, output_features)

    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        x1_residual = self.fc1_residual(x1)
        x2 = F.relu(self.fc2(x1) + x1_residual)
        x3 = F.relu(self.fc3(x2) + self.fc3_residual(x2))
        output = self.fc4(x3)
        return output
