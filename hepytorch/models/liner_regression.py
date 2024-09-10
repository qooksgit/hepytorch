import torch as T
import torch.nn as nn

__all__ = (
    'LinearRegression',
)
class LinearRegression(nn.Module):
    def __init__(self, **kwargs):
        input_features = kwargs.pop('input_features')
        output_features = kwargs.pop('output_features')
        super(LinearRegression, self).__init__(**kwargs)
        self.dense_1 = T.nn.Linear(in_features=input_features, out_features=output_features)
        
    def forward(self, x):
        x = self.dense_1(x)
        return x