from .abs_preprocesser import AbsPreprocesser
import torch

class LinearRegressionPreprocesser(AbsPreprocesser):
    def data(self, df):
        data = torch.from_numpy(df[['x1', 'x2']].values).type(torch.float)
        return data
    def target(self, df):
        target = torch.from_numpy(df[['y']].values).type(torch.float)
        return target