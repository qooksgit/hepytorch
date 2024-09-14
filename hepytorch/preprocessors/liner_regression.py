from .abs_preprocessor import AbsPreprocessor
import torch


class LinearRegressionPreprocessor(AbsPreprocessor):
    def __init__(self, **kwargs):
        super(LinearRegressionPreprocessor, self).__init__(**kwargs)

    def data(self, df):
        data = torch.from_numpy(df[["x1", "x2"]].values).type(torch.float)
        return data

    def target(self, df):
        return torch.from_numpy(df["y"].values).type(torch.float).reshape(-1, 1)
