from .abs_preprocessor import AbsPreprocessor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch


def normalize_data(data, mean, std):
    return (data - mean) / std

class PreprocessorV4E4A6(AbsPreprocessor):
    def __init__(self, **kwargs):
        self.train_true_mean = kwargs.pop("train_true_mean", 0.0)
        self.train_true_std = kwargs.pop("train_true_std", 1.0)


    def data(self, df):
        observed = df["observed_mass"].values
        observed = normalize_data(observed, self.train_true_mean, self.train_true_std)
        return  torch.tensor(observed, dtype=torch.float32).view(-1, 1)

    def target(self, df):
        true = df["true_mass"].values
        true = normalize_data(true, self.train_true_mean, self.train_true_std)
        return torch.tensor(true, dtype=torch.float32).view(-1, 1)
