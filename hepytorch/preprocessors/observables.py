from .abs_preprocessor import AbsPreprocessor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch
import logging


class ObservablesPreprocessor(AbsPreprocessor):
    def __init__(self):
        super(ObservablesPreprocessor, self).__init__()
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)

    def data(self, df):
        col = [
            "j_1px",
            "j_1py",
            "j_1pz",
            "j_1mass",
            "l_1px",
            "l_1py",
            "l_1pz",
            "l_1mass",
            "j_2px",
            "j_2py",
            "j_2pz",
            "j_2mass",
            "l_2px",
            "l_2py",
            "l_2pz",
            "l_2mass",
            "mex",
            "mey",
        ]
        pd.options.mode.copy_on_write = True
        observed = df[col].copy()
        self.mask = (observed.iloc[:, 7] < 1) & (
            observed.iloc[:, 15] < 1
        )  # filter that l1 mass and l2 mass should be less that 1
        observed = observed[self.mask]
        observed["ex"] = (
            observed["j_1px"]
            + observed["j_2px"]
            + observed["l_1px"]
            + observed["l_2px"]
        )

        observed["ey"] = (
            observed["j_1py"]
            + observed["j_2py"]
            + observed["l_1py"]
            + observed["l_2py"]
        )
        dataset = StandardScaler().fit_transform(observed)
        data = torch.from_numpy(dataset).type(torch.float)
        return data

    def target(self, df):
        pd.options.mode.copy_on_write = True
        col = ["topmass", "atopmass"]
        target = df[col].copy()
        target = target[self.mask]
        target = StandardScaler().fit_transform(target)
        out = torch.from_numpy(target).type(torch.float).reshape(-1, 2)
        return out
