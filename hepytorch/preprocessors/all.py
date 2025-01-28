from .abs_preprocessor import AbsPreprocessor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch


class AllPreprocessor(AbsPreprocessor):

    def data(self, df):
        import numpy as np

        # columns to drop
        cols = [
            "topmass",
            "atopmass",
        ]
        observed = df.drop(columns=cols)
        pd.options.mode.copy_on_write = True

        def calculate_energy(px, py, pz, mass):
            return np.sqrt(px**2 + py**2 + pz**2 + mass**2)

        # Calculate energies using vectorized operations
        observed["j_1e"] = calculate_energy(
            observed["j_1px"], observed["j_1py"], observed["j_1pz"], observed["j_1mass"]
        )
        observed["j_2e"] = calculate_energy(
            observed["j_2px"], observed["j_2py"], observed["j_2pz"], observed["j_2mass"]
        )
        observed["l_1e"] = calculate_energy(
            observed["l_1px"], observed["l_1py"], observed["l_1pz"], observed["l_1mass"]
        )
        observed["l_2e"] = calculate_energy(
            observed["l_2px"], observed["l_2py"], observed["l_2pz"], observed["l_2mass"]
        )

        observed["ex"] = (
            observed["j_1px"]
            + observed["j_2px"]
            + observed["l_1px"]
            + observed["l_2px"]
        )
        # mex = observed["nu_px"]+ observed["nu~_px"]
        # mey = observed["nu_px"]+ observed["nu~_px"]
        # ex = j1_px + j2_px + l1_px + l2_px + mex = 0
        # ey = j1_py + j2_py + l1_py + l2_py + mey = 0
        # mex = ex - (j1_px + j2_px + l1_px + l2_px)
        # mey = ey - (j1_py + j2_py + l1_py + l2_py)

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
        target = df["topmass"].copy()

        return torch.from_numpy(target.values).type(torch.float).reshape(-1, 1)
