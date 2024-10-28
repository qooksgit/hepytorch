from .abs_preprocessor import AbsPreprocessor
import numpy as np
import math
import pandas as pd
import torch


def phi(px, py):
    return np.arctan2(py, px)


def theta(px, py, pz):
    return np.arctan2(np.sqrt(px**2 + py**2), pz)


def eta(px, py, pz):
    return -1 * np.log(np.tan(theta(px, py, pz) / 2))


def dR(px1, py1, pz1, px2, py2, pz2):
    dphi = phi(px1, py1) - phi(px2, py2)
    # correct dphi if its absolute value is larger than pi
    if np.abs(dphi) > math.pi:
        dphi = 2 * math.pi - np.abs(dphi)
    deta = eta(px1, py1, pz1) - eta(px2, py2, pz2)
    return np.sqrt(dphi**2 + deta**2)


class JetSwapper(AbsPreprocessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mask = None

    #W_mass = (l_mass + nu)_invariant mass
    #  = (pl_x + pnu_x)^2 + .    (pl_y + pnu_y)^2 + .    (pl_z + pnu_z)^2 - .    (pl + pnu)^2

    def data(self, df):
        if self.mask is None:
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

            # match b and jet by the criterion of dR < 0.4
            # if dR (b, j2) < 0.4 then swap j_1 and j_2
            # and also if dR (b~, j1) < 0.4 and dR (b, j1) >= 0.4 and dR (b, j2) >= 0.4 then swap j_1 and j_2
            # finaly, remove the event if all dR (b, j1), dR (b, j2), dR (b~, j1), dR(b~,j2) are larger than 0.4

            observed = self.match(df, observed)
            observed = self.remove_entries(df, observed)

            # swap j_1 and j_2 by 50% chance and add new column containing boolean variable 'swapped' to the dataframe
            self.mask = np.random.choice([0, 1], observed.shape[0]) == 1
            temp = observed[self.mask].copy()
            observed.loc[self.mask, "j_1px"] = temp["j_2px"]
            observed.loc[self.mask, "j_1py"] = temp["j_2py"]
            observed.loc[self.mask, "j_1pz"] = temp["j_2pz"]
            observed.loc[self.mask, "j_1mass"] = temp["j_2mass"]
            observed.loc[self.mask, "j_2px"] = temp["j_1px"]
            observed.loc[self.mask, "j_2py"] = temp["j_1py"]
            observed.loc[self.mask, "j_2pz"] = temp["j_1pz"]
            observed.loc[self.mask, "j_2mass"] = temp["j_1mass"]
            self.data = observed.copy()

        data = torch.tensor(self.data.values).type(torch.float)

        return data

    def target(self, df):
        if self.mask is None:
            _ = self.data(df)
        return (
            torch.from_numpy(self.mask.astype("float")).type(torch.float).reshape(-1, 1)
        )

    def remove_entries(self, df, observed):
        # remove entries according to the condition of all dR >= 0.4
        removed = df.apply(
            lambda x: dR(
                x["abpx"], x["abpy"], x["abpz"], x["j_1px"], x["j_1py"], x["j_1pz"]
            )
            >= 0.4
            and dR(x["abpx"], x["abpy"], x["abpz"], x["j_2px"], x["j_2py"], x["j_2pz"])
            >= 0.4
            and dR(x["bpx"], x["bpy"], x["bpz"], x["j_1px"], x["j_1py"], x["j_1pz"])
            >= 0.4
            and dR(x["bpx"], x["bpy"], x["bpz"], x["j_2px"], x["j_2py"], x["j_2pz"])
            >= 0.4,
            axis=1,
        )
        return observed[~removed]

    def match(self, df, observed):
        swapped = df.apply(
            lambda x: dR(
                x["bpx"], x["bpy"], x["bpz"], x["j_2px"], x["j_2py"], x["j_2pz"]
            )
            < 0.4,
            axis=1,
        )

        swapped2 = df.apply(
            lambda x: dR(
                x["abpx"], x["abpy"], x["abpz"], x["j_1px"], x["j_1py"], x["j_1pz"]
            )
            < 0.4
            and dR(x["bpx"], x["bpy"], x["bpz"], x["j_1px"], x["j_1py"], x["j_1pz"])
            >= 0.4
            and dR(x["bpx"], x["bpy"], x["bpz"], x["j_2px"], x["j_2py"], x["j_2pz"])
            >= 0.4,
            axis=1,
        )
        swapped = swapped | swapped2
        temp = observed[swapped].copy()
        observed.loc[swapped, "j_1px"] = temp["j_2px"]
        observed.loc[swapped, "j_1py"] = temp["j_2py"]
        observed.loc[swapped, "j_1pz"] = temp["j_2pz"]
        observed.loc[swapped, "j_1mass"] = temp["j_2mass"]
        observed.loc[swapped, "j_2px"] = temp["j_1px"]
        observed.loc[swapped, "j_2py"] = temp["j_1py"]
        observed.loc[swapped, "j_2pz"] = temp["j_1pz"]
        observed.loc[swapped, "j_2mass"] = temp["j_1mass"]
        observed["dR_j1_l1"] = df.apply(
            lambda x: dR(
                x["j_1px"], x["j_1py"], x["j_1pz"], x["l_1px"], x["l_1py"], x["l_1pz"]
            ),
            axis=1,
        )
        observed["dR_j1_l2"] = df.apply(
            lambda x: dR(
                x["j_1px"], x["j_1py"], x["j_1pz"], x["l_2px"], x["l_2py"], x["l_2pz"]
            ),
            axis=1,
        )
        observed["dR_j2_l1"] = df.apply(
            lambda x: dR(
                x["j_2px"], x["j_2py"], x["j_2pz"], x["l_1px"], x["l_1py"], x["l_1pz"]
            ),
            axis=1,
        )
        observed["dR_j2_l2"] = df.apply(
            lambda x: dR(
                x["j_2px"], x["j_2py"], x["j_2pz"], x["l_2px"], x["l_2py"], x["l_2pz"]
            ),
            axis=1,
        )
        return observed
