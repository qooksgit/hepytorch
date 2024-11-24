from .abs_lossfn import AbsLossFn
import torch.nn as nn
import torch


def weighted_mse_loss(y_true, y_pred, weights=None):
    residual = (y_true - y_pred) ** 2
    if weights is None:
        weights = 1 + residual
    return torch.mean(weights * residual)


class CustomWeightedMSELoss(AbsLossFn):
    def get_loss_fn(self):
        return weighted_mse_loss
