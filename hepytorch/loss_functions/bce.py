from .abs_lossfn import AbsLossFn
import torch.nn as nn


class BCELoss(AbsLossFn):
    def get_loss_fn(self):
        return nn.BCELoss()
