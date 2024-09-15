from .abs_lossfn import AbsLossFn
import torch.nn as nn


class MSELoss(AbsLossFn):
    def get_loss_fn(self):
        return nn.MSELoss()
