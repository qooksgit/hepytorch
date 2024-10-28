from .abs_optimizer import AbsOptimizer
import torch.optim as optim


class Adam(AbsOptimizer):
    def __init__(self, **kwargs):
        self.lr = kwargs.pop("learning_rate", 1e-4)

    def get_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=self.lr)
