from .abs_optimizer import AbsOptimizer
import torch.optim as optim


class SGD(AbsOptimizer):
    def __init__(self, **kwargs):
        self.lr = kwargs.pop("learning_rate", 1e-5)
        self.momentum = kwargs.pop("momentum", 0.9)

    def get_optimizer(self, model):
        return optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum)
