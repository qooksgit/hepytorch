import abc


class AbsTrainer(abc.ABC):
    @abc.abstractmethod
    def train(self, device, data, target, model, loss_fn, optimizer):
        pass
