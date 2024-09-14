import abc


class AbsLossFn(abc.ABC):
    @abc.abstractmethod
    def get_loss_fn(self):
        pass
