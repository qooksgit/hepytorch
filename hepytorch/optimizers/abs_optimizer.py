import abc


class AbsOptimizer(abc.ABC):
    @abc.abstractmethod
    def get_optimizer(self, model):
        pass
