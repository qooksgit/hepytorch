import abc


class AbsPreprocessor(abc.ABC):
    @abc.abstractmethod
    def data(self, data):
        pass

    @abc.abstractmethod
    def target(self, data):
        pass
