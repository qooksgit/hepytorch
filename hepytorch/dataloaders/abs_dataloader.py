import abc


class AbsDataLoader(abc.ABC):
    @abc.abstractmethod
    def load_data(self):
        pass
