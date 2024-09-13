import abc
import typing


class AbsFactory(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def create_instance(cfg) -> typing.Any:
        pass
