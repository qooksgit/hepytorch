import typing

from .abs_factory import AbsFactory
from .. import loss_functions
from ..loss_functions.abs_lossfn import AbsLossFn
from .utils import get_instance


class LossFnFactory(AbsFactory):
    @staticmethod
    def create_instance(cfg) -> typing.Any:
        return get_instance(AbsLossFn, loss_functions, cfg)
