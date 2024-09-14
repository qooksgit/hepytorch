import typing

from .abs_factory import AbsFactory
from .. import optimizers
from ..optimizers.abs_optimizer import AbsOptimizer
from .utils import get_instance


class OptimizerFactory(AbsFactory):
    @staticmethod
    def create_instance(cfg) -> typing.Any:
        return get_instance(AbsOptimizer, optimizers, cfg)
