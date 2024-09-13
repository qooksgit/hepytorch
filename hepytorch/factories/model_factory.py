from torch import nn
import typing

from .abs_factory import AbsFactory
from .. import models
from .utils import get_instance


class ModelFactory(AbsFactory):
    @staticmethod
    def create_instance(cfg) -> typing.Any:
        return get_instance(nn.Module, models, cfg)
