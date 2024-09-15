import typing

from .abs_factory import AbsFactory
from .. import trainers
from ..trainers.abs_trainer import AbsTrainer
from .utils import get_instance


class TrainerFactory(AbsFactory):
    @staticmethod
    def create_instance(cfg) -> typing.Any:
        return get_instance(AbsTrainer, trainers, cfg)
