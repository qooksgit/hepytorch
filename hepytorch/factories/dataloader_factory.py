import typing

from .abs_factory import AbsFactory
from .. import dataloaders
from ..dataloaders.abs_dataloader import AbsDataLoader
from .utils import get_instance


class DataLoaderFactory(AbsFactory):
    @staticmethod
    def create_instance(cfg) -> typing.Any:
        return get_instance(AbsDataLoader, dataloaders, cfg)
