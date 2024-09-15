import typing

from .abs_factory import AbsFactory
from .. import preprocessors
from ..preprocessors.abs_preprocessor import AbsPreprocessor
from .utils import get_instance


class PreprocessorFactory(AbsFactory):
    @staticmethod
    def create_instance(cfg) -> typing.Any:
        return get_instance(AbsPreprocessor, preprocessors, cfg)
