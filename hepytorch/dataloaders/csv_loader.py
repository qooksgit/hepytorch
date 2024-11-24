from .abs_dataloader import AbsDataLoader
import pandas as pd
import logging


class CSVLoader(AbsDataLoader):
    def __init__(self, **kwargs):
        self.path = kwargs.pop("path")
        self.format = kwargs.pop("format")
        assert self.format == "csv", "Data format not supported"
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        super(CSVLoader, self).__init__(**kwargs)

    def load_data(self):
        self.logger.debug(f"Data loaded from {self.path}")
        df = pd.read_csv(self.path)
        self.logger.debug(f"Data shape: {df.shape}")
        self.logger.debug(f"Data columns: {df.columns}")

        return pd.read_csv(self.path)
