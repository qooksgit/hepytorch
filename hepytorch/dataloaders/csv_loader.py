from .abs_dataloader import AbsDataLoader
import pandas as pd


class CSVLoader(AbsDataLoader):
    def __init__(self, **kwargs):
        self.path = kwargs.pop("path")
        self.format = kwargs.pop("format")
        super(CSVLoader, self).__init__(**kwargs)

    def load_data(self):
        assert self.format == "csv", "Data format not supported"
        return pd.read_csv(self.path)