import json
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from .factories.model_factory import ModelFactory
from .factories.dataloader_factory import DataLoaderFactory
from .factories.preprocessor_factory import PreprocessorFactory
from .factories.lossfn_factory import LossFnFactory
from .factories.optimizer_factory import OptimizerFactory
from .factories.trainer_factory import TrainerFactory

__all__ = ("HEPTorch",)


# TODO : error handling and test


class HEPTorch:
    def __init__(self, config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        with open(config, "r") as f:
            self.config = json.load(f)
        dataloader = DataLoaderFactory().create_instance(self.config.get("data"))
        data = dataloader.load_data()
        self.preprocessor = PreprocessorFactory().create_instance(
            self.config.get("preprocessor")
        )
        self.data = self.preprocessor.data(data)
        self.target = self.preprocessor.target(data)
        self.model = ModelFactory().create_instance(self.config.get("model"))
        loss_fn = LossFnFactory().create_instance(self.config.get("loss_fn"))
        self.loss_fn = loss_fn.get_loss_fn()
        optimizer = OptimizerFactory().create_instance(self.config.get("optimizer"))
        self.optimizer = optimizer.get_optimizer(self.model)
        self.trainer = TrainerFactory().create_instance(self.config.get("trainer"))

    def train(self):
        return self.trainer.train(
            self.device,
            self.data,
            self.target,
            self.model,
            self.loss_fn,
            self.optimizer,
        )
