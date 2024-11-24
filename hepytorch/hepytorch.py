import json
import logging
import torch.cuda as cuda
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
        handle = "hepytorch"
        logger = logging.getLogger(handle)
        self.device = "cuda" if cuda.is_available() else "cpu"
        logger.info(f"Using {self.device} device")
        with open(config, "r") as f:
            self.config = json.load(f)
        logger.info(
            "Using the following configuration\n" + json.dumps(self.config, indent=2)
        )

        dataloader = DataLoaderFactory().create_instance(self.config.get("data"))
        data = dataloader.load_data()
        logger.info(f"Data shape: {data.shape}")
        logger.info(f"Data columns: {data.columns}")
        self.preprocessor = PreprocessorFactory().create_instance(
            self.config.get("preprocessor")
        )
        self.data = self.preprocessor.data(data)
        logger.info(f"Data shape after preprocessing: {self.data.shape}")
        self.target = self.preprocessor.target(data)
        logger.info(f"Target shape after preprocessing: {self.target.shape}")
        self.model = ModelFactory().create_instance(self.config.get("model"))
        logger.info(f"Model: {self.model}")
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
