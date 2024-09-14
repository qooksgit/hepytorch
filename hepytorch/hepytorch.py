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

__all__ = ("HEPTorch",)


# TODO : error handling and test
class LinearDataset(Dataset):
    def __init__(self, X, y):
        assert X.size()[0] == y.size()[0]
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.size()[0]

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]


class HEPTorch:
    def __init__(self, config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        with open(config, "r") as f:
            self.config = json.load(f)
        dataloader = DataLoaderFactory().create_instance(self.config.get("data"))
        self.data = dataloader.load_data()
        self.preprocessor = PreprocessorFactory().create_instance(
            self.config.get("preprocessor")
        )

        self.model = ModelFactory().create_instance(self.config.get("model"))

        loss_fn = LossFnFactory().create_instance(self.config.get("loss_fn"))
        self.loss_fn = loss_fn.get_loss_fn()
        optimizer = OptimizerFactory().create_instance(self.config.get("optimizer"))
        self.optimizer = optimizer.get_optimizer(self.model)

    def _construct_optimizer(self):
        cfg = self.config.get("optimizer")
        optimizer_name = cfg.get("name")
        lr = cfg.get("learning_rate", 1e-5)
        momentum = cfg.get("momentum", 0.9)
        match optimizer_name:
            case "SGD":
                return optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
            case _:
                raise ValueError("Optimizer not found: ", optimizer_name)

    def train(self):
        cfg = self.config.get("train")
        batch_size = cfg.get("batch_size", 4)
        epochs = cfg.get("epochs", 10)
        data = self.preprocessor.data(self.data)
        data = data.to(self.device)
        target = self.preprocessor.target(self.data)
        target = target.to(self.device)
        train_data = DataLoader(
            LinearDataset(data, target), batch_size=batch_size, shuffle=True
        )
        num_examples = len(train_data.dataset)
        losses = []
        self.model.to(self.device)
        for e in range(epochs):
            cumulative_loss = 0
            # inner loop
            for i, (data, label) in enumerate(train_data):
                data = data.to(self.device)
                label = label.to(self.device)
                yhat = self.model(data)
                loss = self.loss_fn(yhat, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                cumulative_loss += loss.item()
            print("Epoch %s, loss: %s" % (e, cumulative_loss / num_examples))
            losses.append(cumulative_loss / num_examples)
        return {"model": self.model, "losses": losses}
