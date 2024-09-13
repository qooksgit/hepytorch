import json
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from .factories import model_factory, dataloader_factory

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
        with open(config, "r") as f:
            self.config = json.load(f)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dataloader = dataloader_factory.DataLoaderFactory().create_instance(
            self.config.get("data")
        )
        self.data = dataloader.load_data()
        # self.data = self._load_data()
        self.model = model_factory.ModelFactory().create_instance(
            self.config.get("model")
        )
        self.loss_fn = self._construct_loss_fn()
        self.optimizer = self._construct_optimizer()

    def _load_data(self):
        cfg = self.config.get("data")
        path = cfg.get("path")
        format = cfg.get("format")
        match format:
            case "csv":
                data = pd.read_csv(path)
            case "json":
                data = pd.read_json(path)
            case "pickle":
                data = pd.read_pickle(path)
            case "parquet":
                data = pd.read_parquet(path)
            case _:
                raise ValueError("Data format not found: ", format)
        return data

    def _construct_loss_fn(self):
        cfg = self.config.get("loss_fn")
        loss_fn_name = cfg.get("name")
        match loss_fn_name:
            case "MSELoss":
                return torch.nn.MSELoss()
            case _:
                raise ValueError("Loss function not found: ", loss_fn_name)

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
        data = torch.from_numpy(self.data[["x1", "x2"]].values).type(torch.float)
        data = data.to(self.device)
        target = (
            torch.from_numpy(self.data["y"].values).type(torch.float).reshape(-1, 1)
        )
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
