from .abs_trainer import AbsTrainer
from sklearn.model_selection import StratifiedKFold
import copy
import tqdm
import torch
import torch.nn as nn
import numpy as np


@torch.no_grad()
def weight_reset(m: nn.Module):
    # - check if the current module has reset_parameters & if it's callabed called it on m
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


class KFoldCrossValidationTrainer(AbsTrainer):
    def __init__(self, **kwargs):
        self.batch_size = kwargs.pop("batch_size", 4)
        self.epochs = kwargs.pop("epochs", 10)
        self.k = kwargs.pop("k", 5)

    def _model_train(self, model, X_train, y_train, X_test, y_test, loss_fn, optimizer):
        model.apply(weight_reset)
        batch_start = torch.arange(0, len(X_train), self.batch_size)
        best_acc = -np.inf
        best_weights = None
        losses = []
        for epoch in range(self.epochs):
            model.train()
            with tqdm.tqdm(
                batch_start, unit="batch", mininterval=0, disable=True
            ) as bar:
                bar.set_description(f"Epoch {epoch}")
                for start in bar:
                    X_batch = X_train[start : start + self.batch_size]
                    y_batch = y_train[start : start + self.batch_size]
                    y_pred = model(X_batch)
                    loss = loss_fn(y_pred, y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    acc = (y_pred.round() == y_batch).float().mean().item()
                    bar.set_postfix(loss=loss.item(), acc=acc)
            losses.append(loss.item())
            model.eval()
            y_pred = model(X_test)
            acc = (y_pred.round() == y_test).float().mean().item()
            acc = float(acc)
            if acc > best_acc:
                best_acc = acc
                best_weights = copy.deepcopy(model.state_dict())
            print(f"epoch {epoch}, Accuracy: {acc}, Loss: {loss.item()}")
        model.load_state_dict(best_weights)
        return best_acc, losses

    def train(self, device, data, target, model, loss_fn, optimizer):
        data = data.to(device)
        target = target.to(device)
        losses = []
        model.to(device)

        kfold = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=42)
        cv_scores = []
        loss_collection = []
        for train, test in kfold.split(data, target):
            acc, losses = self._model_train(
                model,
                data[train],
                target[train],
                data[test],
                target[test],
                loss_fn,
                optimizer,
            )
            print("Accuracy: %s" % acc)
            cv_scores.append(acc)
            loss_collection.append(losses)
        return {"model": model, "losses": loss_collection, "cv_scores": cv_scores}
