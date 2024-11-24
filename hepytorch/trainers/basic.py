from .abs_trainer import AbsTrainer
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, X, y):
        assert X.size()[0] == y.size()[0]
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.size()[0]

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]


class BasicTrainer(AbsTrainer):
    def __init__(self, **kwargs):
        self.batch_size = kwargs.pop("batch_size", 4)
        self.epochs = kwargs.pop("epochs", 10)
        self.regularization_type = kwargs.pop("regularization_type", None)
        self.lambda_reg = kwargs.pop("lambda_reg", 0.00001)

    def train(self, device, data, target, model, loss_fn, optimizer):
        data = data.to(device)
        target = target.to(device)
        train_data = DataLoader(
            MyDataset(data, target), batch_size=self.batch_size, shuffle=True
        )
        num_examples = len(train_data.dataset)
        losses = []
        model.to(device)
        for e in range(self.epochs):
            cumulative_loss = 0
            # inner loop
            for _i, (data2, label) in enumerate(train_data):
                data2 = data2.to(device)
                label = label.to(device)
                yhat = model(data2)
                loss = loss_fn(yhat, label)

                # Apply L1 regularization
                if self.regularization_type == "L1":
                    l1_norm = sum(p.abs().sum() for p in model.parameters())
                    loss += self.lambda_reg * l1_norm
                # Apply L2 regularization
                elif self.regularization_type == "L2":
                    l2_norm = sum(p.pow(2).sum() for p in model.parameters())
                    loss += self.lambda_reg * l2_norm

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                cumulative_loss += loss.item()
            print("Epoch %s, loss: %s" % (e, cumulative_loss / num_examples))
            losses.append(cumulative_loss / num_examples)
        return {"model": model, "losses": losses}
