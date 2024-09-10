from inspect import getmembers,  isclass, isabstract
import torch.nn as nn
from . import models

class ModelFactory(object):
    models = {}  # Key = model name, Value = class for the model

    def __init__(self):
        self.load_models()

    def load_models(self):
        classes = getmembers(models, lambda m: isclass(m))
        for name, _type in classes:
            if isclass(_type) and issubclass(_type, nn.Module):
                self.models.update([[name, _type]])

    def create_instance(self, cfg):
        modelname = cfg.get("name")
        if modelname in self.models:
            kwargs = cfg.get("kwargs")
            return self.models[modelname](**kwargs)
        else:
            raise ValueError("Model not found: ", modelname)