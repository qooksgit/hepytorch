# HePytorch (a framework for experimental High energy physics based on Pytorch)

## Introduction
**HePytorch** is a general deep learning framework for experimental high-energy physics. 
It is built on top of PyTorch and provides a set of tools for training and evaluating deep learning models using the factory pattern[^1]. 
So you can easily build your own model and train it with your own dataset by adding a new class to the modules.

HePyporch is currently focusing on the following tasks:
- Estimation of the top quark mass based from the kinematic properties of the top quark decay products based on the Monte Carlo simulation
- B-jet matching between the reconstructed jets and the leptons from the top quark decay

## Installation
### Using Pip
With the option -e is used to install the package in editable mode, so you can modify the code and see the changes immediately without reinstalling the package. 

```
pip install -e -r requirements.txt
```

## Quick Start
You can start by creating a recipe file that contains the information of the data, model, loss function, optimizer, and trainer. 
Then you can instantiate the HEPTorch class with the recipe file and call the train method to train the model.
```jupyter
import hepytorch as hep
myHEP = hep.HEPTorch('./recipes/linear_regression.json') 
results = myHEP.train()
model = myHEP.model
```
### How to write a recipe
The recipe is a json file that contains the information of the data, model, loss function, optimizer, and trainer. Each module is defined by the name and the keyword arguments. **The name of the module should be the same as the class name in the modules**. So that the factory of the module can find the class and create an instance of the class with the keyword arguments. 


#### example of a recipe for linear regression
```
{
    "data": {
        "name": "CSVLoader",
        "kwargs": {
            "path": "examples/data/linear_regression_data.csv",
            "format": "csv"
        }
    },
    "preprocessor": {
        "name": "LinearRegressionPreprocessor",
        "kwargs": {}
    },
    "model": {
        "name": "LinearRegression",
        "kwargs": {
            "input_features": 2,
            "output_features": 1
        }
    },
    "loss_fn": {
        "name": "MSELoss",
        "kwargs": {}
    },
    "optimizer": {
        "name": "SGD",
        "kwargs": {
            "learning_rate": 1e-5,
            "momentum": 0.9
        }
    },
    "trainer": {
        "name": "BasicTrainer",
        "kwargs": {
            "epochs": 30,
            "batch_size": 10
        }
    }
}
```

### How to add new modules
You can add a new module by creating a new class in the modules directory. The class should inherit from the base class of the module.  And also you need to add the class to the \_\_all__ list in the \_\_init__.py file in the modules directory.

#### example of a new model 

1. hepytorch/models/linear_regression.py
    ```python
    # this file should be saved in the models directory
    import torch.nn as nn

    __all__ = ("LinearRegression",)


    class LinearRegression(nn.Module):
        def __init__(self, **kwargs):
            input_features = kwargs.pop("input_features")
            output_features = kwargs.pop("output_features")
            super(LinearRegression, self).__init__(**kwargs)
            self.dense_1 = nn.Linear(
                in_features=input_features, out_features=output_features
            )

        def forward(self, x):
            x = self.dense_1(x)
            return x
    ```

2. hepytorch/models/\_\_init__.py
   ```python
   from .liner_regression import LinearRegression
   __all__ = ["LinearRegression"]
   ```


For more information, please have a look subdirectories in the hepytorch directory.

## License
HePytorch is released under the MIT License. See the LICENSE file for more information. 

## References

[^1]: Erich Gamma, Richard Helm, Ralph Johnson, John Vlissides, Design patterns, software engineering, object-oriented programming (Addison-Wesley, 1994), p. 34.

