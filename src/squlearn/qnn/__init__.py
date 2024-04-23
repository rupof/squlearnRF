"""QNN module for classification and regression."""
<<<<<<< HEAD
from .loss import SquaredLoss, VarianceLoss, ConstantLoss, ParameterRegularizationLoss, ODELoss
=======

from .loss import SquaredLoss, VarianceLoss, ConstantLoss, ParameterRegularizationLoss
>>>>>>> 0ae9430c3dcc019704e069dbf2bf9d5b718260b8
from .qnnc import QNNClassifier
from .qnnr import QNNRegressor
from .training import get_variance_fac, get_lr_decay, ShotsFromRSTD

__all__ = [
    "SquaredLoss",
    "VarianceLoss",
    "ConstantLoss",
    "ODELoss",
    "ParameterRegularizationLoss",
    "QNNClassifier",
    "QNNRegressor",
    "get_variance_fac",
    "get_lr_decay",
    "ShotsFromRSTD",
]
