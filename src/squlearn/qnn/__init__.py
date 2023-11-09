"""QNN module for classification and regression."""
from .loss import SquaredLoss, VarianceLoss, ConstantLoss, ParameterRegularizationLoss
from .qnnc import QNNClassifier
from .qnnr import QNNRegressor
from .training import get_variance_fac, get_lr_decay, ShotsFromRSTD

__all__ = [
    "SquaredLoss",
    "VarianceLoss",
    "ConstantLoss",
    "ParameterRegularizationLoss",
    "QNNClassifier",
    "QNNRegressor",
    "get_variance_fac",
    "get_lr_decay",
    "ShotsFromRSTD",
]
