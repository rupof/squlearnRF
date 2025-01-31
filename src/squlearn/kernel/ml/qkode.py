"""Quantum Kernel ODE"""

from ..matrix.kernel_matrix_base import KernelMatrixBase
from ..ml.qkrr import QKRR
from ..loss.kernel_loss_base import KernelLossBase
from ...optimizers.optimizer_base import OptimizerBase


import scipy
import numpy as np
from typing import Optional, Union
from functools import partial
from sklearn.base import BaseEstimator


class QKODE(QKRR):
    r"""
    Quantum Kernel Ordinary Differential Equation (QKODE) solver.

    This class implements a quantum kernel-based solver for ordinary differential equations
    using ridge regression. It extends the Quantum Kernel Ridge Regression (QKRR) model.

    Args:
        quantum_kernel (Optional[Union[KernelMatrixBase, str]]) :
            The quantum kernel matrix to be used in the KRR pipeline (either a fidelity
            quantum kernel (FQK) or projected quantum kernel (PQK) must be provided). By
            setting quantum_kernel="precomputed", X is assumed to be a kernel matrix
            (train and test-train). This is particularly useful when storing quantum kernel
            matrices from real backends to numpy arrays.
        loss (KernelLossBase) :
            The loss function to be minimized.
        alpha (Union[float, np.ndarray], default=1.0e-6) :
            Hyperparameter for the regularization strength; must be a positive float. This
            regularization improves the conditioning of the problem and assure the solvability
            of the resulting linear system. Larger values specify stronger regularization, cf.,
            e.g., Ref. [2]
        optimizer (OptimizerBase) :
            The optimizer to be used.
        **kwargs: Keyword arguments for the quantum kernel matrix, possible arguments can be obtained
            by calling ``get_params()``. 
     
     Attributes:
    -----------
        dual_coeff\_ : (np.ndarray) :
            Array containing the weight vector in kernel space
        k_train (np.ndarray) :
            Training kernel matrix of shape (n_train, n_train) which is available after calling the fit procedure
        k_testtrain (np.ndarray) :
            Kernel matrix of shape (n_test, n_train) which is evaluated at the predict step

    
    Methods:
    --------
    """

    def __init__(
        self,
        quantum_kernel: Optional[Union[KernelMatrixBase, str]] = None,
        loss: KernelLossBase = None,
        alpha: Union[float, np.ndarray] = 1.0e-6,
        optimizer: OptimizerBase = None,
        **kwargs,
    ) -> None:
        super().__init__(quantum_kernel=quantum_kernel, alpha=alpha, **kwargs)
        self._loss = loss
        self._loss.set_quantum_kernel(quantum_kernel)
        self._optimizer = optimizer

        

    def fit(self, X, y, param_ini = None):
        """
        
        """
        X, y = self._validate_data(
            X, y, accept_sparse=("csr", "csc"), multi_output=True, y_numeric=True
        )
        self.X_train = X

        # set up kernel matrix
        if isinstance(self._quantum_kernel, str):
            if self._quantum_kernel == "precomputed":
                self.k_train = X
            else:
                raise ValueError("Unknown quantum kernel: {}".format(self._quantum_kernel))
        elif isinstance(self._quantum_kernel, KernelMatrixBase):
            # check if quantum kernel is trainable
            if self._quantum_kernel.is_trainable:
                self._quantum_kernel.run_optimization(self.X_train, y)

            self.k_train = self._quantum_kernel.evaluate(x=self.X_train)  # set up kernel matrix
        else:
            raise ValueError(
                "Unknown type of quantum kernel: {}".format(type(self._quantum_kernel))
            )

        if param_ini is None:
            param_ini = np.random.rand(len(y)+1)

        loss_function = partial(self._loss.compute, data=X, labels=y)
        opt_result = self._optimizer.minimize(fun=loss_function, x0=param_ini)
        self.dual_coeff_ = opt_result.x    
        self._is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the Quantum Kernel Ridge model.

        Args:
            X (np.ndarray) : Samples of data of shape (n_samples, n_features) on which QKRR
                model makes predictions. If quantum_kernel == "precomputed" this is instead a
                precomputed (test-train) kernel matrix of shape (n_samples, n_samples_fitted),
                where n_samples_fitted is the number of samples used in the fitting.

        Returns:
            np.ndarray :
                Returns predicted labels (at X) of shape (n_samples,)
        """
        if self.k_train is None:
            raise ValueError("The fit() method has to be called beforehand.")

        X = self._validate_data(X, accept_sparse=("csr", "csc"), reset=False)

        if isinstance(self._quantum_kernel, str):
            if self._quantum_kernel == "precomputed":
                self.k_testtrain = X
        elif isinstance(self._quantum_kernel, KernelMatrixBase):
            self.k_testtrain = self._quantum_kernel.evaluate(x=X, y=self.X_train)
        else:
            raise ValueError(
                "Unknown type of quantum kernel: {}".format(type(self._quantum_kernel))
            )

        prediction = np.dot(self.k_testtrain, self.dual_coeff_[1:]) + self.dual_coeff_[0]
        return prediction
