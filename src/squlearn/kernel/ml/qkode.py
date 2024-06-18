"""Quantum Kernel ODE Solver"""

from ..matrix.kernel_matrix_base import KernelMatrixBase

import scipy
import numpy as np
from typing import Optional, Union, Callable
from sklearn.base import BaseEstimator, RegressorMixin

from ..matrix.regularization import thresholding_regularization, tikhonov_regularization
from ..optimization.ODE_loss import ODE_loss
from ..optimization.kernel_optimizer import KernelOptimizer
from ...optimizers.optimizer_base import OptimizerBase


class QKODE(BaseEstimator, RegressorMixin):
    """
    Quantum Kernel ODE Solver based on https://arxiv.org/abs/2203.08884

    This class implements the Quantum Kernel ODE Solver. It classically minimizes a loss function given by a differential equation.

    Args:

        quantum_kernel (Optional[Union[KernelMatrixBase, str]]) :
            The quantum kernel matrix to be used in the ODE pipeline (either a fidelity
            quantum kernel (FQK) or projected quantum kernel (PQK) must be provided). By
            setting quantum_kernel="precomputed" (NOT SUPPORTED YET), X is assumed to be a kernel matrix
            (train and test-train). This is particularly useful when storing quantum kernel
            matrices from real backends to numpy arrays.
        L_functional (Callable) :
            The loss function representing the differential equation to be solved.
        optimizer (OptimizerBase) :
            The optimizer to be used.

    Methods:
    --------
    """

    def __init__(
        self,
        quantum_kernel: Optional[Union[KernelMatrixBase, str]] = None,
        L_functional: Callable = None,
        optimizer: OptimizerBase = None,
        **kwargs,
    ) -> None:
        self._quantum_kernel = quantum_kernel
        self.X_train = None
        self.y_initial = None
        self.L_functional = L_functional
        self.optimizer = optimizer
        self.initial_parameters_classical = None
        self.kernel_optimizer = None
        self.ode_loss = None

        # Apply kwargs to set_params
        update_params = self.get_params().keys() & kwargs.keys()
        if update_params:
            self.set_params(**{key: kwargs[key] for key in update_params})

    def fit(self, X: np.ndarray, y_initial: np.ndarray, initial_parameters_classical: None, **kwargs):
        """
            Find the $alpha$ parameters of the model represented by the quantum kernel.

            Args:
                X (np.ndarray) : Samples of data of shape (n_samples, n_features) on which QKRR
                    model makes predictions. If quantum_kernel == "precomputed" this is instead a
                    precomputed (test-train) kernel matrix of shape (n_samples, n_samples_fitted),
                    where n_samples_fitted is the number of samples used in the fitting.
                y_initial (np.ndarray) : Initial values of the ODE to be solved.
                initial_parameters_classical (np.ndarray) : Initial parameters for the optimizer.
                **kwargs: Additional keyword arguments for the quantum kernel matrix.
            
            Returns:
                self : Returns an instance of self.
        """
        self.X_train = X
        self.y_initial = y_initial
        
        try:
            if X.shape[1] > 1:
                raise ValueError("Only one-dimensional ODEs are currently supported.")
        except: 
            pass

        if isinstance(self._quantum_kernel, KernelMatrixBase):
            self.ode_loss = ODE_loss(quantum_kernel=self._quantum_kernel, L_functional=self.L_functional, **kwargs)
            self.kernel_optimizer = KernelOptimizer(loss=self.ode_loss, optimizer=self.optimizer)
            self.kernel_optimizer.run_classical_optimization(X=self.X_train, y=y_initial, initial_parameters_classical=initial_parameters_classical)
        if isinstance(self._quantum_kernel, str):
            raise ValueError(
                "Precomputed kernel matrices are not supported yet."
            )
        else:
            raise ValueError(
                "Unknown type of quantum kernel: {}".format(type(self._quantum_kernel))
            )

        return self

    def predict(self) -> np.ndarray:
        """
        Predict using the QKODE.

        Returns:
            np.ndarray : The predicted values.

        """
        
        alphas = self.kernel_optimizer._opt_result.x
        kernel_tensor = self.ode_loss.K_derivatives(self.X_train, self.y_initial)

        return self.ode_loss.f_alpha_order(alphas, kernel_tensor, 0)

    def get_params(self, deep: bool = True) -> dict:
        """
        Returns hyperparameters and their values of the QKODE method.

        Args:
            deep (bool): If True, also the parameters for
                         contained objects are returned (default=True).

        Return:
            Dictionary with hyperparameters and values.
        """
        params = {
            "quantum_kernel": self._quantum_kernel,
        }

        if deep and isinstance(self._quantum_kernel, KernelMatrixBase):
            params.update(self._quantum_kernel.get_params(deep=deep))
        return params

    def set_params(self, **params) -> None:
        """
        Sets value of the encoding circuit hyperparameters.

        Args:
            params: Hyperparameters and their values, e.g. ``num_qubits=2``.
        """
        valid_params = self.get_params()
        for key in params.keys():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r}. "
                    f"Valid parameters are {sorted(valid_params)!r}."
                )

        # Set parameters of the QKRR
        self_params = self.get_params(deep=False).keys() & params.keys()
        for key in self_params:
            try:
                setattr(self, key, params[key])
            except AttributeError:
                setattr(self, "_" + key, params[key])

        if isinstance(self._quantum_kernel, KernelMatrixBase):
            # Set parameters of the Quantum Kernel and its underlying objects
            quantum_kernel_params = self._quantum_kernel.get_params().keys() & params.keys()
            if quantum_kernel_params:
                self._quantum_kernel.set_params(
                    **{key: params[key] for key in quantum_kernel_params}
                )
        return self
