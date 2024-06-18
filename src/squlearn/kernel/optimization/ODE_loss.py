"""Target alignment loss function for kernel matrices."""

import numpy as np

from typing import Sequence
from .kernel_loss_base import KernelLossBase
from ..matrix.kernel_matrix_base import KernelMatrixBase
from typing import Callable


class ODE_loss(KernelLossBase):
    """
    Target alignment loss function.
    This class can be used to compute the target alignment for a given quantum kernel
    :math:`K_{θ}` with variational parameters :math:`θ`.
    The definition of the function is taken from Equation (27,28) of [1].
    The log-likelihood function is defined as:

    .. math::

        TA(K_{θ}) =
        \\frac{\\sum_{i,j} K_{θ}(x_i, x_j) y_i y_j}
        {\\sqrt{\\sum_{i,j} K_{θ}(x_i, x_j)^2 \\sum_{i,j} y_i^2 y_j^2}}

    Args:
        quantum_kernel (KernelMatrixBase): The quantum kernel to be used
            (either a fidelity quantum kernel (FQK)
            or projected quantum kernel (PQK) must be provided).

    References
    -----------
        [1]: T. Hubregtsen et al.,
        "Training Quantum Embedding Kernels on Near-Term Quantum Computers",
        `arXiv:2105.02276v1 (2021) <https://arxiv.org/abs/2105.02276>`_.

    Methods:
    --------
    """

    def __init__(self, quantum_kernel: KernelMatrixBase, L_functional: Callable, regularization_parameter=1):
        super().__init__(quantum_kernel)
        self.L_functional = L_functional
        self.regularization_parameter = regularization_parameter
        self._cached_matrices = None

    def f_alpha_order(self, alpha_, kernel_tensor, order):
        """Calculates f_alpha.

        Args:
            alpha_ (np.ndarray): The vector of alphas, of shape (len(x_span)+1, 1).
            kernel_tensor (tuple): A tuple containing kernel objects for f_alpha_0 and f_alpha_1.
            order (int): Order of the kernel.

        Returns:
            np.ndarray: The vector of f_alphas, of shape (len(x_span), 1).
        """
        alpha = alpha_[1:]
        if order == 0:
            return np.dot(kernel_tensor[order], alpha) + alpha_[0]
        return np.dot(kernel_tensor[order], alpha) 
    
    def loss_function(self, alpha_, L_functional, f_initial, x_span, kernel_tensor):
        """Calculates the loss function.

        Args:
            alpha_ (np.ndarray): The vector of alphas, of shape (len(x_span)+1, 
            L_functional (function): The L functional, that describes the argument of the loss function. For example: L_functional = dfdx(alpha, K_1) - g(f(x, alpha, K_0), x)
            f_initial (np.ndarray): The initial value of the dependent variable.
            x_span (np.ndarray): The span of the independent variable.
            kernel_tensor (tuple): A tuple containing kernel objects for f_alpha_0 and f_alpha_1.
            
        Returns:
            float: The loss function.
        """
    
        f_alpha_tensor = np.array([self.f_alpha_order(alpha_, kernel_tensor, i) for i in range(len(kernel_tensor))])
        sum1 = np.sum((L_functional(f_alpha_tensor, x_span)**2)) #Functional
        sum2 = np.sum((f_alpha_tensor[:,0][:len(f_initial)] - f_initial)**2) #Initial condition
        L = sum2 + sum1 * self.regularization_parameter

        return L
    
    def create_kernel_L_functional(self, L_functional):
        def kernel_L_functional(f_alpha_tensor, x_span):
            return L_functional([x_span, *f_alpha_tensor])
        return kernel_L_functional
    
    def create_loss_function_with_derivatives_info(self, alpha_, L_functional, f_initial, x_span, kernel_tensor):
        def loss_function():
            return self.loss_function(alpha_, L_functional, f_initial, x_span, kernel_tensor)
        return loss_function

    def K_derivatives(self, X_train, y_initial):
        if self._cached_matrices is None:
                Kmatrix = self._quantum_kernel.evaluate(X_train)
                dKdx = self._quantum_kernel.evaluate_derivatives(X_train, evaluation_string="dfdx")
                if len(y_initial) > 1:
                    dKdxdx = self._quantum_kernel.evaluate_derivatives(X_train, evaluation_string="dfdxdx")
                else:
                    dKdxdx = np.zeros_like(dKdx[:,:])
                self._cached_matrices = (Kmatrix, dKdx, dKdxdx)
        else:
            Kmatrix, dKdx, dKdxdx = self._cached_matrices
        return Kmatrix, dKdx, dKdxdx
    
    def compute(
        self,
        alpha: float,
        data: np.ndarray,
        labels: np.ndarray,
        gate_parameter_values: Sequence[float],
    ) -> float:
        """Compute the target alignment.

        Args:
            parameter_values: (Sequence[float]):
                The parameter values for the variational quantum kernel parameters.
            data (np.ndarray): x_domain
            labels (np.ndarray): Corresponds to y_initial
    
        Returns:
            float: the ODE loss
        """
        # Bind training parameters
        self._quantum_kernel.assign_parameters(gate_parameter_values)
        Kmatrix, dKdx, dKdxdx = self.K_derivatives(data, labels)
        _L_functional = self.create_kernel_L_functional(self.L_functional)
        
        return self.loss_function(alpha, _L_functional, labels, data, [Kmatrix, dKdx, dKdxdx])

