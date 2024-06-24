"""Target alignment loss function for kernel matrices."""

import numpy as np

from typing import Sequence
from .kernel_loss_base import KernelLossBase
from ..matrix.kernel_matrix_base import KernelMatrixBase
from typing import Callable


class ODE_loss(KernelLossBase):
    """
    Ordinary Differential Equation (ODE) loss function for quantum kernels.

    This class implements the ODE loss function for quantum kernels. The ODE loss function is defined as:
    .. math::
        \mathcal{L} = \sum_{i=1}^{N} L\left(\ddot{f}(x_i, \alpha), \dot{f}(x_i, \alpha), f(x_i, \alpha), x_i \right) + \lambda \sum_{i=1}^{N} \left(f(x_i, \alpha) - y_i\right)^2 
    Args:
        quantum_kernel (KernelMatrixBase): The quantum kernel to be used
            (either a fidelity quantum kernel (FQK)
            or projected quantum kernel (PQK) must be provided).

    References
    -----------
        [1]: Paine. et al.,
        "Quantum Kernel Methods for Solving Differential Equations",
        `arXiv:2203.08884 (2023) <https://arxiv.org/abs/2203.08884>`_.

    Methods:
    --------
    """

    def __init__(self, quantum_kernel: KernelMatrixBase, L_functional: Callable, regularization_parameter=1):
        super().__init__(quantum_kernel)
        self.L_functional = L_functional
        self.regularization_parameter = regularization_parameter
        self._cached_matrices = None

    def f_alpha_order(self, alpha_, kernel_tensor, order):
        """Calculates the ansatz $f_{\alpha}$ build from the kernel, see Equation (1). $\alpha$ is the vector of parameters. 
        Args:
            alpha_ (np.ndarray): The vector of alphas, of shape (len(x_span)+1, 1).
            kernel_tensor (tuple): A tuple containing kernel objects for f_alpha_0 and f_alpha_1.
            order (int): Order of the kernel. For example, order=0 corresponds to $f_{\alpha}$ and order=1 corresponds to $df_{\alpha}/dx$.

        Returns:
            np.ndarray: The vector of f_alphas, of shape (len(x_span), 1).
        """
        alpha = alpha_[1:]
        if order == 0:
            return np.dot(kernel_tensor[order], alpha) + alpha_[0]
        return np.dot(kernel_tensor[order], alpha) 
    
    def loss_function(self, alpha_, L_functional, f_initial, x_span, kernel_tensor):
        """Calculates the loss function, using the L functional, the initial condition and the regularization parameter.

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
        """Creates a wrapper function for the L functional, that takes in the kernel tensor and the x_span as arguments.

        Args:
            L_functional (function): The L functional, that describes the argument of the loss function. For example: L_functional = dfdx(alpha, K_1) - g(f(x, alpha, K_0), x)
        Returns:
            function: a function L_functional, that takes in the kernel tensor and the x_span as arguments.
        """
        def kernel_L_functional(f_alpha_tensor, x_span):
            return L_functional([x_span, *f_alpha_tensor])
        return kernel_L_functional
    

    def K_derivatives(self, X_train, y_initial):
        """Computes the kernel matrix and its derivatives.
        Args:
            X_train (np.ndarray): The input data.
            y_initial (np.ndarray): The labels.
        Returns:    
            tuple: A tuple containing the kernel matrix and its derivatives.
        """
        if self._cached_matrices is None:
                Kmatrix = self._quantum_kernel.evaluate(X_train)
                dKdx = self._quantum_kernel.evaluate_derivatives(X_train, evaluation_string="dKdx")
                if len(y_initial) > 1:
                    dKdxdx = self._quantum_kernel.evaluate_derivatives(X_train, evaluation_string="dKdxdx")
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
        """Compute the ODE loss function.

        Args:
            alpha (float): The regularization parameter.
            data (np.ndarray): The input data.
            labels (np.ndarray): The labels.
            gate_parameter_values (Sequence[float]): The gate parameter values.
    
        Returns:
            float: the ODE loss
        """
        # Bind training parameters
        self._quantum_kernel.assign_parameters(gate_parameter_values)
        Kmatrix, dKdx, dKdxdx = self.K_derivatives(data, labels)
        _L_functional = self.create_kernel_L_functional(self.L_functional)
        
        return self.loss_function(alpha, _L_functional, labels, data, [Kmatrix, dKdx, dKdxdx])

