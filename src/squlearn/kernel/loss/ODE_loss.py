""" Negative log likelihood loss function"""

import scipy
import numpy as np
import sympy as sp

from typing import Sequence
from .kernel_loss_base import KernelLossBase
from ..matrix.kernel_matrix_base import KernelMatrixBase


class ODELoss(KernelLossBase):
    r"""
    Ordinary Differential Equation (ODE) loss function for Quantum Kernels.

    This class implements the ODE loss function for Quantum Kernels. The ODE loss function is
    defined as the sum of the squared residuals of the ODE functional and the initial conditions.

    Args:
        ODE_functional (Union[Callable, sympy.Expr]): Functional representation of the ODE
                                                      (Homogeneous diferential equation).
                                                      This can be a callable function or a
                                                      sympy expression.
        symbols_involved_in_ODE (list): The list of symbols involved in the ODE problem. The
                                        list of symbols should be in order of differentiation,
                                        with the first element being the independent variable,
                                        i.e. [x, f, dfdx, dfdxdx]
        initial_values (np.ndarray): Initial values of the ODE
        eta (float): Weighting factor for the ODE functional
        boundary_handling (str): Method for handling boundary conditions. Currently only "pinned"
                                    is supported.   
        
        


    Methods:
    --------
    """

    def __init__(self,         
        ODE_functional=None,
        symbols_involved_in_ODE=None,
        initial_values: np.ndarray = None,
        eta=np.float64(1.0),
        boundary_handling="pinned",
        ):
        super().__init__()
        self._verify_size_of_ivp_with_order_of_ODE(initial_values, symbols_involved_in_ODE)
        self.order_of_ODE = (
            len(symbols_involved_in_ODE) - 2
        )  # symbols_involved_in_ODE = [x, f, f_, f__, ...]
        self.symbols_involved_in_ODE = symbols_involved_in_ODE
        self.ODE_functional = self._create_ODE_loss_format(ODE_functional, symbols_involved_in_ODE)
        self.initial_values = initial_values
        self.eta = eta
        self.boundary_handling = boundary_handling
        
    def _create_ODE_loss_format(self, ODE_functional, symbols_involved_in_ODE=None):
        """
        Given an ODE_functional, returns a function that takes the QNN derivatives list and
        returns the loss function.

        Args:
            ODE_functional (Union[Callable, sympy.Expr]): Functional representation of the ODE
                                                          (Homogeneous diferential equation).
                                                          This can be a callable function or a
                                                          sympy expression. If a sympy expression
                                                          is given, then, the
                                                          symbols_involved_in_ODE must be provided.
            symbols_involved_in_ODE (list): The list of symbols involved in the ODE problem. The
                                            list of symbols should be in order of differentiation,
                                            with the first element being the independent variable,
                                            i.e. [x, f, dfdx, dfdxdx]
        Returns:
            QNN_loss (function): The loss function for the QNN with input in the format of the QNN
                                 tuple derivatives
        """

        if isinstance(ODE_functional, sp.Expr):  # if ode_question isinstance of sympy equation
            if symbols_involved_in_ODE is None:
                raise ValueError(
                    "symbols_involved_in_ODE must be provided"
                    " if ODE_functional is a sympy equation"
                )  # Perhaps this can be somehow improved by list(ODE_functional.free_symbols)
            _ODE_functional = lambda f_alpha_tensor: sp.lambdify(
                symbols_involved_in_ODE, ODE_functional, "numpy"
            )(*f_alpha_tensor)
        else:
            raise ValueError("Only sympy expressions are allowed")

        return _ODE_functional
    
    def _verify_size_of_ivp_with_order_of_ODE(self, initial_values, symbols_involved_in_ODE):
        """
        Verifies that the length of the initial values vector matches the order of the ODE.

        Args:
            initial_values (np.ndarray): Initial values of the ODE
            order_of_ODE (int): Order of the ODE
        """
        order_of_ODE = len(symbols_involved_in_ODE) - 2
        if order_of_ODE != len(initial_values):
            raise ValueError(
                f"Initial values must have the same length as the order of the ODE. Order of ODE:"
                f"{len(symbols_involved_in_ODE)-2},"
                f"Length of initial values: {len(initial_values)}"
            )
        elif order_of_ODE == 2:
            print(
                "WARNING: 2nd order DEs differentiate the QNN loss function by calculating the"
                " second derivative. This can be computationally expensive and inneficient."
                " An alternative is to set-up coupled 1rst order DEs (currently not implemented)"
            )
        elif order_of_ODE > 2:
            raise ValueError("Currently, only 1rst and 2nd order ODEs are supported")    
        
    def set_quantum_kernel(self, quantum_kernel: KernelMatrixBase) -> None:
        """
        Set the quantum kernel to be used in the loss function.

        Args:
            quantum_kernel (KernelMatrixBase): The quantum kernel to be used in the loss function.
        """
        if quantum_kernel == "precomputed":
            self._quantum_kernel = quantum_kernel
        else:
            self._quantum_kernel = quantum_kernel

    def compute(
        self,
        parameter_values: np.ndarray,
        data: np.ndarray,
        labels: np.ndarray,
        kernel_tensor: np.ndarray = None, #[K, dKdx, dKdxdx] where dKdx is a np.ndarray of shape (n_samples, n_samples) and dKdxdx is a np.ndarray of shape (n_samples, n_samples)
    ) -> float:
        """
        """
        
        def f_alpha_order(alpha_, kernel_tensor, order):
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
                return np.dot(kernel_tensor[order], alpha).reshape(-1, 1) + alpha_[0]
            return np.dot(kernel_tensor[order], alpha).reshape(-1, 1)


        f_alpha_tensor = np.array([f_alpha_order(parameter_values, kernel_tensor, i) for i in range(self.order_of_ODE+1)])        
        sum1 = np.sum((self.ODE_functional([data, *f_alpha_tensor])**2)) #Functional
        sum2 = np.sum((f_alpha_tensor[:,0][:len(self.initial_values)] - self.initial_values)**2) #Initial condition
        L = sum2 + sum1 * self.eta
        
        return L
