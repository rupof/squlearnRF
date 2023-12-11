import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit

from ..encoding_circuit_base import EncodingCircuitBase
from functools import reduce


class IQPLikeCircuit(EncodingCircuitBase):
    """
    Creates a feature map based on the hardware efficient embedding as described in arXiv:2208.11060v1 

    One layer has the following structure:



    Args:
        num_qubits (int): Number of qubits of the HardwareEfficientEmbedding feature map
        num_features (int): Dimension of the feature vector
        num_layers (int): Number of layers (default: 1)
        rotation_gate (str): rotation gate to use. Choose from 'rx', 'ry', 'rz', or 'h_rz'.
    """

    def __init__(
        self,
        num_qubits: int,
        num_layers: int = 1,
    ) -> None:
        super().__init__(num_qubits, num_qubits)
        self.num_layers = num_layers

    @property
    def num_parameters(self) -> int:
        return 1    
    
    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
        parameters: Union[ParameterVector, np.ndarray] = [1],
    ) -> QuantumCircuit:
        """
        Returns the circuit of the HardwareEfficientEmbedding feature map

        Args:
            features Union[ParameterVector,np.ndarray]: Input vector of the features
                from which the gate inputs are obtained
            param_vec Union[ParameterVector,np.ndarray]: Input vector of the parameters
                from which the gate inputs are obtained

        Return:
            Returns the circuit in Qiskit's QuantumCircuit format
        """
        from qiskit.circuit.library import ZZFeatureMap
        
        def self_product(self, *args):
            """
            adapted from https://github.com/rsln-s/Importance-of-Kernel-Bandwidth-in-Quantum-Machine-Learning/blob/main/code/utils.py
            Define a function map from R^n to R.

            Args:
                x: data

            Returns:
                float: the mapped value
            """
            return np.prod(np.array(args))*parameters[0]

        QC = ZZFeatureMap(self.num_qubits, reps=self.num_layers, data_map_func=self_product)
        return QC



