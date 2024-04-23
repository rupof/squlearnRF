import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit

from ..encoding_circuit_base import EncodingCircuitBase


class HardwareEfficientEmbeddingCircuit(EncodingCircuitBase):
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
        rotation_gate: str = "rx",
    ) -> None:
        super().__init__(num_qubits, num_qubits)
        self.num_layers = num_layers
        self.rotation_gate = rotation_gate

    @property
    def num_parameters(self) -> int:
        return 0    

    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
        parameters: Union[ParameterVector, np.ndarray],
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

        QC = QuantumCircuit(self.num_qubits)
        #C = parameters[0]
        def h_rz_gate(theta, qubit):
            QC.h(qubit)
            QC.rz(theta, qubit)

        gate_mapping = {
            'rx': QC.rx,
            'ry': QC.ry,
            'rz': QC.rz,
            'h_rz': h_rz_gate
        }


        rotation_func = gate_mapping.get(self.rotation_gate)
        if rotation_func is None:
            raise ValueError("Invalid rotation_gate value. Choose 'rx', 'ry', 'rz', or 'h_rz'.")
            
        for layer in range(self.num_layers):
            # Apply single-qubit rotations
            for i in range(self.num_qubits):
                rotation_func(features[i], i % self.num_qubits)

            # Apply entangling gates
            for i in range(self.num_qubits - 1):
                QC.cx(i, i+1)

        return QC
