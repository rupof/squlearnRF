import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit

from ..feature_map_base import FeatureMapBase


class HardwareEfficientEmbeddingMap(FeatureMapBase):
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
        num_features: int,
        num_layers: int = 1,
        rotation_gate: str = "rx",
    ) -> None:
        super().__init__(num_qubits, num_features)
        self.num_layers = num_layers
        self.rotation_gate = rotation_gate

    @property
    def num_parameters(self) -> int:
        return 0    

    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
        parameters: Union[ParameterVector, np.ndarray] = None,
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
        if self.num_features != len(features):
            raise ValueError("Wrong number of features")
        
        if self.num_features > self.num_qubits:
            raise ValueError("More features than qubits. Reduce number of features or increase number of qubits")
        elif self.num_features < self.num_qubits:
            raise ValueError("More qubits than features. Data reuploading not yet implemented for more qubits than features")
        
        QC = QuantumCircuit(self.num_qubits)

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

