import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit

from ..encoding_circuit_base import EncodingCircuitBase


class ChebyshevTower_with_HEE(EncodingCircuitBase):
    r"""
    A feature-map that is based on the Chebyshev Tower encoding.

    **Example for 4 qubits, a 2 dimensional feature vector, 2 Chebyshev terms per feature,
    and 2 layers:**

    .. plot::

        from squlearn.encoding_circuit import ChebyshevTower
        pqc = ChebyshevTower(4, 2, 2, num_layers=2)
        pqc.draw(output="mpl", style={'fontsize':15,'subfontsize': 10})
        plt.tight_layout()

    The encoding gate and the scaling factor can be adjusted by parameters.
    It is also possible to change the indexing of the features.

    Args:
        num_qubits (int): Number of qubits of the ChebyshevTower encoding circuit
        num_features (int): Dimension of the feature vector
        n_chebyshev (int): Number of Chebyshev tower terms per feature dimension
        alpha (float): Scaling factor of Chebyshev tower
        num_layers (int): Number of layers
        rotation_gate (str): Rotation gate to use. Either ``rx``, ``ry`` or ``rz`` (default: ``ry``)
        hadamard_start (bool): If true, the circuit starts with a layer of Hadamard gates
                               (default: True)
        arrangement (str): Arrangement of the layers, either ``block`` or ``alternating``.
                          ``block``: The features are stacked together, ``alternating``:
                          The features are placed alternately (default: ``block``).
    """

    def __init__(
        self,
        num_qubits: int,
        num_features: int,
        num_chebyshev: int,
        alpha: float = 1.0,
        num_layers: int = 1,
        rotation_gate: str = "ry",
        hadamard_start: bool = True,
        arrangement: str = "block",
    ) -> None:
        super().__init__(num_qubits, num_features)

        self.num_chebyshev = num_chebyshev
        self.alpha = alpha
        self.num_layers = num_layers
        self.rotation_gate = rotation_gate
        self.hadamard_start = hadamard_start
        self.arrangement = arrangement

        if self.rotation_gate not in ("rx", "ry", "rz"):
            raise ValueError("Rotation gate must be either 'rx', 'ry' or 'rz'")

        if self.arrangement not in ("block", "alternating"):
            raise ValueError("Arrangement must be either 'block' or 'alternating'")

    @property
    def num_parameters(self) -> int:
        """The number of trainable parameters of the MultiControlEncodingCircuit encoding circuit."""
        num_param = 3 * (self.num_qubits) * self.num_layers
        
        return num_param

    @property
    def parameter_bounds(self) -> np.ndarray:
        """The bounds of the trainable parameters of the MultiControlEncodingCircuit encoding circuit."""
        return np.array([[-2.0 * np.pi, 2.0 * np.pi]] * self.num_parameters)

    def get_params(self, deep: bool = True) -> dict:
        """
        Returns hyper-parameters and their values of the Chebyshev Tower encoding

        Args:
            deep (bool): If True, also the parameters for
                         contained objects are returned (default=True).

        Return:
            Dictionary with hyper-parameters and values.
        """
        params = super().get_params()
        params["num_chebyshev"] = self.num_chebyshev
        params["alpha"] = self.alpha
        params["num_layers"] = self.num_layers
        params["rotation_gate"] = self.rotation_gate
        params["hadamard_start"] = self.hadamard_start
        params["arrangement"] = self.arrangement

        return params

    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
        parameters: Union[ParameterVector, np.ndarray] = None,
    ) -> QuantumCircuit:
        """
        Generates and returns the circuit of the Chebyshev encoding circuit.

        Args:
            features (Union[ParameterVector,np.ndarray]): Input vector of the features
                                                          from which the gate inputs are obtained
            param_vec (Union[ParameterVector,np.ndarray]): Input vector of the parameters
                                                           from which the gate inputs are obtained

        Return:
            Returns the circuit in Qiskit's QuantumCircuit format
        """

        if self.rotation_gate not in ("rx", "ry", "rz"):
            raise ValueError("Rotation gate must be either 'rx', 'ry' or 'rz'")

        if self.arrangement not in ("block", "alternating"):
            raise ValueError("Arrangement must be either 'block' or 'alternating'")

        def entangle_layer(QC: QuantumCircuit):
            """Creation of a simple NN entangling layer"""
            for i in range(0, self.num_qubits - 1, 2):
                QC.cx(i, i + 1)
            for i in range(1, self.num_qubits - 1, 2):
                QC.cx(i, i + 1)
            return QC

        def mapping(x, i):
            """Non-linear mapping for x: alpha*i*arccos(x)"""
            return self.alpha * i * np.arccos(x)

        nfeature = len(features)

        QC = QuantumCircuit(self.num_qubits)

        if self.hadamard_start:
            QC.h(range(self.num_qubits))

        index_count = 0
        for layer in range(self.num_layers):
            index_offset = 0
            iqubit = 0
            icheb = 1
            # Loops through the data encoding gates
            if self.arrangement == "block":
                outer = self.num_features
                inner = self.num_chebyshev
            elif self.arrangement == "alternating":
                inner = self.num_features
                outer = self.num_chebyshev
            else:
                raise ValueError("Arrangement must be either 'block' or 'alternating'")

            for outer_ in range(outer):
                for inner_ in range(inner):
                    if self.rotation_gate.lower() == "rx":
                        QC.rx(
                            mapping(features[index_offset % nfeature], icheb),
                            iqubit % self.num_qubits,
                        )
                    elif self.rotation_gate.lower() == "ry":
                        QC.ry(
                            mapping(features[index_offset % nfeature], icheb),
                            iqubit % self.num_qubits,
                        )
                    elif self.rotation_gate.lower() == "rz":
                        QC.rz(
                            mapping(features[index_offset % nfeature], icheb),
                            iqubit % self.num_qubits,
                        )
                    else:
                        raise ValueError(
                            "Rotation gate {} not supported".format(self.rotation_gate)
                        )
                    iqubit += 1
                    if self.arrangement == "block":
                        icheb += 1
                    elif self.arrangement == "alternating":
                        index_offset += 1

                if self.arrangement == "block":
                    index_offset += 1
                    icheb = 1
                elif self.arrangement == "alternating":
                    icheb += 1

            for i in range(self.num_qubits):
                QC.rz(parameters[index_count], i)
                QC.rx(parameters[index_count + 1], i)
                QC.rz(parameters[index_count + 2], i)
                index_count += 3

            # Entangling layer
            if layer + 1 < self.num_layers:
                QC = entangle_layer(QC)

        return QC
