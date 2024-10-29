import numpy as np
from typing import Union
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import random_statevector

from ..encoding_circuit_base import EncodingCircuitBase


class HamiltonianEvolution_EncodingCircuit(EncodingCircuitBase):
    """
    
    Creates the data reuploading encoding circuit as presented in reference [1], Eq. L4. That is based on an evolving 1D Heisenberg model with interactions [1]. The encoding circuit is defined as:

    .. math::
        |\phi(\mathbf{x})\rangle=\left(\prod_{j=1}^{d}\exp\left(-i\frac{t}{T}\left( \hat{X}_j\hat{X}_{j+1}\mathbf{x}_{j} + \hat{Y}_j\hat{Y}_{j+1}\mathbf{x}_{j} + \hat{Z}_j\hat{Z}_{j+1}\mathbf{x}_{j})\right) \right)\right)^{T}\otimes_{j}^{j+1}|\psi\rangle\,,

     

    **Example for a 2 dimensional feature vector, 2 Trotterized layers, and an evolution time of 1:**

    .. plot::
    
            from squlearn.encoding_circuit import HamiltonianEvolution_EncodingCircuit
            pqc = HamiltonianEvolution_EncodingCircuit(2, 2, 1)
            plt = pqc.draw(output="mpl", style={'fontsize':15,'subfontsize': 10})
            plt.tight_layout()

    Args:
        num_features (int): The number of features to encode. The number of features also defines the number of qubits, which is equal to num_features + 1.
        num_layers_T (int): The number of Trotterized layers.
        evolution_time_t: The evolution time of the Hamiltonian Evolution encoding circuit.
        trotterize: Whether to use Trotterization in the encoding circuit.

    """

    def __init__(
        self,
        num_features: int,
        num_layers_T: int = 1,
        evolution_time_t: float = 1.0,
        trotterize: bool = True,
    ) -> None:
        super().__init__(num_features+1, num_features)
        self._num_layers = num_layers_T
        self.evolution_time_t = evolution_time_t
        self.trotterize = trotterize 
        self.use_random_initial_state = False
        self.random_initial_state_seed = 1

    @property
    def num_parameters(self) -> int:
        """The number of trainable parameters of the Hamiltonian Evolution encoding circuit."""
        return 0

    @property
    def parameter_bounds(self) -> np.ndarray:
        """The bounds of the trainable parameters of the Hamiltonian Evolution encoding circuit."""
        return np.array([])

    @property
    def num_layers(self) -> int:
        """The number of layers of the Hamiltonian Evolution encoding circuit."""
        return self._num_layers

    @property
    def evolution_time(self) -> float:
        """ The evolution time of the Hamiltonian Evolution encoding circuit, equivalent to the bandwidth-tuning parameter."""
        return self.evolution_time_t

    def get_params(self, deep: bool = True) -> dict:
        """
        Returns hyper-parameters and their values of the Hamiltonian Evolution encoding circuit
        """
        params = super().get_params()
        params["num_layers"] = self._num_layers
        params["evolution_time_t"] = self.evolution_time_t
        params["trotterize"] = self.trotterize
        return params

    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
    ) -> QuantumCircuit:
        """
        Return the circuit of the Hamiltonian Evolution encoding circuit.

        Args:
            features: The features to encode.

        Return:
            Returns the circuit in qiskit format.
        """
        # Creates the layers of the encoding circuit
        QC = QuantumCircuit(self.num_qubits)
        def H_j_m(QC, j):
            if self.trotterize:
                encoding_angle = features[j]/self.num_layers                
            else:
                encoding_angle = features[j]
            
            jp1 = j+1
            # XX
            QC.h([j, jp1])
            QC.cx(j, jp1)
            QC.rz(encoding_angle, jp1)
            QC.cx(j, jp1)
            QC.h([j, jp1])
            # YY
            QC.rx(-np.pi/2, [j, jp1])
            QC.cx(j, jp1)
            QC.rz(encoding_angle, jp1)
            QC.cx(j, jp1)
            QC.rx(np.pi/2, [j, jp1])
            # ZZ
            QC.cx(j, jp1)
            QC.rz(encoding_angle, jp1)
            QC.cx(j, jp1)
            return QC
        
        QC = QuantumCircuit(self.num_qubits)
        if self.use_random_initial_state:
            QC.prepare_state(random_statevector(2**self.num_qubits, seed=self.random_initial_state_seed))

        for n_l in range(self.num_layers):
            for j in range(len(features)):
                QC = H_j_m(QC, j)
        return QC

