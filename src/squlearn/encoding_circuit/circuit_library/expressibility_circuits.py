import numpy as np
from typing import Union

from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit

from ..encoding_circuit_base import EncodingCircuitBase


class ExpressibilityEmbeddingCircuit(EncodingCircuitBase):
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
        circuit_index: int = 1,
        independent_degrees_of_freedom: bool = False,
    ) -> None:
        super().__init__(num_qubits, num_features)
        self.num_layers = num_layers
        self.circuit_index = circuit_index
        self.independent_degrees_of_freedom = independent_degrees_of_freedom
    @property
    def num_parameters(self) -> int:
        """The number of parameters depends on the circuit_index"""
        if self.circuit_index == 1:
            num_parameters = 2*self.num_qubits*self.num_layers
        elif self.circuit_index == 2:
            num_parameters = 2*self.num_qubits*self.num_layers
        elif self.circuit_index == 3: 
            num_parameters = (2*self.num_qubits+(self.num_qubits-1))*self.num_layers
        elif self.circuit_index == 4:
            num_parameters = (2*self.num_qubits+(self.num_qubits-1))*self.num_layers
        elif self.circuit_index == 6:
            num_parameters = (4*self.num_qubits+(self.num_qubits-1))*self.num_layers
        elif self.circuit_index == 7:
            num_parameters = (4*self.num_qubits+(self.num_qubits-1))*self.num_layers
        elif self.circuit_index == 8:
            num_parameters = (4*self.num_qubits+(self.num_qubits-1))*self.num_layers


        return num_parameters    

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

        

        if self.circuit_index == 1: 
            def rotation_func(theta, qubit):
                    QC.rx(theta, qubit)
                    QC.rz(theta, qubit)
            count = 0
            if self.independent_degrees_of_freedom:
                for layer in range(self.num_layers):
                    # Apply single-qubit rotations
                    for i in range(self.num_qubits):
                        QC.rx(parameters[count], i % self.num_qubits)
                        count += 1
                        QC.rz(parameters[count], i % self.num_qubits)
                        count += 1
                    QC.barrier()
            else:
                for layer in range(self.num_layers):
                    # Apply single-qubit rotations
                    for i in range(self.num_qubits):
                        rotation_func(features[i], i % self.num_qubits)
                    QC.barrier()

        elif self.circuit_index == 2: 
                def rotation_func(theta, qubit):
                    QC.rx(theta, qubit)
                    QC.rz(theta, qubit)
                if self.independent_degrees_of_freedom:
                    count = 0
                    for layer in range(self.num_layers):
                        # Apply single-qubit rotations
                        # Apply single-qubit rotations
                        for i in range(self.num_qubits):
                            QC.rx(parameters[count], i % self.num_qubits)
                            count += 1
                            QC.rz(parameters[count], i % self.num_qubits)
                            count += 1
                        # Apply entangling gates
                        for i in range(self.num_qubits - 1):
                            QC.cx(i, i+1)
                    QC.barrier()
                else:
                    for layer in range(self.num_layers):
                        # Apply single-qubit rotations
                        for i in range(self.num_qubits):
                            rotation_func(features[i], i % self.num_qubits)    

                        # Apply entangling gates
                        for i in range(self.num_qubits - 1):
                            QC.cx(i, i+1)
                        QC.barrier()

        elif self.circuit_index == 3: 
                count = 0
                def rotation_func(theta, qubit):
                    QC.rx(theta, qubit)
                    QC.rz(theta, qubit)
                    
                if self.independent_degrees_of_freedom:
                    for layer in range(self.num_layers):
                        # Apply single-qubit rotations
                        for i in range(self.num_qubits):
                            QC.rx(parameters[count], i % self.num_qubits)
                            count += 1
                            QC.rz(parameters[count], i % self.num_qubits)
                            count += 1
                        # Apply entangling gates
                        for i in range(self.num_qubits - 1):
                            QC.crz(parameters[count], i, i+1)
                            count += 1
                        QC.barrier()
                
                else:
                    for layer in range(self.num_layers):
                        # Apply single-qubit rotations
                        for i in range(self.num_qubits):
                            rotation_func(features[i], i % self.num_qubits)    

                        # Apply entangling gates
                        for i in range(self.num_qubits - 1):
                            QC.crz(features[i], i, i+1)    #Attention with this line
                        QC.barrier()
        elif self.circuit_index == 4: 
                count = 0
                def rotation_func(theta, qubit):
                        QC.rx(theta, qubit)
                        QC.rz(theta, qubit)
                        
                if self.independent_degrees_of_freedom:
                    
                    for layer in range(self.num_layers):
                        # Apply single-qubit rotations
                        for i in range(self.num_qubits):
                            QC.rx(parameters[count], i % self.num_qubits)
                            count += 1
                            QC.rz(parameters[count], i % self.num_qubits)
                            count += 1
                        # Apply entangling gates
                        for i in range(self.num_qubits - 1):
                            QC.crx(parameters[count], i, i+1)   #Attention with this line
                            count += 1
                else: 
                    raise ValueError("Not implemented")
        elif self.circuit_index == 7: 
            count = 0
                    
            if self.independent_degrees_of_freedom:
                
                for layer in range(self.num_layers):
                    # Apply single-qubit rotations
                    for i in range(self.num_qubits):
                        QC.rx(parameters[count], i )
                        count += 1
                        QC.rz(parameters[count], i )
                        count += 1
                    # Apply entangling gates
                    for i in range(self.num_qubits):
                        #between only odd and even: 0-1, 2-3, 4-5, 6-7
                        if i % 2 == 0:
                            try:
                                QC.crz(parameters[count], i, i+1)
                            except:
                                break
                            count += 1
                    for i in range(self.num_qubits):
                        QC.rx(parameters[count], i )
                        count += 1
                        QC.rz(parameters[count], i )
                        count += 1
                    QC.barrier()
                    # Apply entangling gates between not previously entangled qubits
                    for i in range(self.num_qubits):
                        if i % 2 == 1:
                            try:
                                QC.crz(parameters[count], i, i+1)
                            except:
                                break

                            count += 1


            else: 
                raise ValueError("Not implemented")
        elif self.circuit_index == 8: 
            count = 0
                    
            if self.independent_degrees_of_freedom:
                
                for layer in range(self.num_layers):
                    # Apply single-qubit rotations
                    for i in range(self.num_qubits):
                        QC.rx(parameters[count], i )
                        count += 1
                        QC.rz(parameters[count], i )
                        count += 1
                    # Apply entangling gates
                    for i in range(self.num_qubits):
                        #between only odd and even: 0-1, 2-3, 4-5, 6-7
                        if i % 2 == 0:
                            try:
                                QC.crx(parameters[count], i, i+1)
                            except:
                                break
                            count += 1
                    for i in range(self.num_qubits):
                        QC.rx(parameters[count], i)
                        count += 1
                        QC.rz(parameters[count], i )
                        count += 1
                    # Apply entangling gates between not previously entangled qubits
                    for i in range(self.num_qubits):
                        if i % 2 == 1:
                            try:
                                QC.crx(parameters[count], i, i+1)
                            except:
                                break
                            count += 1
                    QC.barrier()


            else: 
                raise ValueError("Not implemented")
            

        elif self.circuit_index == 100: 
            def rotation_func(theta, qubit):
                    QC.h(qubit)
                   
            for layer in range(self.num_layers):
                # Apply single-qubit rotations
                # one hadamard, one skip
                for i in range(self.num_qubits):
                    QC.h(i % self.num_qubits)

                # Apply entangling gates
                for i in range(self.num_qubits - 1):
                    QC.cnot(i+1, i)   #Attention with this line
                for i in range(self.num_qubits):
                    QC.rx(features[i], i % self.num_qubits) 
             

        return QC

