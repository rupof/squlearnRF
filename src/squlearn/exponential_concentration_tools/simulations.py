import numpy as np


from qiskit.quantum_info import Statevector, partial_trace
from squlearn.feature_map import *



def meyer_wallach_given_circuit(circuit, num_qubits):
    """
    Returns the Meyer-Wallach entanglement measure for the given circuit. 
    """
    res = 0
    N = num_qubits
    ansatz = circuit
    U = Statevector(ansatz)

    entropy = 0
    qb = list(range(N))
    for j in range(N):
        dens = partial_trace(U, qb[:j]+qb[j+1:]).data
        trace = np.trace(dens**2)
        entropy += trace
    entropy /= N
    res = 1 - entropy
    
    return res 


def meyer_wallach_given_features(fmap, num_qubits, x_list, save_evolution=False):
    """
    Returns the Meyer-Wallach entanglement measure for a given list of features x_list.
    """
    sample = len(x_list)

    res = np.zeros(sample, dtype=complex)
    N = num_qubits

    entropy_list = []
    
    for i in range(len(x_list)):
        features = x_list[i]
        ansatz = fmap.get_circuit(features)
        res[i]=meyer_wallach_given_circuit(ansatz, N)
        if save_evolution:  
            entropy_list.append(2*np.sum(res).real/(i+1))  

    if save_evolution:
        return entropy_list
    
    return 2*np.sum(res).real/sample