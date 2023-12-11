import numpy as np


from qiskit.quantum_info import Statevector, partial_trace
from squlearn.encoding_circuit import *
from itertools import combinations
from squlearn.observables import CustomObservable



def generate_combinations(string_len, n_elements, symbol):
    """
    Generates all possible combinations of a given number of elements in a string of a given length.
    example: 
    generate_combinations(3, 2, "X") -> ["XXI", "IXX", "XIX"]
    """
    # If the number of elements requested is greater than the string length,
    # it's not possible to form such combinations, so return an empty list.
    if n_elements > string_len:
        return []

    # Generate all possible combinations of indices where the symbol ("X")
    # should be placed in the string.
    combinations_list = list(combinations(range(string_len), n_elements))

    # Initialize the list to store all the combinations.
    result = []
    
    # Iterate through each combination of indices and create the corresponding string.
    for combo in combinations_list:
        # Initialize a list with "I" (identity) characters for the string.
        string_characters = ["I"] * string_len
        
        # Set the symbol ("X") at the appropriate positions based on the combination.
        for idx in combo:
            string_characters[idx] = symbol
        
        # Convert the list of characters into a single string and add it to the result list.
        result.append("".join(string_characters))
    
    # Return the list of all unique combinations.
    return result


def generate_n_density_reduced_matrices(num_qubits, nDRM, basis  ):
    """
    Generates all possible density matrices for a given number of qubits and returns them as a list of CustomExpectationOperators
    """
    X_measurements = generate_combinations(num_qubits, nDRM, "X")
    Y_measurements = generate_combinations(num_qubits, nDRM, "Y")
    Z_measurements = generate_combinations(num_qubits, nDRM, "Z")
    if basis == "XYZ":
        measurements = X_measurements + Y_measurements + Z_measurements
    elif basis == "XY":
        measurements = X_measurements + Y_measurements
    elif basis == "XZ":
        measurements = X_measurements + Z_measurements
    elif basis == "YZ":
        measurements = Y_measurements + Z_measurements
    elif basis == "Z":
        measurements = Z_measurements
    elif basis == "X": 
        measurements = X_measurements
        
    for i in range(len(measurements)):
        measurements[i] = CustomObservable(num_qubits, measurements[i])
    return measurements

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