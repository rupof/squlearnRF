import numpy as np
import os
import json


def SaveParameterDatasetAndKernel(directory, feature_map, num_layers, num_qubits, parameters, dataset_name, dataset, kernel, dataset_preprocessing, extra_parameters, extra_name, entanglement_score=None, measurement_basis = None, outerkernel = None, outerkernel_parameter = None):
    """
    Saves the parameters, dataset, kernel, and optional entanglement score to files within the specified directory.
    The default naming convention is as follows:
        {directory/{feature_map}_num_layers{num_layers}_num_qubits{num_qubits}_{dataset_name}_{dataset_preprocessing}_extra_parameters{extra_parameters}_extra_name{extra_name}

    Args:
        directory (str): The directory path to save the files.
        feature_map (str): The feature map used.
        num_layers (int): The number of layers.
        num_qubits (int): The number of qubits.
        parameters (str or list or dict): The parameters used. Can be a string, list, or dictionary.
        dataset_name (str): The name of the dataset.
        dataset (list of lists): The dataset as a list of lists.
        kernel (list of lists): The kernel as a list of lists.
        dataset_preprocessing (str): The dataset preprocessing information.
        extra_parameters (list or dict): Additional parameters (optional). Can be a list or dictionary.
        extra_name (str): Additional name (optional).
        entanglement_score (ndarray, optional): The entanglement score as a 1D NumPy array. Default is None.

    Returns:
        None
    """
    # Create the directory name based on the naming convention
    directory_name = f"{feature_map}_num_layers{num_layers}_num_qubits{num_qubits}_{dataset_name}_{dataset_preprocessing}"
    if outerkernel is not None:
        directory_name += f"_{outerkernel}_{outerkernel_parameter}"

    # Add extra_parameters to the directory name if provided
    if extra_parameters is not None:
        extra_parameters_str = json.dumps(extra_parameters)
        extra_parameters_str = extra_parameters_str.replace("[", "").replace("]", "").replace(",", "_").replace(" ", "")
        directory_name += f"_extra_parameters{extra_parameters_str}"

    # Add extra_name to the directory name if provided
    if extra_name is not None:
        directory_name += f"_{extra_name}"

    # Create the full directory path
    directory_path = os.path.join(directory, directory_name)

    # Create the directory to save the files if it doesn't exist
    os.makedirs(directory_path, exist_ok=True)

    # Save the parameters
    parameters_file = os.path.join(directory_path, "parameters.txt")
    with open(parameters_file, "w") as f:
        f.write(f"Feature Map: {feature_map}\n")
        f.write(f"Number of Layers: {num_layers}\n")
        f.write(f"Number of Qubits: {num_qubits}\n")
        f.write(f"Parameters: {parameters}\n")
        f.write(f"Dataset Name: {dataset_name}\n")
        f.write(f"Dataset Preprocessing: {dataset_preprocessing}\n")
        if extra_parameters is not None:
            f.write(f"Extra Parameters: {json.dumps(extra_parameters)}\n")
        if extra_name is not None:
            f.write(f"Extra Name: {extra_name}\n")
        if measurement_basis is not None:
            f.write(f"Measurement Basis: {measurement_basis}\n")
        if outerkernel is not None:
            f.write(f"Outer Kernel: {outerkernel}\n")
        if outerkernel_parameter is not None:
            f.write(f"Outer Kernel Parameter: {outerkernel_parameter}\n")

    # Save the dataset
    dataset_file = os.path.join(directory_path, "dataset.txt")
    with open(dataset_file, "w") as f:
        for row in dataset:
            f.write(" ".join(str(element) for element in row) + "\n")

    # Save the kernel
    kernel_file = os.path.join(directory_path, "kernel.txt")
    with open(kernel_file, "w") as f:
        for row in kernel:
            f.write(" ".join(str(element) for element in row) + "\n")

    # Save the entanglement score if provided
    if entanglement_score is not None:
        entanglement_score_file = os.path.join(directory_path, "entanglement_score.txt")
        with open(entanglement_score_file, "w") as f:
            f.write("\n".join(str(score) for score in entanglement_score))

    print("Files saved successfully.")



def ReverseEngineerParameters(directory):
    """
    Reverse engineers the parameters, dataset, kernel, and entanglement score from the specified directory.

    Args:
        directory (str): The directory path containing the saved files.

    Returns:
        dict: A dictionary containing the reverse engineered parameters, dataset, kernel, and entanglement score (if present).
            The dictionary has the following keys:
            - 'feature_map': The feature map used.
            - 'num_layers': The number of layers.
            - 'num_qubits': The number of qubits.
            - 'parameters': The parameters used. Can be a string, list, or dictionary.
            - 'dataset_name': The name of the dataset.
            - 'dataset_preprocessing': The dataset preprocessing information.
            - 'extra_parameters': Additional parameters (optional). Can be a list or dictionary.
            - 'extra_name': Additional name (optional).
            - 'dataset': The reverse engineered dataset as a list of lists.
            - 'kernel': The reverse engineered kernel as a list of lists.
            - 'entanglement_score': The entanglement score as a 1D NumPy array (if present), otherwise None.
            - 'measurement_basis': The measurement basis (if present), otherwise None. 
            - 'outerkernel': The outer kernel (if present), otherwise None.
            - 'outerkernel_parameter': The outer kernel parameter (if present), otherwise None.
    """
    # Get the subdirectory with the additional name
    #subdirectories = os.listdir(directory)
    #additional_directory = None
    #for subdirectory in subdirectories:
    #    if os.path.isdir(os.path.join(directory, subdirectory)):
    #        additional_directory = subdirectory
    #        break

    # Check if the additional directory was found
    #if additional_directory is None:
    #    raise ValueError("Additional directory not found")

    # Update the parameters_file path
    parameters_file = os.path.join(directory, "parameters.txt")
    with open(parameters_file, "r") as f:
        lines = f.readlines()

    # Extract feature map
    feature_map = lines[0].split(":")[1].strip()

    # Extract number of layers
    num_layers = int(lines[1].split(":")[1].strip())

    # Extract number of qubits
    num_qubits = int(lines[2].split(":")[1].strip())

    # Extract parameters if they exist
    parameters_str = lines[3].split(":")[1].strip()
    if parameters_str != "None":
        try:
            parameters = json.loads(parameters_str)
        except:
        #if parameters are not a list or dictionary, then they are a string
            parameters = parameters_str
    else:
        parameters = None

    # Extract dataset name
    dataset_name = lines[4].split(":")[1].strip()

    # Extract dataset preprocessing information
    dataset_preprocessing = lines[5].split(":")[1].strip()

    # Extract extra parameters if they exist
    extra_parameters = None
    if len(lines) > 6 and "Extra Parameters" in lines[6]:
        extra_parameters = json.loads(lines[6].split(":")[1].strip())

    # Extract extra name if it exists
    extra_name = None
    if len(lines) > 7 and "Extra Name" in lines[7]:
        extra_name = lines[7].split(":")[1].strip()

    # Extract measurement basis if it exists
    measurement_basis = None
    if len(lines) > 8 and "Measurement Basis" in lines[8]:
        measurement_basis = lines[8].split(":")[1].strip()
    
    # Extract outer kernel if it exists
    outerkernel = None
    if len(lines) > 9 and "Outer Kernel" in lines[9]:
        outerkernel = lines[9].split(":")[1].strip()
    
    # Extract outer kernel parameter if it exists
    outerkernel_parameter = None
    if len(lines) > 10 and "Outer Kernel Parameter" in lines[10]:
        outerkernel_parameter = lines[10].split(":")[1].strip()
    

    # Read dataset file
    dataset_file = os.path.join(directory, "dataset.txt")
    dataset = []
    with open(dataset_file, "r") as f:
        for line in f:
            row = [float(element) for element in line.strip().split()]
            dataset.append(row)

    # Read kernel file
    kernel_file = os.path.join(directory, "kernel.txt")
    kernel = []
    with open(kernel_file, "r") as f:
        for line in f:
            row = [float(element) for element in line.strip().split()]
            kernel.append(row)

    # Read entanglement score if it exists
    entanglement_score = None
    entanglement_score_file = os.path.join(directory,"entanglement_score.txt")
    if os.path.isfile(entanglement_score_file):
        with open(entanglement_score_file, "r") as f:
            entanglement_score = [float(line.strip()) for line in f]

    # Return the reverse engineered parameters as a dictionary
    parameters_dict = {
        "feature_map": feature_map,
        "num_layers": num_layers,
        "num_qubits": num_qubits,
        "parameters": parameters,
        "dataset_name": dataset_name,
        "dataset_preprocessing": dataset_preprocessing,
        "extra_parameters": extra_parameters,
        "extra_name": extra_name,
        "dataset": np.array(dataset),
        "kernel": np.array(kernel),
        "entanglement_score": np.array(entanglement_score),
        "measurement_basis": measurement_basis,
        "outerkernel": outerkernel,
        "outerkernel_parameter": outerkernel_parameter
    }

    return parameters_dict


def GenerateDirectoryName(original_directory, feature_map, num_layers, num_qubits, dataset_name, dataset_preprocessing, 
extra_parameters=None, extra_name=None, outerkernel=None, outerkernel_parameter=None):
    """
    Generates a directory name based on the provided inputs.

    Args:
        original_directory (str): The original directory path.
        feature_map (str): The feature map used.
        num_layers (int): The number of layers.
        num_qubits (int): The number of qubits.
        dataset_name (str): The name of the dataset.
        dataset_preprocessing (str): The dataset preprocessing information.
        extra_parameters (list or dict, optional): Additional parameters (optional). Can be a list or dictionary. Default is None.
        extra_name (str, optional): Additional name (optional). Default is None.

    Returns:
        str: The generated directory name.
    """
    
    directory_name = f"{feature_map}_num_layers{num_layers}_num_qubits{num_qubits}_{dataset_name}_{dataset_preprocessing}"
    if outerkernel is not None:
        directory_name += f"_{outerkernel}_{outerkernel_parameter}"

    if extra_parameters is not None:
        extra_parameters_str = json.dumps(extra_parameters)
        extra_parameters_str = extra_parameters_str.replace("[", "").replace("]", "").replace(",", "_").replace(" ", "")
        directory_name += f"_extra_parameters{extra_parameters_str}"

    if extra_name is not None:
        directory_name += f"_{extra_name}"

    

    return os.path.join(original_directory, directory_name)


def ExtractResults(original_directory, feature_map, num_layers, num_qubits, dataset_name, dataset_preprocessing, extra_parameters=None, extra_name=None,  outerkernel=None, outerkernel_parameter=None):
    """
    Wrapper functions that returns the experiment results as a dictionary.
    """
    directory_name = GenerateDirectoryName(original_directory, feature_map, num_layers, num_qubits, dataset_name, dataset_preprocessing, extra_parameters, extra_name, outerkernel, outerkernel_parameter)
    
    try: 
        return ReverseEngineerParameters(directory_name)
    except:
        raise ValueError(f"Could not extract results for {directory_name}")
    

########################################### Some examples ###########################################

# Example usage: Save parameters, dataset, kernel, and entanglement score
"""
directory = "test/"
feature_map = "featuremap"
num_layers = 5
num_qubits = 8
parameters = [1.2, 3.4, 5.6]
dataset_name = "datasetName"
dataset = [[0.1, 0.2, 0.3, 0.4, 0.5],
           [0.6, 0.7, 0.8, 0.9, 1.0],
           [1.1, 1.2, 1.3, 1.4, 1.5],
           [1.6, 1.7, 1.8, 1.9, 2.0],
           [2.1, 2.2, 2.3, 2.4, 2.5]]
kernel = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
dataset_preprocessing = "MinMax"
extra_parameters = [-0.91, 0.99]
extra_name = None
entanglement_score = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

# Save the parameters, dataset, kernel, and entanglement score
SaveParameterDatasetAndKernel(directory, feature_map, num_layers, num_qubits, parameters, dataset_name, dataset, kernel, dataset_preprocessing, extra_parameters, extra_name, entanglement_score)


# Extract Dictionary
parameters_dict = ExtractResults(directory, feature_map, num_layers, num_qubits, dataset_name, dataset_preprocessing, extra_parameters, extra_name)

"""
