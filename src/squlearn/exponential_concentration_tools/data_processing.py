import numpy as np
import os
import json


def SaveParameterDatasetAndKernel(directory, feature_map, num_layers, num_qubits, parameters, dataset_name, 
                                  dataset, kernel, dataset_preprocessing, extra_parameters, extra_name, entanglement_score=None,
                                    measurement_basis = None, outerkernel = None, outerkernel_parameter = None):
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
        directory_name += f"_{outerkernel}_{outerkernel_parameter}_basis{measurement_basis}"

    # Add extra_parameters to the directory name if provided
    if extra_parameters is not None:
        extra_parameters_str = json.dumps(extra_parameters)
        extra_parameters_str = extra_parameters_str.replace("[", "").replace("]", "").replace(",", "_").replace(" ", "")
        directory_name += f"_extra_parameters{extra_parameters_str}"

    # Add extra_name to the directory name if provided
    if extra_name is not None:
        directory_name += f"_{extra_name}"

    directory_name += f"_samplesize{len(dataset)}"
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




def find_string_and_load(lines, string):
    for line in lines:
        if string in line:
            try:
                result = json.loads(line.split(":", 1)[1].strip())
                return result
            except (IndexError, ValueError):
                return None

    return None

def interpret_measurement_str(input_string):
    """
    # Test examples
    print(interpret_string("Z1DRM"))     # Output: ("Z", 1)
    print(interpret_string("XYZ3DRM"))   # Output: ("XYZ", 3)
    """
    # Initialize measurement_basis and nDRM variables
    measurement_basis = ""
    nDRM = ""

    # Find the index where the numeric part of the string starts
    numeric_index = None
    for i in range(len(input_string)):
        if input_string[i].isdigit():
            numeric_index = i
            if input_string[i+1].isdigit():
                return input_string[:numeric_index], int(input_string[numeric_index:numeric_index+2])
            break

    # If no numeric part found, return empty values for measurement_basis and nDRM
    if numeric_index is None:
        return measurement_basis, nDRM

    # Extract the measurement_basis and nDRM from the input_string
    measurement_basis = input_string[:numeric_index]
    nDRM = input_string[numeric_index]

    return measurement_basis, int(nDRM)




def find_string(lines, string):
    for line in lines:
        if string in line:
            try:
                result = line.split(":", 1)[1].strip()
                return result
            except IndexError:
                return None

    return None

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


    # Extract properties if they exist

    parameters = find_string_and_load(lines, "Parameters")  
    feature_map = find_string(lines, "Feature Map")
    num_layers = int(find_string(lines, "Number of Layers"))
    num_qubits = int(find_string(lines, "Number of Qubits"))
    dataset_name = find_string(lines, "Dataset Name")
    dataset_preprocessing = find_string(lines, "Dataset Preprocessing")
    extra_parameters = find_string_and_load(lines, "Extra Parameters")
    extra_name = find_string(lines, "Extra Name")
    measurement_basis = find_string(lines, "Measurement Basis")
    outerkernel = find_string(lines, "Outer Kernel")
    outerkernel_parameter = find_string(lines, "Outer Kernel Parameter")
    

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
    
    test_dataset = []
    test_dataset_file = os.path.join(directory,"test_dataset.txt")
    if os.path.isfile(test_dataset_file):
        with open(test_dataset_file, "r") as f:
            for line in f:
                row = [float(element) for element in line.strip().split()]
                test_dataset.append(row)
    
    test_kernel = []
    test_kernel_file = os.path.join(directory,"test_kernel.txt")
    if os.path.isfile(test_kernel_file):
        with open(test_kernel_file, "r") as f:
            for line in f:
                row = [float(element) for element in line.strip().split()]
                test_kernel.append(row)

    test_prediction = []

    test_prediction_file = os.path.join(directory,"test_prediction.txt")
    if os.path.isfile(test_prediction_file):
        with open(test_prediction_file, "r") as f:
            for line in f:
                row = [float(element) for element in line.strip().split()]
                test_prediction.append(row)
    if test_prediction == []:
        print("it was empty")
        test_prediction_file = os.path.join(directory,"test_predict.txt")
        if os.path.isfile(test_prediction_file):
            with open(test_prediction_file, "r") as f:
                for line in f:
                    row = [float(element) for element in line.strip().split()]
                    test_prediction.append(row)
        


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
        "outerkernel_parameter": outerkernel_parameter,
        "test_dataset": np.array(test_dataset),
        "test_kernel": np.array(test_kernel),
        "test_prediction": np.array(test_prediction)
    }

    return parameters_dict

def window_long_name_adaptation(directory_path, string):
    file = os.path.join(directory_path, string)
    file = file.encode("unicode_escape").decode()
    file = os.path.abspath(os.path.normpath(file))
    return file


def GenerateDirectoryName(original_directory, feature_map, num_layers, num_qubits, dataset_name, dataset_preprocessing, 
extra_parameters=None, extra_name=None, outerkernel=None, outerkernel_parameter=None, measurement_basis=None, old = False, samplesize = None):
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
    if outerkernel is not None and old is False:
        directory_name += f"_{outerkernel}_{outerkernel_parameter}_basis{measurement_basis}"
    elif outerkernel is not None and old is True:
        #print(directory_name)
        directory_name += f"_{outerkernel}_{outerkernel_parameter}"
        pass
        #directory_name += f"_{outerkernel}_{outerkernel_parameter}"
    elif outerkernel is None and old is True:
        pass

    if extra_parameters is not None:
        extra_parameters_str = json.dumps(extra_parameters)
        extra_parameters_str = extra_parameters_str.replace("[", "").replace("]", "").replace(",", "_").replace(" ", "")
        directory_name += f"_extra_parameters{extra_parameters_str}"

    if extra_name is not None:
        directory_name += f"_{extra_name}"
    
    if samplesize is not None:
        directory_name += f"_samplesize{samplesize}"

    

    return window_long_name_adaptation(original_directory, directory_name)


def ExtractResults(original_directory, feature_map, num_layers, num_qubits, dataset_name, dataset_preprocessing, 
                   extra_parameters=None, extra_name=None,  outerkernel=None, outerkernel_parameter=None, measurement_basis = None, samplesize = None):
    """
    Wrapper functions that returns the experiment results as a dictionary.
    """

    
    try: 
        directory_name = GenerateDirectoryName(original_directory, feature_map, num_layers, num_qubits, dataset_name, dataset_preprocessing, 
                                           extra_parameters, extra_name, outerkernel, outerkernel_parameter, measurement_basis, samplesize = samplesize)

        return ReverseEngineerParameters(directory_name)
    except Exception as e:
        print(e)
        print(f"N_Could not extract results for {directory_name}")
        try: 
            directory_name = GenerateDirectoryName(original_directory, feature_map, num_layers, num_qubits, dataset_name, dataset_preprocessing, 
                                           extra_parameters, extra_name, outerkernel, outerkernel_parameter, measurement_basis, old = True, samplesize = samplesize)
            try:
                return ReverseEngineerParameters(directory_name)
            except:
                print(f"Could not extract results for {directory_name}")
                try: 
                    directory_name = GenerateDirectoryName(original_directory, feature_map, num_layers, num_qubits, dataset_name, dataset_preprocessing, 
                                           extra_parameters, extra_name, None, outerkernel_parameter, measurement_basis, old = True, samplesize = samplesize)
                    try:
                        return ReverseEngineerParameters(directory_name)
                    except:
                        pass
                except:
                    print(f"Could not extract results for {directory_name}")
        except:
            print(f"Could not extract results for {directory_name}")
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
