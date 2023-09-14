import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_datasets as tfds


############ MNIST ############


np.random.seed(1)


def pca_paper(X, n_components):
    """
    Performs PCA on the input dataset X and returns the projected dataset X_pca as implemented in the paper.
    """
    # Step 1: Compute the average of the input data points
    x_avg = np.mean(X, axis=0)

    # Step 2: Normalize each individual input data point
    X_norm = X - x_avg

    # Step 3: Construct the normalized dataset matrix
    X_norm_T = X_norm.T

    # Step 4: Perform eigenvalue decomposition of X^T X
    covariance_matrix = np.dot(X_norm_T, X_norm)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Step 5: Select n eigenvectors corresponding to the largest n eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1][:n_components]
    selected_eigenvectors = eigenvectors[:, sorted_indices]

    # Step 6: Project the data points onto the subspace formed by the selected eigenvectors
    X_pca = np.dot(X_norm, selected_eigenvectors)

    return X_pca

def pca_sklearn(X, n_components):
    """
    Performs PCA on the input dataset X and returns the projected dataset X_pca as implemented in sklearn"""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca

def get_MNIST_pca_filtered_dataset(n, num_samples=40, pca=pca_sklearn, mnist=None, random_seed=1):
    """
    Gets the MNIST dataset, performs PCA on it, and filters for only 0 and 1.
    Returning the projected dataset X_pca and the corresponding labels y.
    If random_seed is not None, it will set the random seed for reproducibility.
    """
    np.random.seed(random_seed)

    # Load MNIST dataset
    if mnist is None:
        mnist = fetch_openml('mnist_784')
        print("fetching mnist")

    X = mnist.data
    y = mnist.target

    X_pca = pca(X, n)

    # Filter the dataset to include only "0" and "1" digits
    binary_digits = ['0', '1']
    binary_indices = np.isin(y, binary_digits)
    X_pca_binary = X_pca[binary_indices]
    y_binary = y[binary_indices]

    # Generate random indices
    indices = np.random.choice(len(y_binary), size=num_samples, replace=False)
    # Select elements using the random indices for both arrays
    y_binary_f = y_binary.reset_index(drop=True)[indices]
    y_binary_f = y_binary_f.to_numpy().astype(float)

    X_pca_binary = X_pca_binary[indices]

    return X_pca_binary, y_binary_f

from sklearn.datasets import fetch_openml



def get_filtered_MNIST_pca_dataset(n, num_samples= 40, pca=pca_sklearn):
    """
    gets the MNIST dataset, filters for only 0 and 1 and performs PCA on it. 
    Returning the projected dataset X_pca and the corresponding labels y
    """

    if mnist is None:
        mnist = fetch_openml('mnist_784')
    X = mnist.data
    y = mnist.target
    print("here")
    # Filter the dataset to include only "0" and "1" digits
    binary_digits = ['0', '1']
    binary_indices = np.isin(y, binary_digits)
    X_binary = X[binary_indices]
    y_binary = y[binary_indices]


    # Perform PCA on the filtered dataset
    X_pca = pca(X_binary, n)
    print(len(X_pca))
    print(len(y_binary))
    
    
     # Generate random indices
    indices = np.random.choice(len(X_pca), size=num_samples, replace=False)

    # Select elements using the random indices for both arrays
    X_pca = X_pca[indices]
    y_binary = y_binary.reset_index(drop=True)
    y_binary_f = y_binary[indices]

    return X_pca, y_binary_f

def hypercube_classifier(vector, half_width = np.pi):
    n = len(vector)
    width = 2 * half_width / 2**(1/n)
    for coordinate in vector:
        if coordinate < -width/2 or coordinate > width/2:
            return -1 
    return 1

def get_hypercube_dataset(num_qubits, num_samples=100, half_width=np.pi, random_seed=None):
    """
    Generates a dataset of num_samples points sampled uniformly from the hypercube [-1, 1]^num_qubits.
    If random_seed is not None, it will set the random seed for reproducibility.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    X = np.random.uniform(-half_width, half_width, (num_samples, num_qubits))
    y = np.array([hypercube_classifier(vector, half_width) for vector in X])
    return X, y

def filter_two_numbers(x, y, a, b):
    """
    Filtering for only a and b in the x and y dataset
    """
    keep = (y == a) | (y == b)
    x, y = x[keep], y[keep]
    y = y == 0
    return x,y



def truncate_x(x_train, x_test, n_components=10):
  """Performs PCA on image dataset keeping the top `n_components` components. As done in The Power of Data paper. 
See, https://www.tensorflow.org/quantum/tutorials/quantum_data
  
  """
  n_points_train = tf.gather(tf.shape(x_train), 0)
  n_points_test = tf.gather(tf.shape(x_test), 0)

  # Flatten to 1D
  x_train = tf.reshape(x_train, [n_points_train, -1])
  x_test = tf.reshape(x_test, [n_points_test, -1])

  # Normalize.
  feature_mean = tf.reduce_mean(x_train, axis=0)
  x_train_normalized = x_train - feature_mean
  x_test_normalized = x_test - feature_mean

  # Truncate.
  e_values, e_vectors = tf.linalg.eigh(
      tf.einsum('ji,jk->ik', x_train_normalized, x_train_normalized))
  return tf.einsum('ij,jk->ik', x_train_normalized, e_vectors[:,-n_components:]), \
    tf.einsum('ij,jk->ik', x_test_normalized, e_vectors[:, -n_components:])


def get_fMNIST_dataset(n_components=10, num_samples=100, fMNIST=None, filtered_03=True):
    """
    Gets the Fashion MNIST dataset, performs PCA on it, and filters for only "0" and "3" characters.
    returns x_train, y_train, x_test, y_test. 
    """
    if fMNIST is None: #No dataset and no path provided
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = fMNIST
    x_train, x_test = x_train, x_test
    
    if filtered_03:
        x_train, y_train = filter_two_numbers(x_train, y_train, 0, 3)
        x_test, y_test = filter_two_numbers(x_test, y_test, 0, 3)

    x_train, x_test = truncate_x(x_train/1.0, x_test/1.0, n_components=n_components)

    return x_train[:num_samples].numpy(), y_train[:num_samples], x_test[:int(num_samples)].numpy(), y_test[:int(num_samples)]


def get_KMNIST_pca_filtered_dataset(n_components, num_samples=40, kmnist=None, filtered_01=True):
    """
    Gets the KMNIST dataset, performs PCA on it, and filters for only "0" and "1" characters.
    Returning the projected dataset X_pca and the corresponding labels y.
    If random_seed is not None, it will set the random seed for reproducibility.
    """
    #if random_seed is not None:
    #    np.random.seed(random_seed)

    # Load KMNIST dataset
    if kmnist is None:
        train_dataset = tfds.load('kmnist', split='train[:80%]', as_supervised=True)
        test_dataset = tfds.load('kmnist', split='train[80%:]', as_supervised=True)
    else:
        train_dataset, test_dataset = kmnist



    # Extract data and labels for training set
    x_train = np.array([data.numpy() for data, _ in train_dataset])
    y_train = np.array([label.numpy() for _, label in train_dataset])

    # Extract data and labels for testing set
    x_test = np.array([data.numpy() for data, _ in test_dataset])
    y_test = np.array([label.numpy() for _, label in test_dataset])


    if filtered_01:
        x_train, y_train = filter_two_numbers(x_train, y_train, 0, 1)
        x_test, y_test = filter_two_numbers(x_test, y_test, 0, 1)
    
    x_train, x_test = truncate_x(x_train/1.0, x_test/1.0, n_components=n_components)

    

    return x_train[:num_samples].numpy(), y_train[:num_samples], x_test[:int(num_samples)].numpy(), y_test[:int(num_samples)]

def get_plasticc_PCA_dataset(n_components, num_samples, plasticc):
    """ dataset from https://arxiv.org/abs/2101.09581
    Normalized, downscaled to dataset_dim and truncated to n_train, n_test
    """
    if plasticc is None:
        raise ValueError("plasticc dataset not provided")
    else:
        data = plasticc

    X = data[:,:67]
    Y = data[:,67]
    
    x_train_normalized, x_test_normalized, y_train, y_test = train_test_split(X, Y, train_size=num_samples, test_size=num_samples, random_state=42, stratify=Y)
    scikit_pca = PCA(n_components=n_components)
    x_train = scikit_pca.fit_transform(x_train_normalized)
    x_test = scikit_pca.transform(x_test_normalized)
    return x_train, x_test, y_train, y_test



def get_train_test_data(dataset_name, num_qubits, num_samples, dataset_files = []):
    MNIST, fMNIST, kMNIST, plasticc = dataset_files 
    random_seed=1
    if dataset_name == "MNIST":    
        # Generate some training data
        X_train, y_train = get_MNIST_pca_filtered_dataset(num_qubits, num_samples = num_samples, pca=pca_sklearn, mnist = MNIST, random_seed = random_seed)
        x_test, y_test = get_MNIST_pca_filtered_dataset(num_qubits, num_samples = num_samples, pca=pca_sklearn, mnist = MNIST, random_seed = 2)
    elif dataset_name == "Hypercube":
        X_train, y_train = get_hypercube_dataset(num_qubits, num_samples = num_samples, half_width = np.pi, random_seed = random_seed)
        x_test, y_test = get_hypercube_dataset(num_qubits, num_samples = num_samples, half_width = np.pi, random_seed = 2)
    elif dataset_name == "fMNIST":
        X_train, y_train, x_test, y_test = get_fMNIST_dataset(num_qubits, num_samples, fMNIST)
    elif dataset_name == "kMNIST":
        X_train, y_train, x_test, y_test = get_KMNIST_pca_filtered_dataset(num_qubits, num_samples, kmnist=kMNIST)
    elif dataset_name == "plasticc":
        X_train, x_test, y_train, y_test = get_plasticc_PCA_dataset(num_qubits, num_samples, plasticc)
        
    return X_train, y_train, x_test, y_test