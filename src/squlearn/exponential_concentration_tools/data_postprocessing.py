import numpy as np



def variance_off_diagonal(M):
    """
    Remove main diagonal of M, transform M to a 1D array and calculates the variance of this array.
    """
    #removes main diagonal:
    M = M[~np.eye(M.shape[0],dtype=bool)].reshape(M.shape[0],-1)
    M = M.flatten()
    variance = np.var(M)
    return variance