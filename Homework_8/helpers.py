import numpy as np

def dist_mat_vec(M, vec):
    # Compute distance between each row of 'M' with 'vec'
    # method: 'ncc', 'dot', 'ssd'
    # M : ndarray ( _ x k); vec: (1 x k)
    # Returns a 1D numpy array of distances.
    return np.linalg.norm(M - vec, axis = 1)

def dist_mat_mat(M1, M2):
    # M1, M2 --> ndarray (y1 x k) and (y2 x k)
    # Returns y1 x y2 ndarray with the distances.
    # If y1 and y2 are huge, it might run into MemoryError
    D = np.zeros((M1.shape[0], M2.shape[0]))
    for idx2 in range(M2.shape[0]):
        D[:, idx2] = dist_mat_vec(M1, M2[idx2, :])
    return D
