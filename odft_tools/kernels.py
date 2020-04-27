import numpy as np

class LinearKernel():
    """Only used for comparison to standard ridge regression"""
    
    def __call__(self, X, Y, dy=False):
        n = X.shape[0]
        m = Y.shape[0]
        n_dim = X.shape[1]
        
        K = np.einsum('ik,jk->ij', X, Y)
        if not dy:
            return K
        return np.concatenate([K, np.repeat(X[:, np.newaxis, :], m, axis=1).reshape(n, m*n_dim)], axis=1)

    
class RBFKernel():

    def __init__(self, length_scale=1.0, scale=1.0, constant=0.0):
        self.length_scale = length_scale
        self.scale = scale
        self.constant = constant
        
    def __call__(self, X, Y, dx=False, dy=False, h=1.0):
        n = X.shape[0]
        m = Y.shape[0]
        n_dim = X.shape[1]
        
        K = np.zeros((n*(1 + int(dx)*n_dim), m*(1 + int(dy)*n_dim)))
        lh = self.length_scale*h
        lh2 = lh**2
        
        # Doing this in a loop to avoid massive memory overhead
        for i in range(n):
            for j in range(m):
                # Index ranges for the derivatives are given by the following
                # slice objects:
                di = slice(n + i*n_dim, n + (i + 1)*n_dim, 1)
                dj = slice(m + j*n_dim, m + (j + 1)*n_dim, 1)
                scaled_diff = (X[i, :] - Y[j, :])/self.length_scale
                K[i, j] = self.scale * np.exp(-.5*np.dot(scaled_diff, scaled_diff))
                if dy:
                    K[i, dj] = K[i, j]*scaled_diff/lh
                    if dx:
                        K[di, j] = -K[i, dj]
                        K[di, dj] = K[i, j]/lh2 * (
                            np.eye(n_dim) - np.outer(scaled_diff, scaled_diff))
                else:
                    if dx:
                        K[di, j] = -K[i, j]*scaled_diff/lh
        K[:n, :m] += self.constant
        return K