import numpy as np
from scipy.linalg import cho_solve, cholesky
from odft_tools.kernels import RBFKernel

class CustomKRR():
    
    def __init__(self, kernel=RBFKernel(), lamb=1e-6, kappa=1e-6, h=1.0):
        self.kernel = kernel
        self.lamb = lamb
        self.kappa = kappa
        self.h = h
        
    def fit(self, X_train, Y, dY_dX=None, offset=None, inplace=True):
        n = X_train.shape[0]
        n_dim = X_train.shape[1]
        self.X_train = X_train
        self.fit_deriv = not dY_dX is None
        
        try:
            self.offset = float(offset)
        except (ValueError, TypeError):
            if offset is None:
                self.offset = 0.0
            elif offset == 'mean':
                self.offset = np.mean(Y)
            elif offset == 'max':
                self.offset = np.max(Y)
            elif isinstance(offset, str) and offset.startswith('max+'):
                self.offset = (np.max(Y) + float(offset[4:]))
            else:
                raise NotImplementedError(
                    'Unknown option: %s' % offset)
        
        if self.fit_deriv:
            K = self.kernel(X_train, X_train, dx=True, dy=True, h=self.h)
            #K[:n, n:] /= self.h
            #K[n:, :n] /= self.h
            #K[n:, n:] /= self.h**2
            K[np.diag_indices(n*(1+n_dim))] += np.concatenate(
                [self.lamb*np.ones(n), self.lamb*n_dim/self.kappa*np.ones(n*n_dim)])
            target_vector = np.concatenate([Y - self.offset, dY_dX.flatten()])
        else:
            K = self.kernel(X_train, X_train)
            K[np.diag_indices(n)] += self.lamb*np.ones(n)
            target_vector = Y - self.offset
        if inplace:
            # Using K.T and lower=False instead of K and lower=True for overwrite_a to work
            # check: https://stackoverflow.com/questions/14408873/how-to-do-in-place-cholesky-factorization-in-python
            cholesky(K.T, lower=False, overwrite_a=True)
        else:
            # Somehow the scipy version segfaults for large arrays, fallback to numpy:
            K = np.linalg.cholesky(K)
        self.alpha = cho_solve((K, True), target_vector)
        
        return self
    
    def predict(self, X, derivative=False):
        n = self.X_train.shape[0]
        m = X.shape[0]
        K_star = self.kernel(self.X_train, X, dx=self.fit_deriv, dy=derivative, h=self.h)
        #K_star[:n, m:] /= self.h
        #K_star[n:, :m] /= self.h
        #K_star[n:, m:] /= self.h**2
        out = self.alpha.dot(K_star)
        if derivative:
            return self.offset + out[:m], out[m:].reshape(m, -1)
        return self.offset + out
    
    def save(self, save_file):
        np.savez(save_file, X_train=self.X_train, alpha=self.alpha,
                 offset=self.offset, h=self.h, lamb=self.lamb, 
                 kappa=self.kappa, fit_deriv=self.fit_deriv)
        
    def load(self, save_file):
        data = np.load(save_file)
        self.X_train = data['X_train']
        self.alpha = data['alpha']
        self.offset = data['offset']
        self.h = data['h']
        self.lamb = data['lamb']
        self.kappa = data['kappa']
        self.fit_deriv = data['fit_deriv']
   
    
class CustomKRR_old():
    
    def __init__(self, kernel=RBFKernel(), lamb=1e-6):
        self.kernel = kernel
        self.lamb = lamb
        
    def fit(self, X_train, y=None, dy=None, offset=None):
        n = X_train.shape[0]
        self.X_train = X_train
        K = self.kernel(X_train, X_train)
        L = cholesky(K + self.lamb*np.eye(len(K)), lower=True)

        if offset == 'mean':
            self.offset = np.mean(y)
        else:
            self.offset = 0.0
        target_vector = y - self.offset
        self.alpha = cho_solve((L, True), target_vector)
        
        return self
    
    def predict(self, X, derivative=False):
        n = self.X_train.shape[0]
        m = X.shape[0]
        K_star = self.kernel(self.X_train, X, dy=derivative)
        out = self.alpha.dot(K_star)
        if derivative:            
            return self.offset + out[:m], out[m:].reshape(m, -1)
        return self.offset + out