import numpy as np
from scipy.sparse.linalg import eigsh

class PCA():
    
    def __init__(self, n_train, m=30, l=5):
        self.n_train = n_train
        self.m = m
        self.l = l
        
    def __call__(self, n, use_eigsh=True):
        # find m closest training densities
        X = self.n_train - n
        indices = np.argsort(np.sum(X**2, axis=-1))[:self.m]
        C = X[indices].T.dot(X[indices])/self.m
        if use_eigsh:
            _, V = eigsh(C, k=self.l, which='LM')
        else:
            _, V = np.linalg.eigh(C)
        # select l largest eigenvalues and construct 
        # projection matrix. This looks different to 
        # Synder et al because the eigenvectors are
        # sorted differently
        P = V[:, -self.l:].dot(V[:, -self.l:].T)
        return P
    
def run_PCA_minimization(n, V, model, projection, eta=1e-3, max_steps=100, g_tol=1e-6, i_print=0):

    for i in range(max_steps):
        T_pred, dT_pred = model.predict(n.reshape(1,-1), derivative=True)    
        dT_pred = dT_pred.flatten()
        grad = projection(n).dot(V + dT_pred)
        grad_norm = np.sum(np.abs(grad))*model.h

        if grad_norm < g_tol:
            break
    
        if i_print > 0:
            if i%i_print == 0:
                print(i, grad_norm)

        n = n - eta*grad
    return n