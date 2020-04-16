import numpy as np
from scipy.sparse.linalg import eigsh
from warnings import warn

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

class SineBasis():
    
    def __init__(self, G=500, h=1.0, l=10):
        """ G number of grid points, 
            h spacing of grid points, 
            l number of sine waves"""
        self.l = l
        x = np.arange(G)*h
        l_vec = np.arange(1, l+1)
        # normalized such that the sum over the squared basis function
        # equals 1
        self.basis_fun = np.sin(np.pi*np.outer(x, l_vec))*np.sqrt(2*h)
        # The norm defined earlier ensures that the eigenvalues of the
        # projection matrix are 1.
        self.P = self.basis_fun.dot(self.basis_fun.T)
    
    def __call__(self, n):
        #n_basis = params['sines_n']
        #basis = np.sqrt(2)*np.sin(dataset.x[np.newaxis, :]*np.pi*np.arange(1, n_basis+1)[:, np.newaxis])
        #project = lambda value: normalize(dataset.h*tf.matmul(tf.matmul(basis, value, transpose_b=True), basis, transpose_a=True), h)*np.sqrt(params['N'])
        return self.P

    
def run_PCA_minimization(n, V, model, projection, h, 
                         eta=1e-3, max_steps=100, 
                         g_tol=1e-6, i_print=0):

    for i in range(max_steps):
        T_pred, dT_pred = model.predict(n.reshape(1,-1), derivative=True)    
        dT_pred = dT_pred.flatten()
        grad = projection(n).dot(V + dT_pred)
        grad_norm = np.sum(np.abs(grad))*h

        if grad_norm < g_tol:
            break
    
        if i_print > 0:
            if i%i_print == 0:
                print(i, grad_norm)

        n = n - eta*grad
    else:
        warn('Not converged within max_steps')
    return n


def run_projected_dens_minimization(n, V, model, projection, h, 
                         eta=1e-3, max_steps=100, 
                         g_tol=1e-6, i_print=0):

    for i in range(max_steps):
        T_pred, dT_pred = model.predict(n.reshape(1,-1), derivative=True)    
        dT_pred = dT_pred.flatten()
        P = projection(n)
        grad_proj = P.dot(V + dT_pred)
        # What does a constant function projected look like:
        proj_norm = np.sum(P, axis=-1)
        
        mu = np.sum(grad_proj)/np.sum(proj_norm)
        
        grad = grad_proj - mu*proj_norm
        grad_norm = np.sum(np.abs(grad))*h

        if grad_norm < g_tol:
            break
    
        if i_print > 0:
            if i%i_print == 0:
                print(i, grad_norm)

        n = n - eta*grad
    else:
        warn('Not converged within max_steps')
    return n


def run_projected_wfn_minimization(n, V, model, projection, h,
                                   eta=1e-3, max_steps=100, 
                                   g_tol=1e-6, i_print=0):
    # Calculate pseudo wave function phi
    phi = np.sqrt(n)
    
    for i in range(max_steps):
        T, dT_dn = model.predict(n.reshape(1,-1), derivative=True)    
        dT_dn = dT_dn.flatten()
        
        P = projection(phi)
        proj_grad = P.dot(phi*(dT_dn + V))
        proj_phi = P.dot(phi)

        mu = np.sum(phi*proj_grad)/np.sum(phi*proj_phi)
        grad = 2*(proj_grad - mu*proj_phi)

        # Norm of gradient
        grad_norm = np.sum(np.abs(grad))*h            

        if grad_norm < g_tol:
            break   
            
        if i_print > 0:
            if i%i_print == 0:
                print(i, grad_norm)    

        phi = phi - eta*grad     
        n = phi**2     
        
    else:
        warn('Not converged within max_steps')
    return n