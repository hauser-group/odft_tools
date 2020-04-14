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
        x = np.arange(G)*h
        l_vec = np.arange(1, l+1)
        # normalized such that the sum over the squared basis function
        # equals 1
        norm = np.sqrt(1/2 - np.sin(2*l*np.pi)/(4*l*np.pi))/np.sqrt(h)
        self.basis_fun = np.sin(np.pi*np.outer(x, l_vec))/norm
        # The norm defined earlier ensures that the eigenvalues of the
        # projection matrix are 1.
        self.P = self.basis_fun.dot(self.basis_fun.T)
    
    def __call__(self, n):
        #n_basis = params['sines_n']
        #basis = np.sqrt(2)*np.sin(dataset.x[np.newaxis, :]*np.pi*np.arange(1, n_basis+1)[:, np.newaxis])
        #project = lambda value: normalize(dataset.h*tf.matmul(tf.matmul(basis, value, transpose_b=True), basis, transpose_a=True), h)*np.sqrt(params['N'])
        return self.P

    
def run_PCA_minimization(n, V, model, projection, h, eta=1e-3, max_steps=100, g_tol=1e-6, i_print=0):

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

def run_projected_wfn_minimization(n, V, model, projection, h,
                                   eta=1e-3, max_steps=100, 
                                   g_tol=1e-6, i_print=0,
                                   update='fixed_mu'):
    # Calculate total number of electrons to which the density should
    # be normalized
    N = np.round(np.sum(n)*h)
    # Calculate pseudo wave function phi
    phi = np.sqrt(n)
    # Start value for mu
    mu = 0.
    
    for i in range(max_steps):
        T, dT_dn = model.predict(n.reshape(1,-1), derivative=True)    
        dT_dn = dT_dn.flatten()
        
        if update == 'no_mu':
            grad = 2*phi*(dT_dn + V)   
        elif update == 'fixed_mu':
            E = T[0] + np.sum(V*n)*h    
            mu = E/N
            grad = 2*phi*(dT_dn + V - mu)             
        elif update == 'iterative_mu':
            grad = 2*phi*(dT_dn + V - mu)
            while True:
                phi_tmp = phi - eta*grad
                phi_tmp /= np.sqrt(np.sum(phi_tmp**2)*h)/np.sqrt(N)

                mu_old = mu
                # h cancels out, sum(phi*phi) should be very close to 1...
                mu = (np.sum(phi_tmp*phi) 
                       - np.sum(phi*phi) 
                       + 2*eta*np.sum(phi*(dT_dn + V)*phi)
                      )/(2*eta*np.sum(phi*phi))
                grad = 2*phi*(dT_dn + V - mu)
                if abs(mu_old - mu) < 1e-6:
                    break
        else:
            raise NotImplemented()

        grad = projection(phi).dot(grad)
        # Norm of gradient with respect to n (divide by 2 phi)
        grad_norm = np.sum(np.abs(grad/(2*(phi+1e-30)**2)*phi))*h            

        if grad_norm < g_tol:
            break
            
        if i_print > 0:
            if i%i_print == 0:
                print(i, grad_norm)

        phi = phi - eta*grad
        phi /= np.sqrt(np.sum(phi**2)*h)/np.sqrt(N)           
        n = phi**2
    else:
        warn('Not converged within max_steps')
    return n