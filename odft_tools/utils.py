import numpy as np

def integrate(fun, h, method='simple'):
    if method == 'simple':
        return np.sum(fun, axis=-1)*h
    elif method == 'trapezoidal':
        return 0.5*np.sum(fun[..., :-1] + fun[..., 1:], axis=-1)*h
    else:
        raise NotImplemented()

def first_derivative_matrix(G, h, method='three_point'):
    if method == 'three_point':
        mat = (np.diag(-0.5*np.ones(G-1), k=-1)
               + np.diag(0.5*np.ones(G-1), k=1))
        mat[0, :2] = (-1, 1)
        mat[-1, -2:] = (1, -1)
        mat /= h
        return mat
    elif method == 'five_point':
        mat = (np.diag(1/12*np.ones(G-2), k=-2)
               + np.diag(-2/3*np.ones(G-1), k=-1) 
               + np.diag(2/3*np.ones(G-1), k=1)
               + np.diag(-1/12*np.ones(G-2), k=2))
        mat[0, :3] = (-3/2, 2., -0.5)
        mat[1, :4] = (-1/3, -0.5, 1., -1/6)
        mat[-2, -4:] = (1/6, -1., 0.5, 1/3)
        mat[-1, -3:] = (0.5, -2., 3/2)
        mat /= h
        return mat
    else:
        raise NotImplemented()

def second_derivative_matrix(G, h, method='three_point'):
    if method == 'three_point':
        mat = (np.diag(np.ones(G-1), k=-1)
               - 2.0*np.eye(G) 
               + np.diag(np.ones(G-1), k=1))
        mat[0, :3] = (1, -2, 1)
        mat[-1, -3:] = (1, -2, 1)
        mat /= h**2    
        return mat
    if method == 'five_point':
        mat = (np.diag(-1/12*np.ones(G-2), k=-2)
               + np.diag(4/3*np.ones(G-1), k=-1)
               - 2.5*np.eye(G) 
               + np.diag(4/3*np.ones(G-1), k=1)
               + np.diag(-1/12*np.ones(G-2), k=2))
        mat[0, :3] = (1., -2., 1.)
        mat[1, :4] = (1., -2., 1., 0.)
        mat[-2, -4:] = (0., 1., -2., 1.)
        mat[-1, -3:] = (1., -2., 1.)
        mat /= h**2    
        return mat
    else:
        raise NotImplemented()