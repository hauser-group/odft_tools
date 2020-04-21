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
        mat[0, :3] = (-3/2, 2, -1/2)
        mat[-1, -3:] = (1/2, -2, 3/2)
        mat /= h
        return mat
    elif method == 'five_point':
        mat = (np.diag(1/12*np.ones(G-2), k=-2)
               + np.diag(-2/3*np.ones(G-1), k=-1) 
               + np.diag(2/3*np.ones(G-1), k=1)
               + np.diag(-1/12*np.ones(G-2), k=2))
        mat[0, :5] = (-25/12, 4, -3, 16/12, -3/12)
        mat[1, :5] = (-3/12, -10/12, 18/12, -6/12, 1/12)
        mat[-2, -5:] = (-1/12, 6/12, -18/12, 10/12, 3/12)
        mat[-1, -5:] = (3/12, -16/12, 3, -4, 25/12)
        mat /= h
        return mat
    else:
        raise NotImplemented()

def second_derivative_matrix(G, h, method='three_point'):
    if method == 'three_point':
        mat = (np.diag(np.ones(G-1), k=-1)
               - 2.0*np.eye(G) 
               + np.diag(np.ones(G-1), k=1))
        mat[0, :4] = (2, -5, 4, -1)
        mat[-1, -4:] = (-1, 4, -5, 2)
        mat /= h**2    
        return mat
    if method == 'five_point':
        mat = (np.diag(-1/12*np.ones(G-2), k=-2)
               + np.diag(4/3*np.ones(G-1), k=-1)
               - 5/2*np.eye(G) 
               + np.diag(4/3*np.ones(G-1), k=1)
               + np.diag(-1/12*np.ones(G-2), k=2))
        #mat[0, :5] = (35/12, -26/3, 19/2, -14/3, 11/12)
        #mat[1, :5] = (11/12, -20/12, 6/12, 4/12, -1/12)
        #mat[-2, -5:] = (-1/12, 4/12, 6/12, -20/12, 11/12)
        #mat[-1, -5:] = (11/12, -14/3, 19/2, -26/3, 35/12)
        mat[0, :6] = (45/12, -154/12., 214/12, -156/12, 61/12, -10/12)
        mat[1, :6] = (10/12, -15/12, -4/12, 14/12, -6/12, 1/12)
        mat[-2, -6:] = (1/12, -6/12, 14/12, -4/12, -15/12, 10/12)
        mat[-1, -6:] = (-10/12, 61/12, -156/12, 214/12, -154/12, 45/12)
        mat /= h**2    
        return mat
    else:
        raise NotImplemented()