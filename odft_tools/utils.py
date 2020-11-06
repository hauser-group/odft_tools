import numpy as np
import tensorflow as tf

from tensorflow.python.framework import dtypes

def integrate(fun, h, method='trapezoidal'):
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
        mat[-1, -3:] = -np.flip(mat[0, :3])    
    elif method == 'five_point':
        mat = (np.diag(1/12*np.ones(G-2), k=-2)
               + np.diag(-2/3*np.ones(G-1), k=-1) 
               + np.diag(2/3*np.ones(G-1), k=1)
               + np.diag(-1/12*np.ones(G-2), k=2))
        mat[0, :5] = (-25/12, 4, -3, 16/12, -3/12)
        mat[1, :5] = (-3/12, -10/12, 18/12, -6/12, 1/12)
        mat[-2, -5:] = -np.flip(mat[1, :5])
        mat[-1, -5:] = -np.flip(mat[0, :5])
    else:
        raise NotImplemented()
    mat /= h
    return mat

def second_derivative_matrix(G, h, method='three_point'):
    if method == 'three_point':
        mat = (np.diag(np.ones(G-1), k=-1)
               - 2.0*np.eye(G) 
               + np.diag(np.ones(G-1), k=1))
        mat[0, :4] = (2, -5, 4, -1)
        mat[-1, -4:] = np.flip(mat[0, :4])
    elif method == 'five_point':
        mat = (np.diag(-1/12*np.ones(G-2), k=-2)
               + np.diag(4/3*np.ones(G-1), k=-1)
               - 5/2*np.eye(G) 
               + np.diag(4/3*np.ones(G-1), k=1)
               + np.diag(-1/12*np.ones(G-2), k=2))
        mat[0, :6] = (45/12, -154/12, 214/12, -156/12, 61/12, -10/12)
        mat[1, :6] = (10/12, -15/12, -4/12, 14/12, -6/12, 1/12)
        mat[-2, -6:] = np.flip(mat[1, :6])
        mat[-1, -6:] = np.flip(mat[0, :6])
    else:
        raise NotImplemented()
    mat /= h**2    
    return mat

def gen_gaussian_kernel_v1_1D(shape, mean, stddev, dtype=dtypes.float32):
    """ Returns a tensor object cotnaining gaussian kernels
    Args:
      shape: Shape of the tensor.
      mean: mean of gaussian
      stddev: stddev of gaussian
      dtype: Optional dtype of the tensor. Only floating point types are
         supported.
    """ 

    kernel_size = shape[0]
    input_size = shape[1]
    filter_size = shape[2]

    gaus_kerne_count =  input_size * filter_size

    truncate = kernel_size/2

    width = int(truncate + 0.5)
    # width = int(truncate * stddev + 0.5)
    # width = truncate
    support = np.arange(-width, width + 1)
    
    gauss_kernels = []


    center = int(len(support)/2)
    left_cut = center - int(kernel_size/2)
    right_cut = center + int(kernel_size/2)

    for i in range(gaus_kerne_count):
        gauss_kernel = np.exp(-((support - mean) ** 2)/(2*stddev ** 2))
        gauss_kernel = gauss_kernel / gauss_kernel.sum()

        if (kernel_size % 2) != 0:
            gauss_kernel = gauss_kernel[left_cut + 1:right_cut + 2]
        else:
            gauss_kernel = gauss_kernel[left_cut + 1:right_cut + 1]
        gauss_kernels.append(gauss_kernel)

    gauss_kernels = np.reshape(
      gauss_kernels, (
        filter_size,
        input_size,
        kernel_size
        ) 
      ).T

    gauss_kernels = tf.convert_to_tensor(
        value=gauss_kernels,
        dtype=dtype)
    return gauss_kernels
