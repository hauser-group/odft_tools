import numba
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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

def gen_gaussian_kernel_v2_1D(shape, weights, dtype=dtypes.float32):
    """ Returns a tensor object cotnaining gaussian kernels
    Args:
      shape: Shape of the tensor.
      weights: mean and stddev of gaussian
      dtype: Optional dtype of the tensor. Only floating point types are
         supported.
    """
    means = tf.reshape(weights[0, :, :], [-1])
    stddevs = tf.reshape(weights[1, :, :], [-1])
    
    means = means
    stddevs = stddevs
    kernel_size = shape[0]
    input_size = shape[1]
    filter_size = shape[2]

    gaus_kernel_count =  input_size * filter_size

    truncate = kernel_size/2

    width = int(truncate + 0.5)

    support = tf.convert_to_tensor(
      value=np.arange(-width, width + 1),
      dtype=dtype
    )
    
    gauss_kernels = tf.TensorArray(dtype, size=0, dynamic_size=True)


    center = int(len(support)/2)
    left_cut = center - int(kernel_size/2)
    right_cut = center + int(kernel_size/2)

    for i in tf.range(gaus_kernel_count):
        mean = means[i]
        stddev = stddevs[i]
        gauss_kernel = tf.math.exp(-((support - mean) ** 2)/(2*stddev ** 2))
        gauss_kernel = gauss_kernel / tf.math.reduce_sum(gauss_kernel)        

        if (kernel_size % 2) != 0:
            gauss_kernel = gauss_kernel[left_cut + 1:right_cut + 2]
        else:
            gauss_kernel = gauss_kernel[left_cut + 1:right_cut + 1]
        gauss_kernels = gauss_kernels.write(gauss_kernels.size(), gauss_kernel)

    gauss_kernels = gauss_kernels.stack()
    gauss_kernels = tf.transpose(
      tf.reshape(
        gauss_kernels, 
        (
          filter_size,
          input_size,
          kernel_size
        )
      )
    )

    gauss_kernels = tf.convert_to_tensor(
        value=gauss_kernels,
        dtype=dtype)
    return gauss_kernels

signature = (
    numba.int64[:],
    numba.int64,
    numba.int64,
    numba.float64,
    numba.float64
)

# @numba.njit(signature)
def calc_gaussians(support, kernel_size, gaus_kernel_count, mean, stddev):
    gauss_kernels = []

    center = int(len(support)/2)
    left_cut = center - int(kernel_size/2)
    right_cut = center + int(kernel_size/2)

    for i in range(gaus_kernel_count):
        gauss_kernel = np.exp(-((support - mean + np.random.normal(5, 3)) ** 2)/(2*(stddev + np.random.normal(5, 3)) ** 2))
        
        # if (i == 0) or (i == (gaus_kernel_count - 1)):
        #     gauss_kernel = np.exp(-((support - mean + np.random.normal(5, 3)) ** 2)/(2*(stddev + np.random.normal(5, 3)) ** 2))
        #     # gauss_kernel = gauss_kernel / gauss_kernel.sum()
        # else:
        #     gauss_kernel = np.random.uniform(size=len(support))
        gauss_kernel = gauss_kernel / gauss_kernel.sum()
        if (kernel_size % 2) != 0:
            gauss_kernel = gauss_kernel[left_cut + 1:right_cut + 2]
        else:
            gauss_kernel = gauss_kernel[left_cut + 1:right_cut + 1]
        gauss_kernels.append(gauss_kernel)
    return gauss_kernels

def gen_gaussian_kernel_v1_1D(shape, weights, dtype=dtypes.float32):
    """ Returns a tensor object cotnaining gaussian kernels
    Args:
      shape: Shape of the tensor.
      mean: mean of gaussian
      stddev: stddev of gaussian
      dtype: Optional dtype of the tensor. Only floating point types are
         supported.
    """ 
    mean = weights[0]
    stddev = weights[1]
    kernel_size = shape[0]
    input_size = shape[1]
    filter_size = shape[2]

    gaus_kernel_count =  input_size * filter_size

    truncate = kernel_size/2

    width = int(truncate + 0.5)
    # width = int(truncate * stddev + 0.5)
    support = np.arange(-width, width + 1)
    
    gauss_kernels = calc_gaussians(support, kernel_size, gaus_kernel_count, mean, stddev)

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
