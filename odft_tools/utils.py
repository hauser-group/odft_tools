import os
import numba
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

from tensorflow.python.framework import dtypes
from scipy.special import (
    wofz,
    factorial
)

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
    def create_res_net_model(self):
        dens
    def create_res_net_model(self):
        dens
    return mat

def gen_gaussian_kernel_v1_1D(shape, weights, dtype=dtypes.float32, random_init=False):
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

    # Distribute uniform means and stddev
    # lenght is kernel size times input size
    if random_init:
        means = np.random.uniform(
            low=0.0,
            high=mean * 2,
            size=gaus_kernel_count
        )
        stddevs = np.random.uniform(
            low=stddev,
            high=stddev * 2,
            size=gaus_kernel_count
        )
    else:
        # Distribute means and stddevs around given mean and stddev
        # random uniform
        means = [mean] * gaus_kernel_count + np.random.uniform(
            low=-mean / 2,
            high=mean / 2,
            size=gaus_kernel_count
        )

        stddevs = [stddev] * gaus_kernel_count + np.random.uniform(
            low=-stddev / 2,
            high=stddev / 2,
            size=gaus_kernel_count
        )

    # calc gaussian kernel
    gauss_kernels = calc_gaussians(
        kernel_size,
        gaus_kernel_count,
        means,
        stddevs,
        random_init
    )

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

def lorentz_dist(support, omega_0, gamma):
    lorentz_kernel = 1/((support ** 2 - omega_0 ** 2) + (gamma ** 2) * (omega_0 ** 2))
    return lorentz_kernel

def cauchy_dist(support, s, t):
    cauchy_kernel = (1/np.pi)*s/(s ** 2 + (support - t) ** 2)
    return cauchy_kernel

def gaussian_dist(support, mean, stddev):
    gauss_kernel =  tf.math.exp(-((support - mean) ** 2)/(2*stddev ** 2))
    gauss_kernel = gauss_kernel / tf.math.reduce_sum(gauss_kernel)
    return gauss_kernel

def generate_kernel(shape, weights, kernel_dist, dtype=dtypes.float32, random_init=False):
    weights_0 = tf.reshape(weights[0, :, :], [-1])
    weights_1 = tf.reshape(weights[1, :, :], [-1])

    kernel_size = shape[0]
    input_size = shape[1]
    filter_size = shape[2]

    kernel_count =  input_size * filter_size

    truncate = kernel_size/2
    width = int(truncate + 0.5)
    kernels = tf.TensorArray(dtype, size=0, dynamic_size=True)

    for i in tf.range(kernel_count):

        shift = 0  # np.random.randint(-kernel_size/2, kernel_size/2)
        support = tf.convert_to_tensor(
            value=np.arange(
                -width - shift,
                width + 1 - shift
                ),

            dtype=dtype
        )
        center = int(len(support)/2)
        left_cut = center - int(kernel_size/2)
        right_cut = center + int(kernel_size/2)

        weight_0 = weights_0[i]
        weight_1 = weights_1[i]

        kernel = kernel_dist(support, weight_0, weight_1)

        if (kernel_size % 2) != 0:
            kernel = kernel[left_cut + 1:right_cut + 2]
        else:
            kernel = kernel[left_cut + 1:right_cut + 1]
        kernels = kernels.write(kernels.size(), kernel)

    kernels = kernels.stack()
    kernels = tf.transpose(
        tf.reshape(
            kernels,
            (
            filter_size,
            input_size,
            kernel_size
            )
        )
    )

    kernels = tf.convert_to_tensor(
        value=kernels,
        dtype=dtype)
    return kernels

def calc_trigonometrics(kernel_size, kernel_count):
    trigonometris = []
    x = np.linspace(-kernel_size/2, kernel_size/2, kernel_size)
    trig_max = 0.05
    trig_min = -0.05
    for i in range(kernel_count):
        # if random.random() > 0.5:
        y = (random.random() * np.exp(-random.random()) * np.sin(random.random() * x)/(x) + random.random() * np.cos(random.random() * x)*np.exp(-random.random()))
        # y = random.random() * np.sin(x * random.random() + random.randint(0, 50)) / x
        # else:
        #     y = random.random() * np.cos(x * random.random() + random.randint(0, 50)) / x
        y = (trig_max - trig_min) * (y - min(y)/(max(y) - min(y))) + trig_min

        trigonometris.append(y)
    return trigonometris

def get_psi(v, q, Hr):
    m = random.randrange(0, 10)
    omega = random.randrange(0, 10)
    n = (m*omega)/np.sqrt(np.sqrt(np.pi)*2**v*factorial(v))
    return n*Hr[v](q)*np.exp(-q*q/2.)

def get_turning_points(v, get_E):
    qmax = np.sqrt(2. * get_E(v + 0.5))
    return -qmax, qmax

def calc_harmonic(kernel_size, kernel_count):
    harmonics = []
    get_E = lambda v: v + 0.5

    hr = [None] * (kernel_count + 1)
    hr[0] = np.poly1d([1.,])
    hr[1] = np.poly1d([2., 0.])
    max_rang = 24
    for v in range(2, kernel_count + 1):
        if v > max_rang:
            v = max_rang
        hr[v] = hr[1]*hr[v-1] - 2*(v-1)*hr[v-2]

    qmin, qmax = get_turning_points(kernel_count, get_E)
    q = np.linspace(qmin, qmax, kernel_size)
    har_max = 0.05
    har_min = -0.05
    for v in range(kernel_count):
        if v > max_rang:
            v = max_rang
        har = get_psi(v, q, hr)

        # har = (har_max - har_min) * (har - min(har)/(max(har) - min(har))) + har_min

        harmonics.append(har/100)
    return harmonics

# @numba.njit(signature)
def calc_gaussians(kernel_size, gaus_kernel_count, means, stddevs, random_init):
    gauss_kernels = []

    truncate = kernel_size/2

    width = int(truncate + 0.5)

    for i in range(gaus_kernel_count):
        if random_init:
            shift_gaussian = np.random.randint(-kernel_size/4, kernel_size/4)
        else:
            shift_gaussian = 0
        support = np.arange(
            -width - shift_gaussian,
            width + 1 - shift_gaussian)

        center = int(len(support)/2)
        left_cut = center - int(kernel_size/2)
        right_cut = center + int(kernel_size/2)

        gauss_kernel = np.exp(-((support - means[i]) ** 2)/(2*(stddevs[i]) ** 2))
        gauss_kernel = gauss_kernel / gauss_kernel.sum()

        if (kernel_size % 2) != 0:
            gauss_kernel = gauss_kernel[left_cut + 1:right_cut + 2]
        else:
            gauss_kernel = gauss_kernel[left_cut + 1:right_cut + 1]
        gauss_kernel = gauss_kernel - 0.04
        gauss_kernels.append(gauss_kernel)
    return gauss_kernels

def gen_trigonometrics_kernel_v1_1D(shape, weights, dtype=dtypes.float32, random_init=False):
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

    kernel_count =  input_size * filter_size

    # calc  kernel
    trigonom_kernels = calc_trigonometrics(
        kernel_size,
        kernel_count
    )

    trigonom_kernels = np.reshape(
        trigonom_kernels, (
        filter_size,
        input_size,
        kernel_size
        )
      ).T

    trigonom_kernels = tf.convert_to_tensor(
        value=trigonom_kernels,
        dtype=dtype)

    return trigonom_kernels

def gen_harmonic_kernel_v1_1D(shape, weights, dtype=dtypes.float32, random_init=False):
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

    kernel_count =  input_size * filter_size

    # calc harmonic kernel
    harmonic_kernels = calc_harmonic(
        kernel_size,
        kernel_count
    )

    harmonic_kernels = np.reshape(
      harmonic_kernels, (
        filter_size,
        input_size,
        kernel_size
        )
      ).T

    harmonic_kernels = tf.convert_to_tensor(
        value=harmonic_kernels,
        dtype=dtype)

    return harmonic_kernels

# def gen_gaussian_kernel_v1_1D(shape, weights, dtype=dtypes.float32, random_init=False):
#     """ Returns a tensor object cotnaining gaussian kernels
#     Args:
#       shape: Shape of the tensor.
#       mean: mean of gaussian
#       stddev: stddev of gaussian
#       dtype: Optional dtype of the tensor. Only floating point types are
#          supported.
#     """
#     mean = weights[0]
#     stddev = weights[1]
#     kernel_size = shape[0]
#     input_size = shape[1]
#     filter_size = shape[2]

#     gaus_kernel_count =  input_size * filter_size

#     # Distribute unoform means and stddev
#     # lenght is kernel size times input size
#     if random_init:
#         means = np.random.uniform(
#             low=0.0,
#             high=mean * 2,
#             size=gaus_kernel_count
#         )
#         stddevs = np.random.uniform(
#             low=stddev,
#             high=stddev * 2,
#             size=gaus_kernel_count
#         )
#     else:
#         # Distribute means and stddevs around given mean and stddev
#         # random uniform
#         means = [mean] * gaus_kernel_count + np.random.uniform(
#             low=-mean / 2,
#             high=mean / 2,
#             size=gaus_kernel_count
#         )

#         stddevs = [stddev] * gaus_kernel_count + np.random.uniform(
#             low=-stddev / 2,
#             high=stddev / 2,
#             size=gaus_kernel_count
#         )

#     # calc gaussian kernel
#     gauss_kernels = calc_gaussians(
#         kernel_size,
#         gaus_kernel_count,
#         means,
#         stddevs,
#         random_init
#     )

#     gauss_kernels = np.reshape(
#       gauss_kernels, (
#         filter_size,
#         input_size,
#         kernel_size
#         )
#       ).T

#     gauss_kernels = tf.convert_to_tensor(
#         value=gauss_kernels,
#         dtype=dtype)

#     return gauss_kernels


def G(x, alpha):
    """ Return Gaussian line shape at x with HWHM alpha """
    return np.sqrt(np.log(2) / np.pi) / alpha\
                             * np.exp(-(x / alpha)**2 * np.log(2))

def L(x, gamma):
    """ Return Lorentzian line shape at x with HWHM gamma """
    return gamma / np.pi / (x**2 + gamma**2)

def V(x, alpha, gamma):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.

    """
    sigma = alpha / np.sqrt(2 * np.log(2))
    v = np.real(wofz((x + 1j*gamma)/sigma/np.sqrt(2))) / sigma\
                                                           /np.sqrt(2*np.pi)
    return v

def gen_voigt_kernel(shape, weights, dtype=dtypes.float32, random_init=False):

    alphas = tf.reshape(weights[0, :, :], [-1])
    gammas = tf.reshape(weights[1, :, :], [-1])

    kernel_size = shape[0]
    input_size = shape[1]
    filter_size = shape[2]

    voigt_kernel_count =  input_size * filter_size

    truncate = kernel_size/2
    width = int(truncate + 0.5)
    voigt_kernels = tf.TensorArray(dtype, size=0, dynamic_size=True)

    for i in tf.range(voigt_kernel_count):

        shift_voigt = np.random.randint(-kernel_size/2, kernel_size/2)
        support = tf.convert_to_tensor(
        value=np.arange(
            -width - shift_voigt,
            width + 1 - shift_voigt
            ),

        dtype=dtype
        )
        center = int(len(support)/2)
        left_cut = center - int(kernel_size/2)
        right_cut = center + int(kernel_size/2)

        alpha = alphas[i]
        gamma = gammas[i]

        voigt_kernel = V(support, alpha, gamma)
        #tf.math.exp(-((support - mean) ** 2)/(2*stddev ** 2))

        if (kernel_size % 2) != 0:
            voigt_kernel = voigt_kernel[left_cut + 1:right_cut + 2]
        else:
            voigt_kernel = voigt_kernel[left_cut + 1:right_cut + 1]
        voigt_kernels = voigt_kernels.write(voigt_kernels.size(), voigt_kernel)

    voigt_kernels = voigt_kernels.stack()
    voigt_kernels = tf.transpose(
      tf.reshape(
        voigt_kernels,
        (
          filter_size,
          input_size,
          kernel_size
        )
      )
    )

    voigt_kernels = tf.convert_to_tensor(
        value=voigt_kernels,
        dtype=dtype)
    return voigt_kernels
