from tensorflow.python.framework import dtypes
import tensorflow as tf
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np


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

    gaus_kernel_count = input_size * filter_size

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


def load_data_cnn(path, data_name):

    with h5py.File(path + data_name, 'r') as f:
        keys = f.keys()
        # build a dict (dataset.value has been deprecated. Use dataset[()]
        # instead.)
        data = {key: f[key][()] for key in keys}

    x = np.linspace(0, 1, 500)
    dx = x[1] - x[0]
    N = 1
    # density is wavefunction squared
    n = np.sum(data['wavefunctions'][:, :, :N]**2, axis=-1)
    # integrate using trapezoidal rule:
    V = np.sum(0.5 * (data['potential'][:, :-1]*n[:, :-1]
               + data['potential'][:, 1:]*n[:, 1:]) * dx, axis=-1)
    # kinetic energy is total energy minus potential energy
    kinetic = np.sum(data['energies'][:, :N], axis=-1) - V
    # kinetic energy derivative
    kinetic_derivative = np.expand_dims(
        np.sum(data['energies'][:, :N], axis=-1)
        / N, axis=-1) - data['potential']
    density = n.reshape((-1, 500, 1))

    return kinetic, kinetic_derivative, density


def plot_gaussian_weights(weights, before_after):
    plt.ylabel('Probabilty Density')
    plt.xlabel('Kernel Length')
    plt.title(f'Kernel Values First Layer {before_after}')
    plt.plot(weights[:, 0, :])
    plt.show()
    plt.close()
    
    
def plot_derivative_energy(x, dT_dn, model, n):
    dT_dn_model = tf.squeeze(model(n[0].reshape((1, 500, 1)).astype(np.float32))['dT_dn'])
    plt.plot(x, dT_dn[0], 'o', color="black", markersize=0.3, label='reference')
    plt.plot(x, dT_dn_model, color="orange", label='predicted')
    plt.ylabel(r'$\frac{\delta T}{\delta n}$')
    plt.xlabel(r'$n$')
    plt.title('Comparison reference with trained energy derivative')
    plt.legend(['reference', 'model'])
    plt.show()
    plt.close()