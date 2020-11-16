from odft_tools.utils import (
    gen_gaussian_kernel_v1_1D,
    gen_gaussian_kernel_v2_1D
)

from tensorflow.python.ops.init_ops_v2 import (
    Initializer,
    _RandomGenerator
)

from tensorflow.python.framework import dtypes
from tensorflow.python.ops.init_ops_v2 import _assert_float_dtype
from tensorflow.python.ops.init_ops import _compute_fans

import tensorflow as tf
import numpy as np
import math


class GaussianKernel1DV1(Initializer):
    def __init__(self,
                 weights_init,
                 stddev=1.0):

        if len(weights_init) != 2:
            raise ValueError("weights_init length must be 2")

        if weights_init[0] < 0:
            raise ValueError("'mean' must be positive float")

        if weights_init[1] < 0:
            raise ValueError("'stddev' must be positive float")

        self.weights_init = weights_init

    def __call__(self, shape, dtype=dtypes.float32):
        """Returns a tensor object initialized as specified by the initializer.
        Args:
          shape: Shape of the tensor.
          dtype: Optional dtype of the tensor. Only floating point types are
              supported.
        Raises:
          ValueError: If the dtype is not floating point
        """

        dtype = _assert_float_dtype(dtype)
        gauss_kernel = gen_gaussian_kernel_v1_1D(
            shape=shape,
            weights=self.weights_init,
            dtype=dtype)
        return gauss_kernel

    def get_config(self):
        return {
            "mean": self.weights_init[0],
            "stddev": self.weights_init[1]
        }


class GaussianKernel1DV2(Initializer):
    """docstring for GaussianKernel1DWeights"""
    def __init__(self,
                 weights_init,
                 seed=None,
                 scale=1.0,
                 mode="fan",
                 distribution="truncated_normal"):

        if len(weights_init) != 2:
            raise ValueError("weights_init length must be 2")

        if weights_init[0] < 0:
            raise ValueError("'mean' must be positive float")

        if weights_init[1] < 0:
            raise ValueError("'stddev' must be positive float")

        if scale <= 0.:
            raise ValueError("`scale` must be positive float.")

        if mode not in {"fan", "fan_out", "fan_avg"}:
            raise ValueError("Invalid `mode` argument:", mode)

        distribution = distribution.lower()
        # Compatibility with keras-team/keras.
        if distribution == "normal":
            distribution = "truncated_normal"

        if distribution not in {"uniform", "truncated_normal",
                                "untruncated_normal"}:
            raise ValueError("Invalid `distribution` argument:", distribution)

        self.seed = seed
        self._random_generator = _RandomGenerator(seed)
        self.weights_init = weights_init
        self.mode = mode
        self.distribution = 'untruncated_normal'#distribution
        self.mode = mode
        self.scale = scale

    def __call__(self, shape, dtype=dtypes.float32):
        """Returns a tensor object initialized as specified by the initializer.
        Args:
          shape: Shape of the tensor.
          dtype: Optional dtype of the tensor. Only floating point types are
          supported.
        Raises:
          ValueError: If the dtype is not floating point
        """

        mean = self.weights_init[0]
        stddev = self.weights_init[1]

        dtype = _assert_float_dtype(dtype)

        partition_info = None  # Keeps logic so can be readded later if necessary
        dtype = _assert_float_dtype(dtype)
        scale = self.scale
        scale_shape = shape

        if partition_info is not None:
            scale_shape = partition_info.full_shape

        fan, fan_out = _compute_fans(scale_shape)
        if self.mode == "fan":
            scale /= max(1., fan)
        elif self.mode == "fan_out":
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan + fan_out) / 2.)
        
        shape_weight = (int(shape[0]/2), shape[1], shape[2])

        limit = math.sqrt(3.0 * scale)
        if stddev is None:
            stddev = self._random_generator.random_uniform(shape_weight, 0, limit, dtype)
        else:
            stddev = tf.convert_to_tensor(
                value=[[stddev] * shape_weight[1]] * shape_weight[2], 
                dtype=dtype  
            )
            stddev = stddev + tf.random.normal(shape=tf.shape(stddev), mean=9, stddev=4)
        if mean is None:
            mean = self._random_generator.random_uniform(shape_weight, 0, limit, dtype)
        else:
            mean = tf.convert_to_tensor(
                value=[[mean] * shape_weight[1]] * shape_weight[2], 
                dtype=dtype
            )
            mean = mean + tf.random.normal(shape=tf.shape(mean), mean=5, stddev=4)
        weights = tf.concat([[mean, stddev]], 0)
        weights = tf.reshape(weights, shape, name=None)
        return weights

    def get_config(self):
        return {
            "mean": self.weights_init[0],
            "stddev": self.weights_init[1],
            "scale": self.scale,
            "mode": self.mode,
            "distribution": self.distribution,
            "seed": self.seed
        }


class LinearKernel():
    """Only used for comparison to standard ridge regression"""
    
    def __call__(self, X, Y, dy=False):
        n = X.shape[0]
        m = Y.shape[0]
        n_dim = X.shape[1]
        
        K = np.einsum('ik,jk->ij', X, Y)
        if not dy:
            return K
        return np.concatenate([K, np.repeat(X[:, np.newaxis, :], m, axis=1).reshape(n, m*n_dim)], axis=1)

    
class RBFKernel():

    def __init__(self, length_scale=1.0, scale=1.0, constant=0.0):
        self.length_scale = length_scale
        self.scale = scale
        self.constant = constant
        
    def __call__(self, X, Y, dx=False, dy=False, h=1.0):
        n = X.shape[0]
        m = Y.shape[0]
        n_dim = X.shape[1]
        
        K = np.zeros((n*(1 + int(dx)*n_dim), m*(1 + int(dy)*n_dim)))
        lh = self.length_scale*h
        lh2 = lh**2
        
        # Doing this in a loop to avoid massive memory overhead
        for i in range(n):
            for j in range(m):
                # Index ranges for the derivatives are given by the following
                # slice objects:
                di = slice(n + i*n_dim, n + (i + 1)*n_dim, 1)
                dj = slice(m + j*n_dim, m + (j + 1)*n_dim, 1)
                scaled_diff = (X[i, :] - Y[j, :])/self.length_scale
                K[i, j] = self.scale * np.exp(-.5*np.dot(scaled_diff, scaled_diff))
                if dy:
                    K[i, dj] = K[i, j]*scaled_diff/lh
                    if dx:
                        K[di, j] = -K[i, dj]
                        K[di, dj] = K[i, j]/lh2 * (
                            np.eye(n_dim) - np.outer(scaled_diff, scaled_diff))
                else:
                    if dx:
                        K[di, j] = -K[i, j]*scaled_diff/lh
        K[:n, :m] += self.constant
        return K