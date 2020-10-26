from odft_tools.utils import gen_gaussian_kernel1D

from tensorflow.python.ops.init_ops_v2 import (
    Initializer,
    _RandomGenerator
)

from tensorflow.python.framework import dtypes
from tensorflow.python.ops.init_ops_v2 import _assert_float_dtype

import tensorflow as tf
import numpy as np


class GaussianKernel1D(Initializer):
    def __init__(self,
                 mean=0.0,
                 stddev=1.0):

        if stddev < 0:
            raise ValueError("'stddev' must be positive float")

        if mean < 0:
            raise ValueError("'mean' must be positive float")

        self.mean = mean
        self.stddev = stddev

    def __call__(self, shape, dtype=dtypes.float32):
        """Returns a tensor object initialized as specified by the initializer.
        Args:
          shape: Shape of the tensor.
          dtype: Optional dtype of the tensor. Only floating point types are
              supported.
        Raises:
          ValueError: If the dtype is not floating point
        """

        stddev = self.stddev
        mean = self.mean

        dtype = _assert_float_dtype(dtype)

        gauss_kernel = gen_gaussian_kernel1D(
            shape=shape,
            mean=mean,
            stddev=stddev,
            dtype=dtype)

        return gauss_kernel

    def get_config(self):
        return {
            "mean": self.mean,
            "stddev": self.stddev
        }

