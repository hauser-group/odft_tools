import numpy as numpy
import unittest
from odft_tools.layers import ContinuousConv1D


class LayersTest(unittest.TestCase):
    def test_gaussian_kernel1D(self):
        sigma = 1
        truncate = 0

        continuous_cnn = ContinuousConv1D(
            filters=None,
            kernels=None,
            mu=None,
            sigma=sigma
        )

        gauss_kernel = continuous_cnn.gaussian_kernel1D(
            sigma=sigma,
            truncate=truncate
        )

        print(gauss_kernel)
        self.assertFalse(False)
