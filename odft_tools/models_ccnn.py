import numpy as np
import keras
import tensorflow as tf

from scipy.linalg import cho_solve, cholesky
from odft_tools.kernels import RBFKernel
from odft_tools.utils import (first_derivative_matrix,
                              second_derivative_matrix,
                              integrate)
from odft_tools.layers import (
    Continuous1DConvV1,
    IntegrateLayer,
    Continuous1DConv
)

from tensorflow.python.framework import dtypes
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.eager import monitoring


class ClassicCNN(keras.Model):
    def __init__(
            self,
            layers=[32,],
            kernel_size=64,
            dx=0.002):

        super(ClassicCNN, self).__init__()
        self.dx = dx
        self.conv_layers = []
        for l in layers:
            self.conv_layers.append(tf.keras.layers.Conv1D(l, kernel_size, padding='same', activation='exponential'))
        # last layer is fixed to use a single filter
        self.conv_layers.append(tf.keras.layers.Conv1D(1, kernel_size, padding='same', activation='linear'))
        self.integrate = IntegrateLayer(dx)

    @tf.function
    def call(self, inputs):
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            # Calculate kinetic energy density tau by applying convolutional layers
            tau = inputs
            for layer in self.conv_layers:
                tau = layer(tau)
            # Kinetic energy T is integral over kinetiv energy density
            T = self.integrate(tau)
        # The discretized derivative needs to be divided by dx
        dT_dn = tape.gradient(T, inputs)/self.dx
        return {'T': T, 'dT_dn': dT_dn}

    @tf.function
    def call(self, inputs):
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            # Calculate kinetic energy density tau by applying convolutional layers
            tau = inputs
            for layer in self.conv_layers:
                tau = layer(tau)
            # Kinetic energy T is integral over kinetiv energy density
            T = self.integrate(tau)
        # The discretized derivative needs to be divided by dx
        dT_dn = tape.gradient(T, inputs)/self.dx
        return {'T': T, 'dT_dn': dT_dn}


class ContCNNV1(ClassicCNN):
    def __init__(
            self,
            layers=[32,],
            kernel_size=100,
            dx=0.002,
            weights=[5, 5]):
        super().__init__()
        self.dx = dx
        self.conv_layers = []
        mean = weights[0]
        stddev = weights[1]

        for l in layers:
            if l == 0 and 2:
                cont_layer = Continuous1DConvV1(
                    filters=32,
                    kernel_size=kernel_size,
                    activation='softplus',
                    padding='same',
                    weights_init=[mean, stddev],
                    random_init=True
                )
                self.conv_layers.append(cont_layer)
            else:
                cont_layer = tf.keras.layers.Conv1D(
                    filters=32,
                    kernel_size=kernel_size,
                    activation='softplus',
                    padding='same',
                    name='Conv1D_act_' + str(l)
                )
                self.conv_layers.append(cont_layer)
            # self.conv_layers.append(cont_layer)
        # last layer is fixed to use a single filter
        cont_layer = Continuous1DConvV1(
            filters=1,
            kernel_size=kernel_size,
            activation='linear',
            padding='same',
            weights_init=[mean, stddev],
            random_init=True
        )
        self.conv_layers.append(cont_layer)
        self.integrate = IntegrateLayer(dx)


class ContCNNModel(ClassicCNN):
    def __init__(
            self,
            layers=[32,],
            kernel_size=64,
            dx=0.002,
            weights=[5, 5, 1],
            distribution='gaussian'):
        super().__init__()
        self.dx = dx
        self.conv_layers = []
        mean = weights[0]
        stddev = weights[1]

        for l in layers:
            cont_layer = Continuous1DConv(
                   filters=32,
                   kernel_size=kernel_size,
                   activation='softplus',
                   padding='same',
                   weights_init=[mean, stddev],
                   random_init=True,
                   costum_kernel_type=distribution
            )
            self.conv_layers.append(cont_layer)
            # self.conv_layers.append(cont_layer)
        # last layer is fixed to use a single filter
        cont_layer = Continuous1DConv(
            filters=1,
            kernel_size=kernel_size,
            activation='linear',
            padding='same',
            weights_init=[mean, stddev],
            random_init=True,
            costum_kernel_type=distribution
        )
        self.conv_layers.append(cont_layer)
        self.integrate = IntegrateLayer(dx)
