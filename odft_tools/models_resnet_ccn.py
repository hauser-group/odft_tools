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
    Continuous1DConvV2,
    IntegrateLayer
)

from tensorflow.python.framework import dtypes
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.eager import monitoring


class ResNetConv1DModel(keras.Model):
    def __init__(
            self,
            filter_size,
            kernel_size,
            layer_size,
            num_res_nat_blocks,
            n_outputs,
            dx=1.0,
            seed=None):

        super(ResNetConv1DModel, self).__init__()
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.layer_size = layer_size
        self.num_res_nat_blocks = num_res_nat_blocks
        self.n_outputs = n_outputs
        self.dx = tf.constant(dx, dtype=dtypes.float32)
        self.seed = seed

    def create_res_net_model(self):
        density = tf.keras.layers.Input(shape=(500,), name='density')
        value = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(density)
        
        for l in range(self.num_res_nat_blocks):
            inputs = value
            value = tf.keras.layers.Conv1D(
                filters=self.filter_size,
                kernel_size=self.kernel_size,
                activation='softplus',
                padding='same',
                name='Conv1D_act_' + str(l),
            )(value)

            # res_net layer for '+ x'
            value = tf.keras.layers.Conv1D(
                filters=self.filter_size,
                kernel_size=self.kernel_size,
                activation=None,
                padding='same',
                name='Conv1D_noact_' + str(l),
            )(value)

            value = tf.keras.layers.Add()([value, inputs])
            value = tf.keras.layers.Activation('softplus', name='act' + str(l))(value)
        # last layer is fixed to use a single filter
        value = tf.keras.layers.Conv1D(
                filters=1,
                kernel_size=self.kernel_size,
                activation=None,
                padding='same',
                name='Conv1D_end_' + str(l),
            )(value)

        dT_dn = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1), name='dT_dn')(value)
        T = IntegrateLayer(self.dx)(dT_dn)

        self.models = keras.Model(inputs={'n': density}, outputs={'T': T, 'dT_dn': dT_dn})

    def call(self, inputs):
        inputs = tf.nest.flatten(inputs)

        with tf.GradientTape() as tape:
            tape.watch(inputs)
            predictions = self.models(inputs)
        # The discretized derivative needs to be divided by dx
        T = predictions['T']
        dT_dn = tape.gradient(T, inputs)/self.dx
        return {'T': T, 'dT_dn': dT_dn}


class ResNetContConv1DModel(ResNetConv1DModel):
    def __init__(
            self,
            weights_gaus,
            random_init=True,
            **kwargs):
        # super(ResNetContConv1DModel, self).__init__()
        super().__init__(**kwargs)

        self.weights_gaus = weights_gaus
        self.random_init = random_init

    def create_res_net_model(self):
        density = tf.keras.layers.Input(shape=(500,), name='density')
        value = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(density)

        # inputs = value

        for l in range(self.num_res_nat_blocks):
            inputs = value

            if l == 0:
                value = Continuous1DConvV1(
                    filters=self.filter_size,
                    kernel_size=self.kernel_size,
                    activation='softplus',
                    padding='same',
                    weights_init=self.weights_gaus,
                    random_init=self.random_init
                )(value)
            else:
                value = tf.keras.layers.Conv1D(
                    filters=self.filter_size,
                    kernel_size=self.kernel_size,
                    activation='softplus',
                    padding='same',
                    name='Conv1D_act_' + str(l)
                )(value)

            # res_net layer for '+ x'
            value = tf.keras.layers.Conv1D(
                filters=self.filter_size,
                kernel_size=self.kernel_size,
                activation=None,
                padding='same',
                name='Conv1D_noact_' + str(l)
            )(value)

            value = tf.keras.layers.Add()([value, inputs])
            value = tf.keras.layers.Activation('softplus', name='act' + str(l))(value)
        
        # last layer is fixed to use a single filter
        value = tf.keras.layers.Conv1D(
            filters=1,
            kernel_size=self.kernel_size,
            activation=None,
            padding='same',
            name='Conv1D_end_' + str(l),
        )(value)

        dT_dn = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1), name='dT_dn')(value)
        T = IntegrateLayer(self.dx)(dT_dn)

        self.models = keras.Model(inputs={'n': density}, outputs={'T': T, 'dT_dn': dT_dn})


class ResNetContConv1DV2Model(ResNetConv1DModel):
    def __init__(
            self,
            weights_gaus,
            random_init=True,
            **kwargs):
        # super(ResNetContConv1DModel, self).__init__()
        super().__init__(**kwargs)

        self.weights_gaus = weights_gaus
        self.random_init = random_init

    def create_res_net_model(self):
        density = tf.keras.layers.Input(shape=(500,), name='density')
        value = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(density)

        # inputs = value

        for l in range(self.num_res_nat_blocks):
            inputs = value

            if l == 0:
                value = Continuous1DConvV2(
                    filters=self.filter_size,
                    kernel_size=self.kernel_size,
                    activation='softplus',
                    padding='same',
                    weights_init=self.weights_gaus,
                    random_init=self.random_init
                )(value)
            else:
                value = tf.keras.layers.Conv1D(
                    filters=self.filter_size,
                    kernel_size=self.kernel_size,
                    activation='softplus',
                    padding='same',
                    name='Conv1D_act_' + str(l)
                )(value)

            # res_net layer for '+ x'
            value = tf.keras.layers.Conv1D(
                filters=self.filter_size,
                kernel_size=self.kernel_size,
                activation=None,
                padding='same',
                name='Conv1D_noact_' + str(l)
            )(value)

            value = tf.keras.layers.Add()([value, inputs])
            value = tf.keras.layers.Activation('softplus', name='act' + str(l))(value)
        
        # last layer is fixed to use a single filter
        value = tf.keras.layers.Conv1D(
            filters=1,
            kernel_size=self.kernel_size,
            activation=None,
            padding='same',
            name='Conv1D_end_' + str(l),
        )(value)

        dT_dn = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1), name='dT_dn')(value)
        T = IntegrateLayer(self.dx)(dT_dn)

        self.models = keras.Model(inputs={'n': density}, outputs={'T': T, 'dT_dn': dT_dn})