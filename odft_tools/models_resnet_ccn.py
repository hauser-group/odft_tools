import numpy as np
import keras
import tensorflow as tf

from odft_tools.utils import (first_derivative_matrix,
                              second_derivative_matrix,
                              integrate)
from odft_tools.layers import (
    IntegrateLayer,
    ContinuousConv1D,
    Continuous1DConvV2
)

from tensorflow.python.framework import dtypes
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.eager import monitoring


# ResNet Basic Model (Manuels version - simplified)
class ResNetConv1DModel(keras.Model):
    def __init__(
            self,
            filter_size,
            kernel_size,
            layer_size,
            num_res_net_blocks,
            n_outputs,
            kernel_regularizer,
            dx=0.002,
            seed=None):

        super(ResNetConv1DModel, self).__init__()
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.layer_size = layer_size
        self.num_res_net_blocks = num_res_net_blocks
        self.n_outputs = n_outputs
        self.dx = tf.constant(dx, dtype=dtypes.float32)
        self.seed = seed
        self.kernel_regularizer = kernel_regularizer

    def create_res_net_model(self):
        # Here we create the Model-
        # The kind of layers is set here
        density = tf.keras.layers.Input(shape=(500,), name='density')
        value = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(density)

        for l in range(self.num_res_net_blocks):
            inputs = value
            value = tf.keras.layers.Conv1D(
                filters=self.filter_size,
                kernel_size=self.kernel_size,
                activation='softplus',
                padding='same',
                # kernel_regularizer=self.kernel_regularizer
            )(value)

            # res_net layer for '+ x'
            value = tf.keras.layers.Conv1D(
                filters=self.filter_size,
                kernel_size=self.kernel_size,
                activation=None,
                padding='same',
                # kernel_regularizer=self.kernel_regularizer
            )(value)

            value = tf.keras.layers.Add()([value, inputs])
            value = tf.keras.layers.Activation('softplus', name='ahct' + str(l))(value)
        # last layer is fixed to use a single filter
        value = tf.keras.layers.Conv1D(
                filters=1,
                kernel_size=self.kernel_size,
                activation=None,
                padding='same',
                # kernel_regularizer=self.kernel_regularizer
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


class ResNetCostumLayer1DModel(ResNetConv1DModel):
    def __init__(
            self,
            weights_gaus,
            create_continuous_kernel,
            activation,
            random_init=True,
            **kwargs):

        super().__init__(**kwargs)

        self.weights_gaus = weights_gaus
        self.random_init = random_init
        self.create_continuous_kernel = create_continuous_kernel
        self.activation = activation

    def create_res_net_model(self):
        density = tf.keras.layers.Input(shape=(500,), name='density')
        value = tf.keras.layers.Lambda(
            lambda x: tf.expand_dims(x, axis=-1)
        )(density)

        for l in range(self.num_res_net_blocks):
            inputs = value

            value = ContinuousConv1D(
                filters=self.filter_size,
                kernel_size=self.kernel_size,
                activation=self.activation,
                padding='same',
                weights_init=self.weights_gaus,
                random_init=self.random_init,
                create_continuous_kernel=self.create_continuous_kernel,
                kernel_regularizer=self.kernel_regularizer
            )(value)

            # res_net layer for '+ x'
            value = ContinuousConv1D(
                filters=self.filter_size,
                kernel_size=self.kernel_size,
                activation=None,
                padding='same',
                weights_init=self.weights_gaus,
                random_init=self.random_init,
                create_continuous_kernel=self.create_continuous_kernel,
                kernel_regularizer=self.kernel_regularizer
            )(value)

            value = tf.keras.layers.Add()([value, inputs])
            value = tf.keras.layers.Activation('softplus',)(value)

        # last layer is fixed to use a single filter
        value = ContinuousConv1D(
            filters=1,
            kernel_size=self.kernel_size,
            activation=None,
            padding='same',
            weights_init=self.weights_gaus,
            random_init=self.random_init,
            create_continuous_kernel=self.create_continuous_kernel,
            kernel_regularizer=self.kernel_regularizer
        )(value)

        dT_dn = tf.keras.layers.Lambda(
            lambda x: tf.reduce_sum(x, axis=-1),
            name='dT_dn'
        )(value)

        T = IntegrateLayer(self.dx)(dT_dn)

        self.models = keras.Model(
            inputs={'n': density},
            outputs={'T': T, 'dT_dn': dT_dn}
        )

# In this Model we set the first layer as
# Continuous CNN Version 2 --> the parameters of the
# distribution are the weights
# All other layers are ordinary 1 dimensional cnn's
class ResNetContConv1DV2Model(ResNetConv1DModel):
    def __init__(
            self,
            weights_gaus,
            distribution,
            random_init=True,
            **kwargs):
        super().__init__(**kwargs)

        self.weights_gaus = weights_gaus
        self.random_init = random_init
        self.distribution = distribution

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        import time
        start = time.time()
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        end = time.time()

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        print(f'elapsed time {end - start}')
        assert False
        return {m.name: m.result() for m in self.metrics}

    def create_res_net_model(self):
        density = tf.keras.layers.Input(shape=(500,), name='density')
        value = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(density)

        for l in range(self.num_res_net_blocks):
            inputs = value

            value = Continuous1DConvV2(
                filters=self.filter_size,
                kernel_size=self.kernel_size,
                activation='softplus',
                padding='same',
                weights_init=self.weights_gaus,
                random_init=self.random_init,
                costum_kernel_type=self.distribution
            )(value)

            value = Continuous1DConvV2(
                filters=self.filter_size,
                kernel_size=self.kernel_size,
                activation=None,
                padding='same',
                weights_init=self.weights_gaus,
                random_init=self.random_init,
                costum_kernel_type=self.distribution
            )(value)

            value = tf.keras.layers.Add()([value, inputs])
            value = tf.keras.layers.Activation('softplus')(value)

        # last layer is fixed to use a single filter
        value = Continuous1DConvV2(
            filters=1,
            kernel_size=self.kernel_size,
            activation=None,
            padding='same',
            weights_init=self.weights_gaus,
            random_init=self.random_init,
            costum_kernel_type=self.distribution
            # kernel_regularizer=self.kernel_regularizer
        )(value)

        dT_dn = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1), name='dT_dn')(value)
        T = IntegrateLayer(self.dx)(dT_dn)

        self.models = keras.Model(inputs={'n': density}, outputs={'T': T, 'dT_dn': dT_dn})
