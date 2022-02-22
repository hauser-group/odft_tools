import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models

from tensorflow.python.ops.init_ops_v2 import Initializer
from tensorflow.python.framework import dtypes
from tensorflow.python.ops.init_ops_v2 import _assert_float_dtype


class ContinuousConvKernel1DV1(Initializer):
    '''
     weights_init: 2 dimensianl array. First entry is mean, secound is std
     create_continuous_kernel: method that creates the values for a continuous
                               kernel
     random_init: Bool if the initialization is random
     seed: integer if radom values should be the same.
    '''
    def __init__(self,
                 weights_init,
                 create_continuous_kernel=None,
                 random_init=False,
                 seed=None):

        if not create_continuous_kernel:
            raise ValueError("Set a continuous kernel")

        # We overwritte the init method. We are checkking if weights_init
        # consists of two variables (mean and std)
        if len(weights_init) != 2:
            raise ValueError("weights_init length must be 2")
        # Checking if mean is greater zero
        if weights_init[0] < 0:
            raise ValueError("'mean' must be positive float")
        # Checking if std is greater zero
        if weights_init[1] < 0:
            raise ValueError("'stddev' must be positive float")

        self.weights_init = weights_init
        self.random_init = random_init
        self.create_continuous_kernel = create_continuous_kernel

    def __call__(self, shape, dtype=dtypes.float32):
        # We are overwritting the call method and setting the gaussian kernel
        """Returns a tensor object initialized as specified by the initializer.
        Args:
          shape: Shape of the tensor.
          dtype: Optional dtype of the tensor. Only floating point types are
              supported.
        Raises:
          ValueError: If the dtype is not floating point
        """

        dtype = _assert_float_dtype(dtype)

        continuous_kernel = self.create_continuous_kernel(
            shape=shape,
            weights=self.weights_init,
            dtype=dtype,
            random_init=self.random_init)
        # The custom kernel is returned
        return continuous_kernel

    def get_config(self):
        return {
            "mean": self.weights_init[0],
            "stddev": self.weights_init[1],
            "raondom_init": self.random_init
        }


class ContinuousConv1D(keras.layers.Conv1D):
    '''
    Here, we just overwritte of the kernel_initializer in the _init_ method.
    The method for creating the custom kernel is passed to the __init__ as a
    parameter
    and can be found in utils/utils.py
    '''
    def __init__(self,
                 weights_init,
                 create_continuous_kernel,
                 random_init=False,
                 seed=None,
                 **kwargs):
        super().__init__(**kwargs)

        # Set custom kernel init. with gaussian kernel
        self.kernel_initializer = ContinuousConvKernel1DV1(
            weights_init=weights_init,
            create_continuous_kernel=create_continuous_kernel,
            random_init=random_init,
            seed=seed
        )


class IntegrateLayer(tf.keras.layers.Layer):
    '''
    In the call method we implement the trapezoidal integral and in the init
    we pass h for inte
    '''
    def __init__(self, h=1.0, **kwargs):
        super().__init__(**kwargs)
        self.h = h

    def call(self, inputs):
        return self.h * tf.reduce_sum(
            (inputs[:, :-1] + inputs[:, 1:]) / 2.,
            axis=1, name='trapezoidal_integral_approx')

    def get_config(self):
        config = super().get_config()
        config.update({'h': self.h})
        return config


class CustomCNNV1Model(keras.Model):
    '''
    filter_size: integer. How many filters should be created in a layer.
    kernel_sice: integer. How many kernel functions are in one filter
    layer_length: integer. Layer length
    create_continuous_kernel: method that creates the values for a continuous
                              kernel
    kernel_regularizer: float. kernel regulizer
    dx: float. integration step
    weights: 2 dimensianl array. First entry is mean, secound is std
    '''
    def __init__(
            self,
            filter_size,
            kernel_size,
            layer_length,
            create_continuous_kernel,
            kernel_regularizer,
            activation,
            dx=0.002,
            weights=[5, 5]):
        super().__init__()
        self.dx = dx
        self.conv_layers = []
        mean = weights[0]
        stddev = weights[1]

        # Here we create continuous kernels in a foor loop
        for l in range(layer_length):
            cont_layer = ContinuousConv1D(
                filters=filter_size,
                kernel_size=kernel_size,
                padding='same',
                weights_init=[mean, stddev],
                create_continuous_kernel=create_continuous_kernel,
                kernel_regularizer=kernel_regularizer,
                activation=activation,
                random_init=True,
            )
            self.conv_layers.append(cont_layer)

        # last layer is fixed to use a single filter
        cont_layer = ContinuousConv1D(
            filters=1,
            kernel_size=kernel_size,
            padding='same',
            weights_init=[mean, stddev],
            create_continuous_kernel=create_continuous_kernel,
            kernel_regularizer=kernel_regularizer,
            activation=activation,
            random_init=True
        )
        self.conv_layers.append(cont_layer)
        self.integrate = IntegrateLayer(dx)

    @tf.function
    def call(self, inputs):
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            # Calculate kinetic energy density tau by applying convolutional
            # layers
            tau = inputs
            for layer in self.conv_layers:
                tau = layer(tau)
            # Kinetic energy T is integral over kinetiv energy density
            T = self.integrate(tau)
        # The discretized derivative needs to be divided by dx
        dT_dn = tape.gradient(T, inputs)/self.dx
        return {'T': T, 'dT_dn': dT_dn}
