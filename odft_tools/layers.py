import tensorflow as tf
import numpy as np
from odft_tools.kernels import GaussianKernel1D
from odft_tools.init_ops_v2 import VarianceScaling
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec
import six
import functools
from tensorflow.python.ops import nn_ops


class IntegrateLayer(tf.keras.layers.Layer):
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


class ContinuousConv1D(tf.python.keras.layers.convolutional.Conv1D):
    """
    Arguments:
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
    """

    def __init__(self,
                 mean=0,
                 stddev=1,
                 **kwargs):
        super().__init__(**kwargs)

        self.mean = mean
        self.stddev = stddev
        # self.kernel_initializer = VarianceScaling()
        self.kernel_initializer = GaussianKernel1D(
            mean=self.mean,
            stddev=stddev
        )

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        if input_channel % self.groups != 0:
            raise ValueError(
                'The number of input channels must be evenly divisible by'
                'thenumber of groups. Received groups={}, but the input has {}'
                'channels (full input shape is {}).'.format(self.groups,
                                                            input_channel,
                                                            input_shape))

        kernel_shape = self.kernel_size + (input_channel // self.groups,
                                           self.filters)

        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        channel_axis = self._get_channel_axis()
        self.input_spec = InputSpec(min_ndim=self.rank + 2,
                                    axes={channel_axis: input_channel})

        # Convert Keras formats to TF native formats.
        if self.padding == 'causal':
            tf_padding = 'VALID'  # Causal padding handled in `call`.
        elif isinstance(self.padding, six.string_types):
            tf_padding = self.padding.upper()
        else:
            tf_padding = self.padding
        tf_dilations = list(self.dilation_rate)
        tf_strides = list(self.strides)

        tf_op_name = self.__class__.__name__
        if tf_op_name == 'Conv1D':
            tf_op_name = 'conv1d'  # Backwards compat.

        self._convolution_op = functools.partial(
            nn_ops.convolution_v2,
            strides=tf_strides,
            padding=tf_padding,
            dilations=tf_dilations,
            data_format=self._tf_data_format,
            name=tf_op_name)
        self.built = True