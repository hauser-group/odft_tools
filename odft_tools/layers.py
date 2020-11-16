from odft_tools.kernels import (
    GaussianKernel1DV1,
    GaussianKernel1DV2
)
from odft_tools.init_ops_v2 import VarianceScaling
from odft_tools.utils import (
    gen_gaussian_kernel_v1_1D,
    gen_gaussian_kernel_v2_1D
)

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn

import tensorflow as tf
import numpy as np
import functools
import six


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


class Continuous1DConvV2(tf.python.keras.layers.convolutional.Conv1D):
    """
    Arguments:
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
    """

    def __init__(self,
                 weights_init=None,
                 **kwargs):
        super().__init__(**kwargs)

        self.weights_init = weights_init
        self.kernel_shape = None

        self.gaussian_weights_initializer = GaussianKernel1DV2(
            weights_init=self.weights_init)

        self.gaussian_weights_shape = None
        self.gaussian_weights_regularizer = None
        self.gaussian_weights_constraint = None


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
        self.kernel_shape = self.kernel_size + (input_channel // self.groups,
                                           self.filters)

        self.gaussian_weights_shape = (2, ) + (input_channel // self.groups,
                                           self.filters)

        self.weights_gausian = self.add_weight(
            name='gaussian_weights',
            shape=self.gaussian_weights_shape,
            initializer=self.gaussian_weights_initializer,
            regularizer=self.gaussian_weights_regularizer,
            constraint=self.gaussian_weights_constraint,
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

    def call(self, inputs):
        if self._is_causal:  # Apply causal padding to inputs for Conv1D.
            inputs = array_ops.pad(inputs,
                                   self._compute_causal_padding(inputs))

        gaussian_weights_kernel = gen_gaussian_kernel_v2_1D(
            shape=self.kernel_shape,
            weights=self.weights_gausian
        )
        outputs = self._convolution_op(inputs, gaussian_weights_kernel)
        
        if self.use_bias:
            output_rank = outputs.shape.rank
        if self.rank == 1 and self._channels_first:
            # nn.bias_add does not accept a 1D input tensor.
            bias = array_ops.reshape(self.bias, (1, self.filters, 1))
            outputs += bias
        else:
            # Handle multiple batch dimensions.
            if output_rank is not None and output_rank > 2 + self.rank:

                def _apply_fn(o):
                    return nn.bias_add(o, self.bias,
                                       data_format=self._tf_data_format)

                outputs = nn_ops.squeeze_batch_dims(
                    outputs,
                    _apply_fn,
                    inner_rank=self.rank + 1)
            else:
                outputs = nn.bias_add(
                    outputs, self.bias, data_format=self._tf_data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class Continuous1DConvV1(tf.python.keras.layers.convolutional.Conv1D):
    """
    Arguments:
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
    """

    def __init__(self,
                 weights_init,
                 stddev=1,
                 **kwargs):
        super().__init__(**kwargs)

        self.weights_init = weights_init
        self.kernel_initializer = GaussianKernel1DV1(weights_init=weights_init)

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