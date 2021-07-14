from odft_tools.kernels import (
    ContinuousConvKernel1DV1,
    GaussianKernel1DV2,
    Kernel1DV2,
)

from odft_tools.init_ops_v2 import VarianceScaling
from odft_tools.utils import (
    gen_gaussian_kernel_v1_1D,
    generate_kernel,
    lorentz_dist,
    cauchy_dist,
    gaussian_dist
)

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn

import tensorflow as tf
from tensorflow import keras

import numpy as np
import functools
import six


class CustomExpandLayer(tf.keras.layers.Layer):   # Inheritance class
    # Define output
    def call(self, inputdata):
        output = tf.expand_dims(inputdata, axis=-1)
        return output


class CustomReduceLayer(tf.keras.layers.Layer):   # Inheritance class
    # Define output
    def call(self, inputdata):
        output = tf.reduce_sum(inputdata, axis=-1)
        return output


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


class ContinuousConv1D(keras.layers.Conv1D):
    def __init__(self,
                 weights_init,
                 create_continuous_kernel,
                 random_init=False,
                 seed=None,
                 **kwargs):
        super().__init__(**kwargs)

        # Set costum kernel init. with gaussian kernel
        self.kernel_initializer = ContinuousConvKernel1DV1(
            weights_init=weights_init,
            create_continuous_kernel=create_continuous_kernel,
            random_init=random_init,
            seed=seed
        )


# For the generic (several distributions choice) the
# Version 2 we can choose in the init func. different
# distributions.
# Here we had to set the parameters from the dist. as
# the weights
class Continuous1DConvV2(keras.layers.Conv1D):
    """
    Arguments:
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
    """

    def __init__(self,
                 weights_init,
                 random_init=False,
                 seed=None,
                 costum_kernel_type='gaussian',
                 **kwargs):
        super().__init__(**kwargs)

        self.weights_init = weights_init
        self.random_init = random_init
        self.seed = seed
        self.kernel_shape = None
        self.costum_kernel_type = costum_kernel_type
        self.count = 0

        if costum_kernel_type not in {'gaussian',
                            'lorentz', 'voigt', 'cauchy'}:
            raise ValueError("Invalid `distribution` argument:", costum_kernel_type)

        self.gen_costum_kernel = generate_kernel

        if costum_kernel_type == 'gaussian':
            self.kernel_dist = gaussian_dist
        elif costum_kernel_type == 'lorentz':
            self.kernel_dist = lorentz_dist
        elif costum_kernel_type == 'cauchy':
            self.kernel_dist = cauchy_dist

        self.costum_weights_initializer = Kernel1DV2(
            weights_init=self.weights_init,
            random_init=self.random_init,
            seed=self.seed
        )

        self.costum_weights_shape = None
        self.costum_weights_regularizer = None
        self.costum_weights_constraint = None

    def build(self, input_shape, **kwargs):
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

        # The construiction of weights shape has changed
        self.costum_weights_shape = (len(self.weights_init), ) + (input_channel // self.groups,
                                           self.filters)

        # means and stddevs are here the weights to be trained
        weghts_type = self.costum_kernel_type + '_weights'
        self.weights_costum = self.add_weight(
            name=weghts_type,
            shape=self.costum_weights_shape,
            initializer=self.costum_weights_initializer,
            regularizer=self.costum_weights_regularizer,
            constraint=self.costum_weights_constraint,
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
        if tf_op_name == self.costum_kernel_type + 'Conv1D':
            tf_op_name = self.costum_kernel_type + 'conv1d'  # Backwards compat.

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

        # bevore every conv. calc new gaussian kerne for conv. from weights
        # or paramters of the dist.
        # import time

        # start = time.time()
        costum_weights_kernel = self.gen_costum_kernel(
            shape=self.kernel_shape,
            weights=self.weights_costum,
            kernel_dist=self.kernel_dist
        )

        outputs = self._convolution_op(inputs, costum_weights_kernel)

        self.count += 1
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
        # end = time.time()
        # print(f'time elapsed {end - start}')
        # assert False
        if self.activation is not None:
            return self.activation(outputs)
        return outputs


# For the generic (several distributions choice) the
# Version 2 we can choose in the init func. different
# distributions.
# Here we had to set the parameters from the dist. as
# the weights
# class Continuous1DConvV2(keras.layers.convolutional.Conv1D):
#     """
#     Arguments:
#         filters: Integer, the dimensionality of the output space
#             (i.e. the number of output filters in the convolution).
#     """

#     def __init__(self,
#                  weights_init,
#                  random_init=False,
#                  seed=None,
#                  costum_kernel_type='gaussian',
#                  **kwargs):
#         super().__init__(**kwargs)

#         self.weights_init = weights_init
#         self.random_init = random_init
#         self.seed = seed
#         self.kernel_shape = None
#         self.costum_kernel_type = costum_kernel_type
#         self.count = 0

#         if costum_kernel_type not in {'gaussian',
#                             'lorentz', 'voigt', 'cauchy'}:
#             raise ValueError("Invalid `distribution` argument:", costum_kernel_type)

#         self.gen_costum_kernel = generate_kernel

#         if costum_kernel_type == 'gaussian':
#             self.kernel_dist = gaussian_dist
#         elif costum_kernel_type == 'lorentz':
#             self.kernel_dist = lorentz_dist
#         elif costum_kernel_type == 'cauchy':
#             self.kernel_dist = cauchy_dist

#         self.costum_weights_initializer = Kernel1DV2(
#             weights_init=self.weights_init,
#             random_init=self.random_init,
#             seed=self.seed
#         )

#         self.costum_weights_shape = None
#         self.costum_weights_regularizer = None
#         self.costum_weights_constraint = None

#     def build(self, input_shape, **kwargs):
#         input_shape = tensor_shape.TensorShape(input_shape)
#         input_channel = self._get_input_channel(input_shape)
#         if input_channel % self.groups != 0:
#             raise ValueError(
#                 'The number of input channels must be evenly divisible by'
#                 'thenumber of groups. Received groups={}, but the input has {}'
#                 'channels (full input shape is {}).'.format(self.groups,
#                                                             input_channel,
#                                                             input_shape))
#         self.kernel_shape = self.kernel_size + (input_channel // self.groups,
#                                            self.filters)

#         # The construiction of weights shape has changed
#         self.costum_weights_shape = (len(self.weights_init), ) + (input_channel // self.groups,
#                                            self.filters)

#         # means and stddevs are here the weights to be trained
#         weghts_type = self.costum_kernel_type + '_weights'
#         self.weights_costum = self.add_weight(
#             name=weghts_type,
#             shape=self.costum_weights_shape,
#             initializer=self.costum_weights_initializer,
#             regularizer=self.costum_weights_regularizer,
#             constraint=self.costum_weights_constraint,
#             trainable=True,
#             dtype=self.dtype)

#         if self.use_bias:
#             self.bias = self.add_weight(
#                 name='bias',
#                 shape=(self.filters,),
#                 initializer=self.bias_initializer,
#                 regularizer=self.bias_regularizer,
#                 constraint=self.bias_constraint,
#                 trainable=True,
#                 dtype=self.dtype)
#         else:
#             self.bias = None
#         channel_axis = self._get_channel_axis()
#         self.input_spec = InputSpec(min_ndim=self.rank + 2,
#                                     axes={channel_axis: input_channel})

#         # Convert Keras formats to TF native formats.
#         if self.padding == 'causal':
#             tf_padding = 'VALID'  # Causal padding handled in `call`.
#         elif isinstance(self.padding, six.string_types):
#             tf_padding = self.padding.upper()
#         else:
#             tf_padding = self.padding
#         tf_dilations = list(self.dilation_rate)
#         tf_strides = list(self.strides)


#         tf_op_name = self.__class__.__name__
#         if tf_op_name == self.costum_kernel_type + 'Conv1D':
#             tf_op_name = self.costum_kernel_type + 'conv1d'  # Backwards compat.

#         self._convolution_op = functools.partial(
#             nn_ops.convolution_v2,
#             strides=tf_strides,
#             padding=tf_padding,
#             dilations=tf_dilations,
#             data_format=self._tf_data_format,
#             name=tf_op_name)
#         self.built = True

#     def call(self, inputs):
#         if self._is_causal:  # Apply causal padding to inputs for Conv1D.
#             inputs = array_ops.pad(inputs,
#                                    self._compute_causal_padding(inputs))

#         # bevore every conv. calc new gaussian kerne for conv. from weights
#         # or paramters of the dist.

#         costum_weights_kernel = self.gen_costum_kernel(
#             shape=self.kernel_shape,
#             weights=self.weights_costum,
#             kernel_dist=self.kernel_dist
#         )
#         outputs = self._convolution_op(inputs, costum_weights_kernel)
#         self.count += 1
#         if self.use_bias:
#             output_rank = outputs.shape.rank
#         if self.rank == 1 and self._channels_first:
#             # nn.bias_add does not accept a 1D input tensor.
#             bias = array_ops.reshape(self.bias, (1, self.filters, 1))
#             outputs += bias
#         else:
#             # Handle multiple batch dimensions.
#             if output_rank is not None and output_rank > 2 + self.rank:

#                 def _apply_fn(o):
#                     return nn.bias_add(o, self.bias,
#                                        data_format=self._tf_data_format)

#                 outputs = nn_ops.squeeze_batch_dims(
#                     outputs,
#                     _apply_fn,
#                     inner_rank=self.rank + 1)
#             else:
#                 outputs = nn.bias_add(
#                     outputs, self.bias, data_format=self._tf_data_format)

#         if self.activation is not None:
#             return self.activation(outputs)
#         return outputs
