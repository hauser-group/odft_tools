import tensorflow as tf


class IntegrateLayer(tf.keras.layers.Layer):
    def __init__(self, h=1.0, **kwargs):
        super().__init__(**kwargs)
        self.h = h

    def call(self, inputs):
        return self.h*tf.reduce_sum(
            (inputs[:, :-1] + inputs[:, 1:])/2.,
            axis=1, name='trapezoidal_integral_approx')

    def get_config(self):
        config = super().get_config()
        config.update({'h': self.h})
        return config


class ContinuousConv1D(tf.keras.layers.Layer):
    """
    Arguments:
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
    """

    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        # Add weights etc

    def call(self, inputs):
        # Actual implementation of the convolution goes here
        pass
