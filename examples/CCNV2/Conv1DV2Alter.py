#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
# from https://github.com/hauser-group/odft_tools
from odft_tools.models_ccnn import (
    ClassicCNN,
    ContCNNModel
)

from odft_tools.layers import (
    IntegrateLayer,
    Continuous1DConv
)

from odft_tools.utils import (
    plot_derivative_energy,
    plot_gaussian_weights_v1,
    plot_gaussian_weights_v2
)

from odft_tools.utils import (
    gen_gaussian_kernel_v1_1D,
    generate_kernel,
    lorentz_dist,
    cauchy_dist,
    gaussian_dist
)

from odft_tools.kernels import (
    GaussianKernel1DV1,
    GaussianKernel1DV2,
    LorentzKernel1D,
    Kernel1DV2
)

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn

import numpy as np
import functools
import six

data_path = '../datasets/orbital_free_DFT/'


# # Load dataset
# Both data .hdf5-Files can be downloaded from https://github.com/hauser-group/datasets/tree/master/orbital_free_DFT

# ### Training

# In[ ]:


with h5py.File(data_path + 'dataset_large.hdf5', 'r') as f:
    keys = f.keys()
    # build a dict (dataset.value has been deprecated. Use dataset[()] instead.)
    data = {key:f[key][()] for key in keys}


# In[ ]:


x = np.linspace(0, 1, 500)
dx = x[1] - x[0]
N = 1
# density is wavefunction squared
n = np.sum(data['wavefunctions'][:, :, :N]**2, axis=-1)
# integrate using trapezoidal rule:
V = np.sum(0.5*(data['potential'][:, :-1]*n[:, :-1]
                + data['potential'][:, 1:]*n[:, 1:])
           * dx, axis=-1)
# kinetic energy is total energy minus potential energy
T = np.sum(data['energies'][:, :N], axis=-1) - V
# kinetic energy derivative
dT_dn = np.expand_dims(np.sum(data['energies'][:, :N], axis=-1)/N, axis=-1) - data['potential']
n = n.reshape((-1, 500, 1))

with h5py.File(data_path + 'dataset_validate.hdf5', 'r') as f:
    keys = f.keys()
    # build a dict (dataset.value has been deprecated. Use dataset[()] instead.)
    data_test = {key:f[key][()] for key in keys}

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
            if l == 0 and l == 2 and l == 4:
                cont_layer = Continuous1DConv(
                    filters=32,
                    kernel_size=kernel_size,
                    activation='softplus',
                    padding='same',
                    weights_init=[mean, stddev],
                    random_init=True,
                    costum_kernel_type=distribution
                )
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


# In[ ]:


# density is wavefunction squared
n_test = np.sum(data_test['wavefunctions'][:, :, :N]**2, axis=-1)
# integrate using trapezoidal rule:
V_test = np.sum(0.5*(data_test['potential'][:, :-1]*n_test[:, :-1]
                + data_test['potential'][:, 1:]*n_test[:, 1:])
                * dx, axis=-1)
# kinetic energy is total energy minus potential energy
T_test = np.sum(data_test['energies'][:, :N], axis=-1) - V_test
# kinetic energy derivative
dT_dn_test = - data_test['potential'] + np.expand_dims(np.sum(data_test['energies'][:, :N], axis=-1)/N, axis=-1)
n_test = n_test.reshape((-1, 500, 1))


# # Define model

# In[ ]:


kernel_size = 100
mean = 5
stddev = 20
# Feel free to use larger kernel size (Manuel used 100) and larger networks (Manuels ResNet used layers=[32, 32, 32, 32, 32, 32]).
distribution = 'gaussian' # cauchy, lorentz
model = ContCNNModel(layers=[32, 32, 32, 32, 32, 32], kernel_size=kernel_size, dx=dx, weights=[mean, stddev, 1], distribution=distribution)
# Tell the model what input to expect. The first dimension (None) represents the batch size and remains undefinded.
model.build(input_shape=(None, 500, 1))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss={'T': 'mse', 'dT_dn': 'mse'},
              loss_weights={'T': 0.2, 'dT_dn': 1.0}, # As recommended by Manuel: scale the loss in T by 0.2
              metrics={'T': ['mae'], 'dT_dn': ['mae']})

callback = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=50,
    restore_best_weights=True,
)

model.summary()


# In[ ]:


# Build a dataset that repeats the data (cast to float32) 10 times to reduce output in model.fit().
# Note that this step is not necessary, you could simply feed the numpy arrays into the model.fit() method.
training_dataset = tf.data.Dataset.from_tensor_slices((n.astype(np.float32), {'T': T.astype(np.float32), 'dT_dn': dT_dn.astype(np.float32)})).batch(100).repeat(10)

path = '/results/ContCNNV2Alter/'


# In[ ]:


# Beware when comparing the results to our paper. The output here is in Hartree!
weights_before_train = model.layers[0].get_weights()[0]
model.fit(training_dataset, epochs=2000, verbose=2, validation_data=(n_test, {'T': T_test, 'dT_dn': dT_dn_test}), validation_freq=10, callbacks=[callback])
weights_after_train = model.layers[0].get_weights()[0]


# ## Inspect results
# The prediction by the CNN exhibits rapid oscillations, which we hope to eliminate by going from a convolution with a discrete kernel towards a convolution with a continuous function

# In[ ]:


plot_gaussian_weights_v2(weights_before_train, mean, stddev, kernel_size, ' before', path)
plot_gaussian_weights_v2(weights_after_train, mean, stddev, kernel_size, ' after', path)
plot_derivative_energy(x, dT_dn, model, n, path)

import pandas as pd
df = pd.DataFrame([])
df['loss'] = model.history.history['loss']
df['dT_dn_loss'] = model.history.history['dT_dn_loss']
df['T_loss'] = model.history.history['T_loss']

df.to_csv(path + 'losses.csv')

plt.figure(figsize=(20, 3))

plt.plot(df['loss'][1:])
plt.xlabel('epochs')
plt.ylabel('loss a.u.')
plt.title('loss over epochs for ResNet CCNN')
plt.savefig('loss_ResNet_CNN.png')
plt.show()

