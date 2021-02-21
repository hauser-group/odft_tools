#!/usr/bin/env python
# coding: utf-8

# In[2]:


import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pathlib import Path
import os

# from https://github.com/hauser-group/odft_tools
from odft_tools.layers import (
    IntegrateLayer,
    Continuous1DConvV1
)

from odft_tools.models_resnet_ccn import (
    ResNetContConv1DModel,
    ResNetContConv1DV2Model,
    ResNetConv1DModel
)

from odft_tools.utils import (
    plot_derivative_energy,
    plot_gaussian_weights_v1,
    plot_gaussian_weights_v2
)

from odft_tools.keras_utils import (
    WarmupExponentialDecay
)

from tensorflow.python.framework import dtypes
import random

data_path = '../datasets/orbital_free_DFT/'


# In[ ]:


with h5py.File(data_path + 'dataset_large.hdf5', 'r') as f:
    keys = f.keys()
    print(keys)
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
n = n.reshape((-1, 500))


# ### Test Set

# In[ ]:


with h5py.File(data_path + 'dataset_validate.hdf5', 'r') as f:
    keys = f.keys()
    print(keys)
    # build a dict (dataset.value has been deprecated. Use dataset[()] instead.)
    data_test = {key:f[key][()] for key in keys}


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
n_test = n_test.reshape((-1, 500))


# In[ ]:


kernel_size = 100
mean = 5
stddev = 10

res_net_blocks_count = 4
epoch = 100

density = {'n': n.astype(np.float32)}
targetdata = {'T': T.astype(np.float32), 'dT_dn': dT_dn.astype(np.float32)}

# training_dataset = tf.data.Dataset.from_tensor_slices((n.astype(np.float32), {'T': T.astype(np.float32), 'dT_dn': dT_dn.astype(np.float32)})).batch(100).repeat(10)

initial_learning_rate = 0.001
decay_steps = 2000
decay_rate= 0.9

initial_learning_rate = WarmupExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=False,
    name=None
)

seed = 0
tf.random.set_seed(seed)

distribution = 'gaussian'
# distribution = 'lorentz'

model = ResNetContConv1DV2Model(
    filter_size=32,
    kernel_size=100,
    layer_size=None,
    num_res_net_blocks=res_net_blocks_count,
    weights_gaus=[mean, stddev],
    n_outputs=None,
    random_init=True,
    dx=0.002,
    distribution=distribution
)

optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate, amsgrad=False)

model.create_res_net_model()
model.build(input_shape=(1, 500))
model.compile(optimizer=optimizer,
              loss={'T': 'mse', 'dT_dn': 'mse'},
              loss_weights={'T': 0.2, 'dT_dn': 1.0}, # As recommended by Manuel: scale the loss in T by 0.2
              metrics={'T': ['mae'], 'dT_dn': ['mae']})
print('--------------------------------->Start<---------------------------------')
print(f'res_net {res_net_blocks_count} with {epoch} epochs')
path = '/ResNetConv2D/' + distribution + '/'

model.models.summary()
weights_before_train = model.layers[0].get_weights()[0]
model.fit(x=density, y=targetdata, epochs=epoch, verbose=2, validation_data=(n_test, {'T': T_test, 'dT_dn': dT_dn_test}), validation_freq=10)
weights_after_train = model.layers[0].get_weights()[0]
print('--------------------------------->END<---------------------------------')


# In[ ]:


plot_gaussian_weights_v2(weights_before_train, mean, stddev, kernel_size, ' before', path)
plot_gaussian_weights_v2(weights_after_train, mean, stddev, kernel_size, ' after', path)
plot_derivative_energy(x, dT_dn, model, n, path)


# In[ ]:


import pandas as pd
df = pd.DataFrame([])
df['loss'] = model.history.history['loss']
df['dT_dn_loss'] = model.history.history['dT_dn_loss']
df['T_loss'] = model.history.history['T_loss']

df.to_csv('results/' + path + 'losses.csv')

plt.figure(figsize=(20, 3))

plt.plot(df['loss'][1:])
plt.xlabel('epochs')
plt.ylabel('loss a.u.')
plt.title('loss over epochs for ResNet CCNN')
plt.savefig('loss_ResNet_CNN.png')
plt.show()

