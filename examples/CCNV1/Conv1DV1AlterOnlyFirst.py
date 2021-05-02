#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# from https://github.com/hauser-group/odft_tools
from odft_tools.models_ccnn import (
    ClassicCNN
)

from odft_tools.layers import (
    IntegrateLayer,
    Continuous1DConvV1
)

from odft_tools.utils import (
    plot_gaussian_weights_v1,
    plot_derivative_energy
)

from odft_tools.keras_utils import (
    WarmupExponentialDecay
)

data_path = '../datasets/orbital_free_DFT/'

seed = 0
tf.random.set_seed(seed)


# In[ ]:


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

        for l in range(len(layers)):
            if l == 0:
                cont_layer = Continuous1DConvV1(
                    filters=32,
                    kernel_size=kernel_size,
                    activation='softplus',
                    padding='same',
                    weights_init=[mean, stddev],
                    random_init=True
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


# # Load dataset
# Both data .hdf5-Files can be downloaded from https://github.com/hauser-group/datasets/tree/master/orbital_free_DFT

# ### Training

# In[ ]:


# with h5py.File(data_path + 'M=100_training_data.hdf5', 'r') as f:
#     keys = f.keys()
#     print(keys)
#     # build a dict (dataset.value has been deprecated. Use dataset[()] instead.)
#     data = {key:f[key][()] for key in keys}
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


# ### Test Set

# In[ ]:


with h5py.File(data_path + 'dataset_validate.hdf5', 'r') as f:
    keys = f.keys()
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
n_test = n_test.reshape((-1, 500, 1))


# # Define model

# In[ ]:


kernel_size = 100
# Feel free to use larger kernel size (Manuel used 100) and larger networks (Manuels ResNet used layers=[32, 32, 32, 32, 32, 32]).
model = ContCNNV1(layers=[32, 32, 32, 32, 32, 32], kernel_size=kernel_size, dx=dx)
# Tell the model what input to expect. The first dimension (None) represents the batch size and remains undefinded.
model.build(input_shape=(None, 500, 1))

initial_learning_rate = 0.001
decay_steps = 400
decay_rate= 0.9

initial_learning_rate = WarmupExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=False,
    name=None
)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=False),
              loss={'T': 'mse', 'dT_dn': 'mse'},
              loss_weights={'T': 0.2, 'dT_dn': 1.0}, # As recommended by Manuel: scale the loss in T by 0.2
                metrics={'T': ['mae'], 'dT_dn': ['mae']})

callback = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=500,
    restore_best_weights=True,
)

model.summary()


# In[ ]:


# Build a dataset that repeats the data (cast to float32) 10 times to reduce output in model.fit().
# Note that this step is not necessary, you could simply feed the numpy arrays into the model.fit() method.
training_dataset = tf.data.Dataset.from_tensor_slices((n.astype(np.float32), {'T': T.astype(np.float32), 'dT_dn': dT_dn.astype(np.float32)})).batch(100).repeat(10)

path = 'results/ContCNNV1FirstOnly/'

def to_weights(model, before_after):
    for lay in range(len(model.layers) - 1):
        weights_layer = pd.DataFrame(model.layers[lay].get_weights()[0][:, 0, :])
        if not os.path.exists(path + 'weigths/'):
            os.makedirs(path + 'weigths/')
        weights_layer.to_csv(path + 'weigths/' + 'weights'+ before_after + '_layer' + str(lay) + '.csv')

to_weights(model, 'before')
# Beware when comparing the results to our paper. The output here is in Hartree!
weights_before_train = model.layers[0].get_weights()[0]
model.fit(training_dataset, epochs=3000, verbose=2, validation_data=(n_test, {'T': T_test, 'dT_dn': dT_dn_test}), validation_freq=10, callbacks=[callback]) #
weights_after_train = model.layers[0].get_weights()[0]
to_weights(model, 'after')

def plot_gaussian_weights_v1(weights, path, before_after):
    if not os.path.exists(path):
        os.makedirs(path)
    plt.ylabel('density')
    plt.xlabel('kernel size')
    plt.title('First Layer of CCNN '+ before_after +' train')
    plt.plot(weights[:, 0, :])
    plt.savefig(path + 'layer_ccnV1_'+ before_after +'.png')
    plt.show()
    plt.close()

import os
def plot_derivative_energy(x, dT_dn, model, n, path):
    if not os.path.exists(path):
        os.makedirs(path)

    plt.plot(x, dT_dn[0])
    plt.plot(x, tf.squeeze(model(n[0].reshape((1, 500, 1)).astype(np.float32))['dT_dn']))
    plt.ylabel('dT_dn')
    plt.title('Comparison reference with trained energy derivative')
    plt.savefig(path + 'ccnnV1_energy_derivatice.png')
    plt.show()
    plt.close()

plot_gaussian_weights_v1(weights_before_train, path, 'before')
plot_gaussian_weights_v1(weights_after_train, path, 'after')

plot_derivative_energy(x, dT_dn, model, n, path)


# In[ ]:


import pandas as pd
df = pd.DataFrame([])
df['loss'] = model.history.history['loss']
df['dT_dn_loss'] = model.history.history['dT_dn_loss']
df['T_loss'] = model.history.history['T_loss']
df.to_csv(path + 'losses.csv')

plt.figure(figsize=(20, 3))

plt.plot(df['loss'])
plt.xlabel('epochs')
plt.ylabel('loss a.u.')
plt.title('loss over epochs for CCNN')
plt.savefig(path + 'loss_CNN.png')
plt.show()
plt.close()

