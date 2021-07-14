#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# from https://github.com/hauser-group/odft_tools
from odft_tools.layers import (
    IntegrateLayer,
    Continuous1DConvV1,
    Continuous1DConvV2
)

data_path = '../datasets/orbital_free_DFT/'


# # Load dataset
# Both data .hdf5-Files can be downloaded from https://github.com/hauser-group/datasets/tree/master/orbital_free_DFT

# ### Training

# In[ ]:


with h5py.File(data_path + 'M=100_training_data.hdf5', 'r') as f:
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
n = n.reshape((-1, 500, 1))


# In[ ]:


dx


# ### Test Set

# In[ ]:


with h5py.File(data_path + 'test_data.hdf5', 'r') as f:
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
n_test = n_test.reshape((-1, 500, 1))


# # Define model

# In[ ]:


class MyModel(tf.keras.Model):

    def __init__(self, layers=[32,], kernel_size=64, dx=1.0):
        super(MyModel, self).__init__()
        self.dx = dx
        self.conv_layers = []
        for l in layers:
            self.conv_layers.append(tf.keras.layers.Conv1D(l, kernel_size, padding='same', activation='exponential'))
        # last layer is fixed to use a single filter
        self.conv_layers.append(tf.keras.layers.Conv1D(1, kernel_size, padding='same', activation='linear'))
        self.integrate = IntegrateLayer(dx)

    @tf.function
    def call(self, inputs):
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            # Calculate kinetic energy density tau by applying convolutional layers
            tau = inputs
            for layer in self.conv_layers:
                tau = layer(tau)
            # Kinetic energy T is integral over kinetiv energy density
            T = self.integrate(tau)
        # The discretized derivative needs to be divided by dx
        dT_dn = tape.gradient(T, inputs)/self.dx
        return {'T': T, 'dT_dn': dT_dn}


# In[ ]:


mean = 5
stddev = 10

class MyModelCont(tf.keras.Model):
    def __init__(self, layers=[32,], kernel_size=64, dx=1.0):
        super(MyModelCont, self).__init__()
        self.dx = dx
        self.conv_layers = []
        
        for l in layers:
            cont_layer = Continuous1DConvV1(
                   filters=32,
                   kernel_size=kernel_size,
                   activation='softplus',
                   padding='same',
                   weights_init=[mean, stddev],
                   random_init=True
            )
            self.conv_layers.append(cont_layer)
            # self.conv_layers.append(cont_layer)
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

    @tf.function
    def call(self, inputs):
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            # Calculate kinetic energy density tau by applying convolutional layers
            tau = inputs
            for layer in self.conv_layers:
                tau = layer(tau)
            # Kinetic energy T is integral over kinetiv energy density
            T = self.integrate(tau)
        # The discretized derivative needs to be divided by dx
        dT_dn = tape.gradient(T, inputs)/self.dx
        return {'T': T, 'dT_dn': dT_dn}


# In[ ]:


kernel_size = 100
# Feel free to use larger kernel size (Manuel used 100) and larger networks (Manuels ResNet used layers=[32, 32, 32, 32, 32, 32]).
model = MyModel(layers=[32, ], kernel_size=kernel_size, dx=dx)
# Tell the model what input to expect. The first dimension (None) represents the batch size and remains undefinded.
model.build(input_shape=(None, 500, 1))


# In[ ]:


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
              loss={'T': 'mse', 'dT_dn': 'mse'}, 
              loss_weights={'T': 0.2, 'dT_dn': 1.0}, # As recommended by Manuel: scale the loss in T by 0.2
              metrics={'T': ['mae'], 'dT_dn': ['mae']})
model.summary()


# In[ ]:


# Build a dataset that repeats the data (cast to float32) 10 times to reduce output in model.fit().
# Note that this step is not necessary, you could simply feed the numpy arrays into the model.fit() method.
training_dataset = tf.data.Dataset.from_tensor_slices((n.astype(np.float32), {'T': T.astype(np.float32), 'dT_dn': dT_dn.astype(np.float32)})).batch(100).repeat(10)

result_type = '/ContCNNV1/32_1/'


# In[ ]:


weigths = model.layers[0].get_weights()[0]

truncate = kernel_size/2

width = int(truncate + 0.5)
# width = int(truncate * stddev + 0.5)
support = np.arange(-width, width + 1)
center = int(len(support)/2)

left_cut = center - int(kernel_size/2)
right_cut = center + int(kernel_size/2)

for mean, stddev in zip(weigths[0][0], weigths[1][0]):
    gauss_kernel = np.exp(-((support - mean) ** 2)/(2*stddev ** 2))
    gauss_kernel = gauss_kernel / gauss_kernel.sum()

    if (kernel_size % 2) != 0:
        gauss_kernel = gauss_kernel[left_cut + 1:right_cut + 2]
    else:
        gauss_kernel = gauss_kernel[left_cut + 1:right_cut + 1]
    plt.plot(gauss_kernel)


# In[ ]:


weigths = model.layers[0].get_weights()[0]
plt.ylabel('density')
plt.xlabel('kernel size')
plt.title('Gaussian Kernel of ContConv1V1 with Layer softplus act. fun')
plt.plot(weigths[:, 0, :])
# plt.savefig('results' + result_type + 'gauss_kernel_softplus_act_V1.png')
plt.show()

weigths = model.layers[1].get_weights()[0]
plt.ylabel('density')
plt.xlabel('kernel size')
plt.title('Gaussian Kernel of ContConv1V1 with Layer linear act. fun')
plt.plot(weigths[:, 0, :])
# plt.savefig('results' + result_type + 'gauss_kernel_linear_act_V1.png')
plt.show()


# In[ ]:


# Beware when comparing the results to our paper. The output here is in Hartree!
model.fit(training_dataset, epochs=50, verbose=2, validation_data=(n_test, {'T': T_test, 'dT_dn': dT_dn_test}), validation_freq=10)


# ## Inspect results
# The prediction by the CNN exhibits rapid oscillations, which we hope to eliminate by going from a convolution with a discrete kernel towards a convolution with a continuous function

# In[ ]:


plt.plot(x, dT_dn[0])
plt.plot(x, tf.squeeze(model(n[0].reshape((1, 500, 1)).astype(np.float32))['dT_dn']))
plt.ylabel('dT_dn')
plt.title('Comparison reference with trained')
plt.savefig('results' + result_type + 'dT_dn_V1.png')
plt.show()


# In[ ]:


weigths = model.layers[0].get_weights()[0]
plt.ylabel('density')
plt.xlabel('kernel size')
plt.title('Gaussian Kernel of ContConv1V1 with Layer softplus act. fun after')
plt.plot(weigths[:, 0, :])
plt.savefig('results' + result_type + 'gauss_kernel_softplus_act_V1_after.png')
plt.show()

weigths = model.layers[1].get_weights()[0]
plt.ylabel('density')
plt.xlabel('kernel size')
plt.title('Gaussian Kernel of ContConv1V1 with Layer linear act. fun after')
plt.plot(weigths[:, 0, :])
plt.savefig('results' + result_type + 'gauss_kernel_linear_act_V1_after.png')
plt.show()


# In[ ]:


weigths = model.layers[0].get_weights()[0]

truncate = kernel_size/2

width = int(truncate + 0.5)
# width = int(truncate * stddev + 0.5)
support = np.arange(-width, width + 1)
center = int(len(support)/2)

left_cut = center - int(kernel_size/2)
right_cut = center + int(kernel_size/2)

for mean, stddev in zip(weigths[0][0], weigths[1][0]):
    gauss_kernel = np.exp(-((support - mean) ** 2)/(2*stddev ** 2))
    gauss_kernel = gauss_kernel / gauss_kernel.sum()

    if (kernel_size % 2) != 0:
        gauss_kernel = gauss_kernel[left_cut + 1:right_cut + 2]
    else:
        gauss_kernel = gauss_kernel[left_cut + 1:right_cut + 1]
    plt.plot(gauss_kernel)

