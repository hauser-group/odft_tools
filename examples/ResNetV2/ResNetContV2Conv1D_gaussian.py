import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pathlib import Path
import os

# from https://github.com/hauser-group/odft_tools
from odft_tools.layers import (
    IntegrateLayer,
    Continuous1DConvV2
)

from odft_tools.models_resnet_ccn import (
    ResNetContConv1DV2Model
)

from odft_tools.utils_train import (
    load_data,
    plot_derivative_energy,
    plot_gaussian_weights_v2
)

from odft_tools.utils import (
    lorentz_dist,
    cauchy_dist,
    gaussian_dist
)

from odft_tools.keras_utils import (
    WarmupExponentialDecay
)

from tensorflow.python.framework import dtypes
import random

from odft_tools.utils import (
    generate_kernel,
    lorentz_dist,
    gaussian_dist
)

import os

os.environ['TF_GPU_THREAD_MODE']='gpu_private'
os.environ['TF_GPU_THREAD_COUNT']='1'

data_path = 'datasets/orbital_free_DFT/'
logs='performance_log'

kinetic_train, kinetic_derivativ_train, density_train = load_data(
    path=data_path,
    data_name='M=100_training_data.hdf5'
)

kinetic_test, kinetic_derivativ_test, density_test = load_data(
    path=data_path,
    data_name='test_data.hdf5'
)


density = {'n': density_train.astype(np.float32)}
targetdata = {
    'T': kinetic_train.astype(np.float32),
    'dT_dn': kinetic_derivativ_train.astype(np.float32)
}


kernel_size = 100
amp = 0.5
mean = 5
stddev = 10

res_net_blocks_count = 3
epoch = 10

path = 'results/ResNetV2/'

density = {'n': density_train}
targetdata = {'T': kinetic_train, 'dT_dn': kinetic_derivativ_train}

# training_dataset = tf.data.Dataset.from_tensor_slices((n.astype(np.float32), {'T': T.astype(np.float32), 'dT_dn': dT_dn.astype(np.float32)})).batch(100).repeat(10)
training_dataset = tf.data.Dataset.from_tensor_slices(
    (
        density_train.astype(np.float32),
        {'T': kinetic_train.astype(np.float32),
        'dT_dn': kinetic_derivativ_train.astype(np.float32)}
    )
).batch(100) #.repeat(100)

initial_learning_rate = WarmupExponentialDecay(
    initial_learning_rate=0.1,
    final_learning_rate=0.0000000001,
    warmup_steps=0,
    decay_steps=2000,
    decay_rate=0.9,
    cold_factor=1.0,
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
    weights_gaus=[amp, mean, stddev],
    n_outputs=None,
    random_init=True,
    dx=0.002,
    distribution=distribution,
    kernel_regularizer=tf.keras.regularizers.l2(0.00001)
)

callback = tf.keras.callbacks.EarlyStopping(
    monitor='dT_dn_loss',
    patience=10000,
    restore_best_weights=True,
)

model.create_res_net_model()
model.build(input_shape=(1, 500))
model.compile(
    optimizer=tf.keras.optimizers.Adadelta(  # Adam
        learning_rate=1.0, # initial_learning_rate,
        rho=0.09
    ),
    loss={'T': 'mse', 'dT_dn': 'mse'},
    loss_weights={'T': 0.2, 'dT_dn': 1.0}, # As recommended by Manuel: scale the loss in T by 0.2
    metrics={'T': ['mae'], 'dT_dn': ['mae']})

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=path + 'cp.ckpt',
    save_freq=10000
)

tboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir = logs,
    histogram_freq = 1,
    profile_batch=2
)

model.models.summary()

weights_before_train = model.layers[0].get_weights()[0]
import time

start = time.time()
with tf.device('/device:GPU:0'):
    model.fit(
        training_dataset,
        epochs=epoch,
        verbose=2,
        validation_data=(
            density_test, {'T': kinetic_test, 'dT_dn': kinetic_derivativ_test}
        ),
        validation_freq=10000,
        callbacks=[callback])
end = time.time()
print(f'time elapsed {end - start}')

weights_after_train = model.layers[0].get_weights()[0]

if distribution == 'lorentz':
    kernel_dist = lorentz_dist
if distribution == 'cauchy':
    kernel_dist = cauchy_dist
if distribution == 'gaussian':
    kernel_dist = gaussian_dist

plot_gaussian_weights_v2(weights_before_train, kernel_size, ' before', path, kernel_dist)
plot_gaussian_weights_v2(weights_after_train, kernel_size, ' after', path, kernel_dist)

x = np.linspace(0, 1, 500)
plot_derivative_energy(x, kinetic_derivativ_test, model, density_test, path)

