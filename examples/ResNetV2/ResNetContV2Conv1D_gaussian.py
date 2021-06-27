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

from odft_tools.keras_utils import (
    WarmupExponentialDecay
)

from tensorflow.python.framework import dtypes
import random

from odft_tools.utils import (
    generate_kernel,
    lorentz_dist
)



# input_shape = tensor_shape.TensorShape(input_shape)
# input_channel = self._get_input_channel(input_shape)

# kernel_shape = (None, 500, 1)

# generate_kernel(
#         shape=self.kernel_shape,
#         weights=[5, 10],
#         kernel_dist=lorentz_dist
# )



data_path = '../datasets/orbital_free_DFT/'


kinetic_train, kinetic_derivativ_train, density_train = load_data(
    path=data_path,
    data_name='dataset_large.hdf5'
)

kinetic_test, kinetic_derivativ_test, density_test = load_data(
    path=data_path,
    data_name='dataset_validate.hdf5'
)

density = {'n': density_train.astype(np.float32)}
targetdata = {
    'T': kinetic_train.astype(np.float32),
    'dT_dn': kinetic_derivativ_train.astype(np.float32)
}


kernel_size = 100
mean = 5
stddev = 10

res_net_blocks_count = 3
epoch = 3

path = 'results/'

density = {'n': density_train.astype(np.float32)}
targetdata = {'T': kinetic_train.astype(np.float32), 'dT_dn': kinetic_derivativ_train.astype(np.float32)}

# training_dataset = tf.data.Dataset.from_tensor_slices((n.astype(np.float32), {'T': T.astype(np.float32), 'dT_dn': dT_dn.astype(np.float32)})).batch(100).repeat(10)

initial_learning_rate = WarmupExponentialDecay(
    initial_learning_rate=0.0001,
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

distribution = 'cauchy'
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

callback = tf.keras.callbacks.EarlyStopping(
    monitor='dT_dn_loss',
    patience=1000,
    restore_best_weights=True,
)

model.create_res_net_model()
model.build(input_shape=(1, 500))
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=initial_learning_rate, amsgrad=False
    ),
    loss={'T': 'mse', 'dT_dn': 'mse'},
    loss_weights={'T': 0.2, 'dT_dn': 1.0}, # As recommended by Manuel: scale the loss in T by 0.2
    metrics={'T': ['mae'], 'dT_dn': ['mae']})

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=path + 'cp.ckpt',
    save_freq=1000
)

model.models.summary()

weights_before_train = model.layers[0].get_weights()[0]
import time

start = time.time()
with tf.device('/device:GPU:0'):
    model.fit(
        x=density,
        y=targetdata,
        epochs=epoch,
        verbose=2,
        validation_data=(
            density_test, {'T': kinetic_test, 'dT_dn': kinetic_derivativ_test}
        ),
        validation_freq=100,
        callbacks=[callback, cp_callback])
end = time.time()
print(f'time elapsed {end - start}')
assert False
weights_after_train = model.layers[0].get_weights()[0]


plot_gaussian_weights_v2(weights_before_train, mean, stddev, kernel_size, ' before', path)
plot_gaussian_weights_v2(weights_after_train, mean, stddev, kernel_size, ' after', path)

x = np.linspace(0, 1, 500)
plot_derivative_energy(x, kinetic_derivativ_test, model, density_test, path)

df = save_losses(model, path)

plot_losses(df=df, path=path)
