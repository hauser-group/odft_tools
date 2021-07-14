import tensorflow as tf
import numpy as np
# from https://github.com/hauser-group/odft_tools
from odft_tools.layers import (
    IntegrateLayer
)

from odft_tools.models_resnet_ccn import (
    ResNetConv1DModel
)

from odft_tools.utils_train import (
    plot_derivative_energy,
    plot_gaussian_weights_v1,
    save_losses,
    plot_losses,
    load_data
)

from odft_tools.keras_utils import (
    WarmupExponentialDecay
)
data_path = 'datasets/orbital_free_DFT/'
path = 'results_new/ResNetNoScale/'

fitler_size = 32
kernel_size = 100
num_res_net_blocks = 3
epoch = 100000

seed = 0
tf.random.set_seed(seed)

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

callback = tf.keras.callbacks.EarlyStopping(
    monitor='dT_dn_loss',
    patience=100000,
    restore_best_weights=True,
)

model = ResNetConv1DModel(
    filter_size=fitler_size,
    kernel_size=kernel_size,
    layer_size=None,
    num_res_net_blocks=num_res_net_blocks,
    n_outputs=None,
    dx=0.002,
    kernel_regularizer=tf.keras.regularizers.l2(0.00025)
)

model.create_res_net_model()
model.build(input_shape=(1, 500))
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=initial_learning_rate, amsgrad=False
    ),
    loss={'T': 'mse', 'dT_dn': 'mse'},
    loss_weights={'T': 1.0, 'dT_dn': 1.0}, # As recommended by Manuel: scale the loss in T by 0.2
    metrics={'T': ['mae'], 'dT_dn': ['mae']})

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=path + 'cp.ckpt',
    save_freq=1000
)

model.models.summary()

import time

start = time.time()
weights_before_train = model.layers[0].get_weights()[0]

model.fit(
    x=density,
    y=targetdata,
    epochs=epoch,
    verbose=2,
    validation_data=(
        density_test, {'T': kinetic_test, 'dT_dn': kinetic_derivativ_test}
    ),
    validation_freq=100, callbacks=[callback]
)

weights_after_train = model.layers[0].get_weights()[0]

end = time.time()
print(f'time elapsed {end - start}')

plot_gaussian_weights_v1(weights_before_train, path, 'before')
plot_gaussian_weights_v1(weights_after_train, path, 'after')

x = np.linspace(0, 1, 500)
plot_derivative_energy(x, kinetic_derivativ_test, model, density_test, path)

df = save_losses(model, density_test, path)

plot_losses(df=df, path=path)

