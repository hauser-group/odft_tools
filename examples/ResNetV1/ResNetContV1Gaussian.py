import tensorflow as tf
import numpy as np
# from https://github.com/hauser-group/odft_tools

from odft_tools.models_resnet_ccn import (
    ResNetCostumLayer1DModel
)

from odft_tools.utils import (
    gen_gaussian_kernel_v1_1D
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

data_path = '../datasets/orbital_free_DFT/'
path = 'results/Guassian1DConvV1Sigmoid/'

fitler_size = 32
kernel_size = 100
num_res_net_blocks = 3
epoch = 30000

seed = 0
tf.random.set_seed(seed)

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
    patience=10000,
    restore_best_weights=True,
)

model = ResNetCostumLayer1DModel(
    filter_size=32,
    kernel_size=100,
    layer_size=None,
    num_res_net_blocks=num_res_net_blocks,
    weights_gaus=[5, 5],
    n_outputs=None,
    random_init=True,
    dx=0.002,
    create_continuous_kernel=gen_gaussian_kernel_v1_1D,
    kernel_regularizer=tf.keras.regularizers.l2(0.0),
    activation='sigmoid'
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

import time

start = time.time()
weights_before_train = model.layers[0].get_weights()[0]

with tf.device('/device:GPU:1'):
    model.fit(
        x=density,
        y=targetdata,
        epochs=epoch,
        verbose=0,
        validation_data=(
            density_test, {'T': kinetic_test, 'dT_dn': kinetic_derivativ_test}
        ),
        validation_freq=100, callbacks=[callback, cp_callback]
    )

weights_after_train = model.layers[0].get_weights()[0]

end = time.time()
print(f'time elapsed {end - start}')

plot_gaussian_weights_v1(weights_before_train, path, 'before')
plot_gaussian_weights_v1(weights_after_train, path, 'after')

x = np.linspace(0, 1, 500)
plot_derivative_energy(x, kinetic_derivativ_test, model, density_test, path)

df = save_losses(model, path)

plot_losses(df=df, path=path)
