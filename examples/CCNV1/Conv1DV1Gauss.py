import tensorflow as tf
import numpy as np

from odft_tools.models_ccnn import (
    CustomCNNV1Model
)

from odft_tools.utils_train import (
    plot_derivative_energy,
    plot_gaussian_weights_v1,
    save_losses,
    plot_losses,
    load_data_cnn
)

from odft_tools.keras_utils import (
    WarmupExponentialDecay
)

from odft_tools.utils import (
    gen_gaussian_kernel_v1_1D
)

data_path = '../datasets/orbital_free_DFT/'
path = 'results/ConvGaussian/'

fitler_size = 32
kernel_size = 100
epoch = 30000
dx = 0.002

seed = 0
tf.random.set_seed(seed)

kinetic_train, kinetic_derivativ_train, density_train = load_data_cnn(
    path=data_path,
    data_name='dataset_large.hdf5'
)

kinetic_test, kinetic_derivativ_test, density_test = load_data_cnn(
    path=data_path,
    data_name='dataset_validate.hdf5'
)

# density = {'n': density_train.astype(np.float32)}
# targetdata = {
#     'T': kinetic_train.astype(np.float32),
#     'dT_dn': kinetic_derivativ_train.astype(np.float32)
# }

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
    patience=1000,
    restore_best_weights=True,
)

model = CustomCNNV1Model(
    filter_size=32,
    kernel_size=kernel_size,
    layer_length=5,
    dx=0.002,
    create_continuous_kernel=gen_gaussian_kernel_v1_1D,
    # kernel_regularizer=tf.keras.regularizers.l2(0.000025)
)

model.build(input_shape=(None, 500, 1))
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=initial_learning_rate, amsgrad=False
    ),
    loss={'T': 'mse', 'dT_dn': 'mse'},
    loss_weights={'T': 0.2, 'dT_dn': 1.0}, # As recommended by Manuel: scale the loss in T by 0.2
    metrics={'T': ['mae'], 'dT_dn': ['mae']})

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=path + 'cp.ckpt',
    save_freq=100
)

model.summary()

training_dataset = tf.data.Dataset.from_tensor_slices(
    (
        density_train.astype(np.float32),
        {'T': kinetic_train.astype(np.float32),
        'dT_dn': kinetic_derivativ_train.astype(np.float32)}
    )
).batch(100).repeat(10)

weights_before_train = model.layers[0].get_weights()[0]

with tf.device('/device:GPU:1'):
    model.fit(
        training_dataset,
        epochs=epoch,
        verbose=2,
        validation_data=(
            density_test, {'T': kinetic_test, 'dT_dn': kinetic_derivativ_test}
        ),
        validation_freq=10, callbacks=[callback, cp_callback]
    )

weights_after_train = model.layers[0].get_weights()[0]

plot_gaussian_weights_v1(weights_before_train, path, 'before')
plot_gaussian_weights_v1(weights_after_train, path, 'after')

x = np.linspace(0, 1, 500)

plot_derivative_energy(x, kinetic_derivativ_train, model, density_train, path)

df = save_losses(model, path)
# plot_losses(loss=df, path=path)
