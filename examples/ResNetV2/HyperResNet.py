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

from tensorboard.plugins.hparams import api as hp

from odft_tools.keras_utils import (
    WarmupExponentialDecay
)

data_path = '../datasets/orbital_free_DFT/'
path = 'results/ResNetHyperV2/'

HP_LR = hp.HParam('lr', hp.Discrete([1.0, 0.1, 0.01, 0.001]))
METRIC_LOSS = 'loss'
METRIC_T_loss = 'T_loss'
METRIC_dT_dn_loss = 'dT_dn_loss'
METRIC_T_mae = 'T_mae'
METRIC_dT_dn_mae = 'dT_dn_mae'

log_dir = 'logs/hparam_tuning_v2/'

with tf.summary.create_file_writer(log_dir).as_default():
    hp.hparams_config(
    hparams=[HP_FILTER, HP_ACT, HP_KERNEL],
    metrics=[
        hp.Metric(METRIC_LOSS, display_name='loss'),
        hp.Metric(METRIC_T_loss, display_name='T_loss'),
        hp.Metric(METRIC_dT_dn_loss, display_name='dT_dn_loss'),
        hp.Metric(METRIC_T_mae, display_name='T_mae'),
        hp.Metric(METRIC_dT_dn_mae, display_name='dT_dn_mae')
    ],
)

fitler_size = 32
kernel_size = 100
num_res_net_blocks = 3
epoch = 20000

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
    patience=1000,
    restore_best_weights=True,
)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=path + 'cp.ckpt',
    save_freq=1000
)

def train_test_model(hparams):

    model = ResNetCostumLayer1DModel(
        filter_size=hparams[HP_FILTER],
        kernel_size=hparams[HP_KERNEL],
        layer_size=None,
        num_res_net_blocks=num_res_net_blocks,
        weights_gaus=[5, 5],
        n_outputs=None,
        random_init=True,
        dx=0.002,
        create_continuous_kernel=gen_gaussian_kernel_v1_1D,
        kernel_regularizer=tf.keras.regularizers.l2(0.00001),
        activation=hparams[HP_ACT]
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


    with tf.device('/device:GPU:0'):
        model.fit(
            x=density,
            y=targetdata,
            epochs=epoch,
            verbose=0,
            validation_data=(
                density_test, {'T': kinetic_test, 'dT_dn': kinetic_derivativ_test}
            ),
            validation_freq=5, callbacks=[callback, cp_callback]
        )
    loss = model.history.history['val_loss'][-1]
    val_T_loss = model.history.history['val_T_loss'][-1]
    val_dT_dn_loss = model.history.history['val_dT_dn_loss'][-1]
    val_T_mae = model.history.history['val_T_mae'][-1]
    val_dT_dn_mae = model.history.history['val_dT_dn_mae'][-1]
    return loss, val_T_loss, val_dT_dn_loss, val_T_mae, val_dT_dn_mae

def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)
        loss, val_T_loss, val_dT_dn_loss, val_T_mae, val_dT_dn_mae = train_test_model(hparams)
        tf.summary.scalar(METRIC_LOSS, loss, step=1)
        tf.summary.scalar(METRIC_T_loss, val_T_loss, step=1)
        tf.summary.scalar(METRIC_dT_dn_loss, val_dT_dn_loss, step=1)
        tf.summary.scalar(METRIC_T_mae, val_T_mae, step=1)
        tf.summary.scalar(METRIC_dT_dn_mae, val_dT_dn_mae, step=1)

session_num = 0

for act in HP_ACT.domain.values:
    for fl_size in HP_FILTER.domain.values:
        for kr_size in HP_KERNEL.domain.values:
            hparams = {
                HP_ACT: act,
                HP_FILTER: fl_size,
                HP_KERNEL: kr_size
            }
            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})
            run(log_dir + run_name, hparams)
            session_num += 1

