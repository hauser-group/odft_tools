import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import numpy as np
# from https://github.com/hauser-group/odft_tools

from odft_tools.models_resnet_ccn import (
    ResNetContConv1DV2Model
)

from odft_tools.utils_train import (
    plot_derivative_energy,
    plot_gaussian_weights_v1,
    save_losses,
    plot_losses,
    load_data
)


optimizer = tf.keras.optimizers.Nadam

data_path = 'datasets/orbital_free_DFT/'
path = 'results/ResNetHyperV2/'

HP_LR = hp.HParam('lr', hp.Discrete([1.0, 0.1, 0.01, 0.001, 0.0001]))
HP_B1 = hp.HParam('beta_1', hp.Discrete([0.9, 0.1, 0.09, 0.009]))
HP_B2 = hp.HParam('beta_2', hp.Discrete([0.999, 0.1, 0.0999, 0.00999]))

METRIC_LOSS = 'loss'
METRIC_T_loss = 'T_loss'
METRIC_dT_dn_loss = 'dT_dn_loss'
METRIC_T_mae = 'T_mae'
METRIC_dT_dn_mae = 'dT_dn_mae'

log_dir = 'logs/hparam_tuning_v2_nadam/'

with tf.summary.create_file_writer(log_dir).as_default():
    hp.hparams_config(
    hparams=[HP_LR, HP_B1, HP_B2],
    metrics=[
        hp.Metric(METRIC_LOSS, display_name='loss'),
        hp.Metric(METRIC_T_loss, display_name='T_loss'),
        hp.Metric(METRIC_dT_dn_loss, display_name='dT_dn_loss'),
        hp.Metric(METRIC_T_mae, display_name='T_mae'),
        hp.Metric(METRIC_dT_dn_mae, display_name='dT_dn_mae')
    ],
)

amp = 0.5
mean = 5
stddev = 10
distribution = 'gaussian'

fitler_size = 32
kernel_size = 100
num_res_net_blocks = 3
epoch = 20000

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

callback = tf.keras.callbacks.EarlyStopping(
    monitor='dT_dn_loss',
    patience=1000,
    restore_best_weights=True,
)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=path + 'cp.ckpt',
    save_freq=1000
)

training_dataset = tf.data.Dataset.from_tensor_slices(
    (
        density_train.astype(np.float32),
        {'T': kinetic_train.astype(np.float32),
        'dT_dn': kinetic_derivativ_train.astype(np.float32)}
    )
).batch(100)

def train_test_model(hparams):

    model = ResNetContConv1DV2Model(
        filter_size=32,
        kernel_size=100,
        layer_size=None,
        num_res_net_blocks=num_res_net_blocks,
        weights_gaus=[amp, mean, stddev],
        n_outputs=None,
        random_init=True,
        dx=0.002,
        distribution=distribution,
        kernel_regularizer=tf.keras.regularizers.l2(0.00001)
    )

    model.create_res_net_model()
    model.build(input_shape=(1, 500))
    model.compile(
        optimizer=optimizer(
            learning_rate=hparams[HP_LR],
            beta_1=hparams[HP_B1],
            beta_2=hparams[HP_B2]
        ),
        loss={'T': 'mse', 'dT_dn': 'mse'},
        loss_weights={'T': 0.2, 'dT_dn': 1.0},
        metrics={'T': ['mae'], 'dT_dn': ['mae']})


    model.fit(
        training_dataset,
        epochs=epoch,
        verbose=2,
        validation_data=(
            density_test, {'T': kinetic_test, 'dT_dn': kinetic_derivativ_test}
        ),
        validation_freq=epoch
    )

    loss = model.history.history['loss'][-1]
    val_T_loss = model.history.history['T_loss'][-1]
    val_dT_dn_loss = model.history.history['dT_dn_loss'][-1]
    val_T_mae = model.history.history['T_mae'][-1]
    val_dT_dn_mae = model.history.history['dT_dn_mae'][-1]
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

for lr in HP_LR.domain.values:
    for beta_1 in HP_B1.domain.values:
        for beta_2 in HP_B2.domain.values:
            hparams = {
                HP_LR: lr,
                HP_B1: beta_1,
                HP_B2: beta_2
            }
            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})
            run(log_dir + run_name, hparams)
            session_num += 1

