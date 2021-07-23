import os
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


def save_losses(model, density, path):
    kinetic_energy = tf.squeeze(model(density)['T'])
    derivative_energy = tf.squeeze(model(density[0].reshape((1, 500, 1)).astype(np.float32))['dT_dn'])

    std_kinetic = np.std(kinetic_energy)
    std_derivative = np.std(derivative_energy)

    df = pd.DataFrame([])
    df_val = pd.DataFrame([])
    df['loss'] = model.history.history['loss']
    df['T_loss'] = model.history.history['T_loss']
    df['dT_dn_loss'] = model.history.history['dT_dn_loss']
    df['T_mae'] = model.history.history['T_mae']
    df['dT_dn_mae'] = model.history.history['dT_dn_mae']
    df_val['val_loss'] = model.history.history['val_loss']
    df_val['val_T_loss'] = model.history.history['val_T_loss']
    df_val['val_dT_dn_loss'] = model.history.history['val_dT_dn_loss']
    df_val['val_T_mae'] = model.history.history['val_T_mae']
    df_val['std_T'] = std_kinetic
    df_val['val_dT_dn_mae'] = model.history.history['val_dT_dn_mae']
    df_val['std_dT_dn'] = std_derivative
    df.to_csv(path + 'losses.csv')
    df_val.to_csv(path + 'losses_val.csv')
    return df

def plot_gaussian_weights_v1(weights, path, before_after):
    if not os.path.exists(path):
        os.makedirs(path)
    plt.ylabel('density')
    plt.xlabel('kernel size')
    plt.title('First Layer of model '+ before_after +' train')
    plt.plot(weights[:, 0, :])
    plt.savefig(path + 'layer_'+ before_after +'.png')
    plt.show()
    plt.close()

def plot_derivative_energy(x, dT_dn, model, n, path):
    if not os.path.exists(path):
        os.makedirs(path)
    dT_dn_model = tf.squeeze(model(n[0].reshape((1, 500, 1)).astype(np.float32))['dT_dn'])
    plt.plot(x, dT_dn[0])
    plt.plot(x, dT_dn_model)
    plt.ylabel('dT_dn')
    plt.title('Comparison reference with trained energy derivative')
    plt.savefig(path + 'energy_derivatice.png')
    plt.show()
    plt.close()

def plot_gaussian_weights_v2(weights, kernel_size, before_after, path, kernel_dist):
    if not os.path.exists('results' + path):
        os.makedirs('results' + path)

    truncate = kernel_size/2
    shift = np.random.randint(-kernel_size/4, kernel_size/4)
    shift = 0
    # print(f'shift {shift}')
    width = int(truncate + 0.5)
    support = np.arange(-width - shift, width + 1 - shift)
    # support = np.arange(-width, width + 1)
    center = int(len(support)/2)

    left_cut = center - int(kernel_size/2)
    right_cut = center + int(kernel_size/2)

    for amp, mean, stddev in zip(weights[0][0], weights[1][0], weights[2][0]):

        kernel = kernel_dist(support, amp, mean, stddev)

        if (kernel_size % 2) != 0:
            kernel = kernel[left_cut + 1:right_cut + 2]
        else:
            kernel = kernel[left_cut + 1:right_cut + 1]
        plt.plot(kernel)
    plt.ylabel('density')
    plt.xlabel('kernel size')
    plt.title('First Layer of model '+ before_after +' train')
    plt.savefig(path + 'weights_plot_' + before_after + '.png')
    plt.show()
    plt.close()

def plot_losses(df, path):
    plt.figure(figsize=(20, 3))
    plt.plot(df['loss'])
    plt.xlabel('epochs')
    plt.ylabel('loss a.u.')
    plt.title('loss over epochs')
    plt.savefig(path + 'loss_CNN.png')
    plt.show()
    plt.close()

def to_weights(model, before_after):
    for lay in range(len(model.layers) - 1):
        weights_layer = pd.DataFrame(model.layers[lay].get_weights()[0][:, 0, :])
        if not os.path.exists(path + 'weigths/'):
            os.makedirs(path + 'weigths/')
        weights_layer.to_csv(path + 'weigths/' + 'weights'+ before_after + '_layer' + str(lay) + '.csv')

def load_data(path, data_name):
    with h5py.File(path + data_name, 'r') as f:
        keys = f.keys()
        # build a dict (dataset.value has been deprecated. Use dataset[()] instead.)
        data = {key:f[key][()] for key in keys}

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
    kinetic = np.sum(data['energies'][:, :N], axis=-1) - V
    # kinetic energy derivative
    kinetic_derivative = np.expand_dims(np.sum(data['energies'][:, :N], axis=-1)/N, axis=-1) - data['potential']
    density = n.reshape((-1, 500))

    return kinetic, kinetic_derivative, density


def load_data_cnn(path, data_name):

    with h5py.File(path + data_name, 'r') as f:
        keys = f.keys()
        # build a dict (dataset.value has been deprecated. Use dataset[()] instead.)
        data = {key:f[key][()] for key in keys}

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
    kinetic = np.sum(data['energies'][:, :N], axis=-1) - V
    # kinetic energy derivative
    kinetic_derivative = np.expand_dims(np.sum(data['energies'][:, :N], axis=-1)/N, axis=-1) - data['potential']
    density = n.reshape((-1, 500, 1))


    return kinetic, kinetic_derivative, density