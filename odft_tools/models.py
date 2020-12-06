import numpy as np
import keras
import tensorflow as tf

from scipy.linalg import cho_solve, cholesky
from odft_tools.kernels import RBFKernel
from odft_tools.utils import (first_derivative_matrix,
                              second_derivative_matrix,
                              integrate)
from odft_tools.layers import (
    Continuous1DConvV1,
    IntegrateLayer
)
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.eager import monitoring


class ResNetConv1DModel(keras.Model):
    def __init__(
            self,
            filter_size,
            kernel_size,
            layer_size,
            num_res_nat_blocks,
            n_outputs,
            dx=1.0):

        super(ResNetConv1DModel, self).__init__()
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.layer_size = layer_size
        self.num_res_nat_blocks = num_res_nat_blocks
        self.n_outputs = n_outputs
        self.dx = dx
        self.conv_layers = []

    def construct_layers_res_net(self):
        for l in range(self.num_res_nat_blocks):
            layer_with_act = tf.keras.layers.Conv1D(
                filters=self.filter_size,
                kernel_size=self.kernel_size,
                   activation='softplus',
                padding='same'
            )

            self.conv_layers.append(layer_with_act)
            # res_net layer for '+ x'
            layer_without_act = tf.keras.layers.Conv1D(
                filters=self.filter_size,
                kernel_size=self.kernel_size,
                activation=None,
                padding='same'
            )

            self.conv_layers.append(layer_without_act)
        # last layer is fixed to use a single filter
        layer = tf.keras.layers.Conv1D(
                filters=1,
                kernel_size=self.kernel_size,
                activation=None,
                padding='same'
            )
        self.conv_layers.append(layer)
        self.integrate = IntegrateLayer(self.dx)

    def call(self, inputs):
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            # Calculate kinetic energy density tau by applying convolutional layers
            tau = inputs
            for i, layer in enumerate(self.conv_layers):
                tau = layer(tau)
                # apply F(x) + x to layer without act. fun
                if (i > 0) and (i % 2 == 0) and (i < len(self.conv_layers) - 1):
                    tau = tf.keras.layers.Add()([tau, inputs])
                    tau = tf.keras.layers.Activation('softplus')(tau)
            # Kinetic energy T is integral over kinetiv energy density
            T = self.integrate(tau)
        # The discretized derivative needs to be divided by dx
        dT_dn = tape.gradient(T, inputs)/self.dx
        return {'T': T, 'dT_dn': dT_dn}


class ResNetContConv1DModel(ResNetConv1DModel):
    def __init__(
            self,
            weights_gaus,
            random_init=True,
            **kwargs):
        # super(ResNetContConv1DModel, self).__init__()
        super().__init__(**kwargs)

        self.weights_gaus = weights_gaus
        self.random_init = random_init
        self.conv_layers = []

    def construct_layers_res_net(self):
        for l in range(self.num_res_nat_blocks):
            if l == 0:
                layer_with_act = Continuous1DConvV1(
                    filters=self.filter_size,
                    kernel_size=self.kernel_size,
                    activation='softplus',
                    padding='same',
                    weights_init=self.weights_gaus,
                    random_init=self.random_init
                )
            else:
                layer_with_act = tf.keras.layers.Conv1D(
                    filters=self.filter_size,
                    kernel_size=self.kernel_size,
                    activation='softplus',
                    padding='same'
                )
            self.conv_layers.append(layer_with_act)

            layer_without_act = tf.keras.layers.Conv1D(
                    filters=self.filter_size,
                    kernel_size=self.kernel_size,
                    activation=None,
                    padding='same'
                )

            self.conv_layers.append(layer_without_act)

        # last layer is fixed to use a single filter
        layer = tf.keras.layers.Conv1D(
                filters=1,
                kernel_size=self.kernel_size,
                activation='linear',
                padding='same'
            )
        self.conv_layers.append(layer)
        self.integrate = IntegrateLayer(self.dx)


class ResNetContConv1DModelV2(ResNetConv1DModel):
    def __init__(
            self,
            weights_gaus,
            random_init=True,
            **kwargs):
        # super(ResNetContConv1DModel, self).__init__()
        super().__init__(**kwargs)

        self.weights_gaus = weights_gaus
        self.random_init = random_init
        self.conv_layers = []

    def construct_layers_res_net(self):

        layer_with_act = Continuous1DConvV1(
            filters=self.filter_size,
            kernel_size=self.kernel_size,
            activation='softplus',
            padding='same',
            weights_init=self.weights_gaus,
            random_init=self.random_init
        )

        self.conv_layers.append(layer_with_act)

        layer_without_act = Continuous1DConvV1(
            filters=self.filter_size,
            kernel_size=self.kernel_size,
            activation=None,
            padding='same',
            weights_init=self.weights_gaus,
            random_init=self.random_init
        )
        
        self.conv_layers.append(layer_without_act)

        for l in range(self.num_res_nat_blocks - 1):
            layer_with_act = tf.keras.layers.Conv1D(
                filters=self.filter_size,
                kernel_size=self.kernel_size,
                activation='softplus',
                padding='same'
            )
            self.conv_layers.append(layer_with_act)

            layer_without_act = tf.keras.layers.Conv1D(
                    filters=self.filter_size,
                    kernel_size=self.kernel_size,
                    activation=None,
                    padding='same'
                )

            self.conv_layers.append(layer_without_act)

        # last layer is fixed to use a single filter
        layer = tf.keras.layers.Conv1D(
                filters=1,
                kernel_size=self.kernel_size,
                activation='linear',
                padding='same'
            )
        self.conv_layers.append(layer)
        self.integrate = IntegrateLayer(self.dx)


class Model():
    """Base class for all models"""
    
    def predict(self, n, derivative=False):
        pass
    
    def predict_phi(self, phi, derivative=False):
        """Allow models to overwrite this if the calculation
        can be done easier on the pseudo wave function"""
        n = phi**2
        if derivative:
            T, dT = self.predict(n, derivative=True)
            return T, 2*phi*dT
        return self.predict(n)
    
class SumModel(Model):
    
    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2
        
    def predict(self, n, derivative=False):
        if derivative:
            T1, dT1 = self.model1.predict(n, derivative=True)
            T2, dT2 = self.model2.predict(n, derivative=True)
            return T1+T2, dT1+dT2
        return self.model1.predict(n) + self.model2.predict(n)
    
    def predict_phi(self, phi, derivative=False):
        if derivative:
            T1, dT1 = self.model1.predict_phi(phi, derivative=True)
            T2, dT2 = self.model2.predict_phi(phi, derivative=True)
            return T1+T2, dT1+dT2
        return self.model1.predict_phi(phi) + self.model2.predict_phi(phi)

class CustomKRR(Model):
    
    def __init__(self, kernel=RBFKernel(), lamb=1e-6, kappa=1e-6, h=1.0):
        self.kernel = kernel
        self.lamb = lamb
        self.kappa = kappa
        self.h = h
        
    def fit(self, X_train, Y, dY_dX=None, offset=None, inplace=True):
        n = X_train.shape[0]
        n_dim = X_train.shape[1]
        self.X_train = X_train
        self.fit_deriv = not dY_dX is None
        
        try:
            self.offset = float(offset)
        except (ValueError, TypeError):
            if offset is None:
                self.offset = 0.0
            elif offset == 'mean':
                self.offset = np.mean(Y)
            elif offset == 'max':
                self.offset = np.max(Y)
            elif isinstance(offset, str) and offset.startswith('max+'):
                self.offset = (np.max(Y) + float(offset[4:]))
            else:
                raise NotImplementedError(
                    'Unknown option: %s' % offset)
        
        if self.fit_deriv:
            K = self.kernel(X_train, X_train, dx=True, dy=True, h=self.h)
            #K[:n, n:] /= self.h
            #K[n:, :n] /= self.h
            #K[n:, n:] /= self.h**2
            K[np.diag_indices(n*(1+n_dim))] += np.concatenate(
                [self.lamb*np.ones(n), self.lamb*n_dim/self.kappa*np.ones(n*n_dim)])
            target_vector = np.concatenate([Y - self.offset, dY_dX.flatten()])
        else:
            K = self.kernel(X_train, X_train)
            K[np.diag_indices(n)] += self.lamb*np.ones(n)
            target_vector = Y - self.offset
        if inplace:
            # Using K.T and lower=False instead of K and lower=True for overwrite_a to work
            # check: https://stackoverflow.com/questions/14408873/how-to-do-in-place-cholesky-factorization-in-python
            cholesky(K.T, lower=False, overwrite_a=True)
        else:
            # Somehow the scipy version segfaults for large arrays, fallback to numpy:
            K = np.linalg.cholesky(K)
        self.alpha = cho_solve((K, True), target_vector)
        
        return self
    
    def predict(self, X, derivative=False):
        n = self.X_train.shape[0]
        m = X.shape[0]
        K_star = self.kernel(self.X_train, X, dx=self.fit_deriv, dy=derivative, h=self.h)
        #K_star[:n, m:] /= self.h
        #K_star[n:, :m] /= self.h
        #K_star[n:, m:] /= self.h**2
        out = self.alpha.dot(K_star)
        if derivative:
            return self.offset + out[:m], out[m:].reshape(m, -1)
        return self.offset + out
    
    def save(self, save_file):
        np.savez(save_file, X_train=self.X_train, alpha=self.alpha,
                 offset=self.offset, h=self.h, lamb=self.lamb, 
                 kappa=self.kappa, fit_deriv=self.fit_deriv)
        
    def load(self, save_file):
        data = np.load(save_file)
        self.X_train = data['X_train']
        self.alpha = data['alpha']
        self.offset = data['offset']
        self.h = data['h']
        self.lamb = data['lamb']
        self.kappa = data['kappa']
        self.fit_deriv = data['fit_deriv']


class DerivOnlyKRR(CustomKRR):
            
    def fit(self, X_train, Y, dY_dX, offset='mean'):
        n = X_train.shape[0]
        n_dim = X_train.shape[1]
        self.X_train = X_train
        self.fit_deriv = True
        
        K = self.kernel(X_train, X_train, dx=True, dy=True, h=self.h)[n:, n:]
            #K[:n, n:] /= self.h
            #K[n:, :n] /= self.h
            #K[n:, n:] /= self.h**2
        K[np.diag_indices(n*n_dim)] += self.lamb*n_dim/self.kappa
        target_vector = dY_dX.flatten()

        K = np.linalg.cholesky(K)
        self.alpha = cho_solve((K, True), target_vector)
        
        if offset == 'mean':
            K = self.kernel(X_train, X_train, dx=True, dy=False, h=self.h)[n:, :]
            self.offset = np.mean(Y - self.alpha.dot(K))
        else:
            self.offset = 0
        return self
    
    def predict(self, X, derivative=False):
        n = self.X_train.shape[0]
        m = X.shape[0]
        K_star = self.kernel(self.X_train, X, dx=True, dy=derivative, h=self.h)[n:, :]
        #K_star[:n, m:] /= self.h
        #K_star[n:, :m] /= self.h
        #K_star[n:, m:] /= self.h**2
        out = self.alpha.dot(K_star)
        if derivative:
            return self.offset + out[:m], out[m:].reshape(m, -1)
        return self.offset + out
        

class CustomKRR_new(CustomKRR):
    
    def fit(self, X_train, Y, dY_dX=None, offset=None, inplace=True):
        n = X_train.shape[0]
        n_dim = X_train.shape[1]
        self.X_train = X_train
        self.fit_deriv = not dY_dX is None
        
        try:
            self.offset = float(offset)
        except (ValueError, TypeError):
            if offset is None:
                self.offset = 0.0
            elif offset == 'mean':
                self.offset = np.mean(Y)
            elif offset == 'max':
                self.offset = np.max(Y)
            elif isinstance(offset, str) and offset.startswith('max+'):
                self.offset = (np.max(Y) + float(offset[4:]))
            else:
                raise NotImplementedError(
                    'Unknown option: %s' % offset)
        
        if self.fit_deriv:
            K = self.kernel(X_train, X_train, dx=True, dy=True, h=self.h)
            K[np.diag_indices(n*(1+n_dim))] += np.concatenate(
                [self.lamb*np.ones(n), self.lamb*n_dim/self.kappa*np.ones(n*n_dim)])
            target_vector = np.concatenate([Y - self.offset, dY_dX.flatten()])
        else:
            K = self.kernel(X_train, X_train)
            K[np.diag_indices(n)] += self.lamb*np.ones(n)
            target_vector = Y - self.offset
        if inplace:
            # Using K.T and lower=False instead of K and lower=True for overwrite_a to work
            # check: https://stackoverflow.com/questions/14408873/how-to-do-in-place-cholesky-factorization-in-python
            cholesky(K.T, lower=False, overwrite_a=True)
        else:
            # Somehow the scipy version segfaults for large arrays, fallback to numpy:
            K = np.linalg.cholesky(K)
        self.alpha = cho_solve((K, True), target_vector)
        
        return self
    
    def predict(self, X, derivative=False):
        n = self.X_train.shape[0]
        m = X.shape[0]
        K_star = self.kernel(self.X_train, X, dx=self.fit_deriv, dy=derivative, h=self.h)
        #K_star[:n, m:] /= self.h
        #K_star[n:, :m] /= self.h
        #K_star[n:, m:] /= self.h**2
        out = self.alpha.dot(K_star)
        if derivative:
            return self.offset + out[:m], out[m:].reshape(m, -1)
        return self.offset + out
        

class WeizsaeckerModel(Model):
    """Warning: this model still predicts wrong values for the derivative at 
    the first and last grid point where the density is zero"""

    def __init__(self, G=500, h=1.0):
        self.h = h
        self.first_deriv = first_derivative_matrix(
            G, h, method='five_point')
        self.second_deriv = second_derivative_matrix(
            G, h, method='five_point')
    
    def predict(self, n, derivative=False, return_tau=False):
        phi = np.sqrt(np.abs(n))
        tau = 0.5*(phi.dot(self.first_deriv.T))**2
        T = integrate(tau, self.h, method='trapezoidal')        
        if derivative:
            dT = -phi.dot(self.second_deriv.T)/(2.*(phi+1e-16)**2)*phi
            if return_tau:
                return T, dT, tau
            return T, dT
        if return_tau:
            return T, tau
        return T
    
    def predict_phi(self, phi):
        tau = 0.5*(phi.dot(self.first_deriv.T))**2
        T = integrate(tau, self.h, method='trapezoidal') 
        if derivative:
            dT = -phi.dot(self.second_deriv.T)
            return T, dT
        return T

class ThomasFermiModel(Model):
    
    def __init__(self, h=1.0):
        self.h = h
        
    def predict(self, n, derivative=False):
        tau = np.pi**2/6*n**3
        T = integrate(tau, self.h, method='trapezoidal')
        if derivative:
            dT = np.pi**2/2*n**2
            return T, dT
        return T
    
    
class TFModelWrapper(Model):

    def __init__(self, path, h=1.0, dict_input=True):
        self.h = h
        self.tf_model = tf.saved_model.load(path)
        self.dict_input = dict_input

    def predict(self, n, derivative=False):
        if derivative:
            T, dT = self.tf_predict(n, derivative=True)
            return T.numpy(), dT.numpy()
        return self.tf_predict(n, derivative=False).numpy()

    @tf.function
    def tf_predict(self, n, derivative=False):
        """Helper function needed to build the graph"""
        if derivative:
            with tf.GradientTape() as tape:
                tape.watch(n)
                if self.dict_input:
                    T = self.tf_model({'density': tf.cast(n, dtype=tf.float32)})['kinetic_energy']
                else:
                    T = self.tf_model(tf.cast(n, dtype=tf.float32))['kinetic_energy']
            dT = 1/self.h*tape.gradient(T, n)
            return T, dT
        if self.dict_input:
            return self.tf_model({'density': tf.cast(n, dtype=tf.float32)})['kinetic_energy']
        return self.tf_model(tf.cast(n, dtype=tf.float32))['kinetic_energy']


class CustomKRR_old():
    
    def __init__(self, kernel=RBFKernel(), lamb=1e-6):
        self.kernel = kernel
        self.lamb = lamb
        
    def fit(self, X_train, y=None, dy=None, offset=None):
        n = X_train.shape[0]
        self.X_train = X_train
        K = self.kernel(X_train, X_train)
        L = cholesky(K + self.lamb*np.eye(len(K)), lower=True)

        if offset == 'mean':
            self.offset = np.mean(y)
        else:
            self.offset = 0.0
        target_vector = y - self.offset
        self.alpha = cho_solve((L, True), target_vector)
        
        return self
    
    def predict(self, X, derivative=False):
        n = self.X_train.shape[0]
        m = X.shape[0]
        K_star = self.kernel(self.X_train, X, dy=derivative)
        out = self.alpha.dot(K_star)
        if derivative:            
            return self.offset + out[:m], out[m:].reshape(m, -1)
        return self.offset + out
