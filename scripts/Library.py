import numpy as np
import matplotlib.pyplot as plt
from abc import ABC
from filterpy.kalman import EnsembleKalmanFilter as EnKF_model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

class Data_Assimilation(ABC):
    def __init__(self, dim_x, dim_z, f, h, get_data, dt=1, t0=0):
        self.dim_x = dim_x              # Dimension of state variables
        self.dim_z = dim_z              # Dimension of data
        self.dt = dt                    # Time step size
        self.get_data = get_data        # Function for the acquisition of data
        self.x = np.zeros((dim_x,))     # State variable vector
        self.z = np.zeros((dim_z,))     # Data vector
        self.f = f                      # Transition function
        self.h = h                      # Measurament function
        self.t0 = t0                    # Initial time
        self.t = t0                     # Current time
        self.model = None               # Model for data acquisition, specified by each subclass

class EnKF(Data_Assimilation):
    def create_model(self, x0, P, R, Q, N=1000):
        self.model = EnKF_model(x=x0, P=P, dim_z=self.dim_z, dt=self.dt, N=N,
            hx=self.h, fx=self.f)
        self.model.R = R
        self.model.Q = Q

    def predict(self):
        self.model.predict()

    def update(self, z):
        self.t += self.dt
        self.model.update(z)

    def predict_and_update(self):
        if self.get_data == None:
            self.z = self.model.x
        else:
            self.z = self.get_data(self.t)
        self.predict()
        self.update(self.z)

    def loop(self, T, verbose=False):
        Nt = np.int32((T-self.t0)/self.dt)
        x_hat = np.zeros((Nt+1, self.dim_x))
        x_hat[0,:] = self.model.x
        if  self.t >= T:
            raise "Current time is {} that is less or equal to end time {}".format(self.t, T)
        for i in range(Nt):
            if verbose:
                print('Advancing: ' + str(i/Nt*100) + '%')
            self.predict_and_update()
            x_hat [i+1,:] = self.model.x
        return x_hat

@tf.keras.utils.register_keras_serializable()
class FFT_Layer(tf.keras.layers.Layer):
    def __init__(self, k_max=None, **kwargs):
        super(FFT_Layer, self).__init__(**kwargs)
        self._fft_shape = None
        self._ifft_shape = None
        self.k_max = k_max

    def build(self, input_shape):
        if self.k_max == None:
            self._fft_shape = tf.convert_to_tensor(input_shape[-1] // 2 + 1, dtype=tf.int32)
            self._ifft_shape = tf.multiply(tf.convert_to_tensor(input_shape[-1] // 2, dtype=tf.int32), 2)
        else:
            self._fft_shape = tf.convert_to_tensor(self.k_max, dtype=tf.int32)
            self._ifft_shape = tf.multiply(tf.convert_to_tensor(self.k_max-1, dtype=tf.int32), 2)
        print('fft_shape set:', self._fft_shape.numpy())
        print('ifft_shape set:', self._ifft_shape.numpy())

        self.kernel = self.add_weight(
            name="kernel",
            shape=(self.fft_shape, self._fft_shape),
            initializer="glorot_uniform",
            trainable=True
        )

    def call(self, inputs):
        fft = tf.signal.rfft(inputs)
        if not(self.k_max==None):
            fft = fft[..., :self.k_max]
        kernel_complex = tf.complex(self.kernel, tf.zeros_like(self.kernel))
        r = tf.linalg.matmul(fft, kernel_complex)
        ifft = tf.signal.irfft(r)
        return ifft
    
    def get_config(self):
        config = super().get_config()
        config["k_max"] = self.k_max
        return config


    @property
    def fft_shape(self):
        return self._fft_shape

    @property
    def ifft_shape(self):
        return self._ifft_shape

@tf.keras.utils.register_keras_serializable()
class Bias_Layer(tf.keras.layers.Layer):
    def __init__(self, fft_layer_object, **kwargs):
        super(Bias_Layer, self).__init__(**kwargs)
        self.fft_layer_object = fft_layer_object

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_shape[-1], self.fft_layer_object.ifft_shape),
            initializer="glorot_uniform",
            trainable=True
        )
        print('Bias layer has shape: '+str(self.fft_layer_object.ifft_shape.numpy()))

    def call(self, inputs):
        bias = tf.linalg.matmul(inputs, self.kernel)
        return bias

    def get_config(self):
        config = super().get_config()
        config["fft_layer_object"] = self.fft_layer_object
        return config
    
@tf.keras.utils.register_keras_serializable()
class Fourier_Layer(tf.keras.layers.Layer):
    def __init__(self, k_max=None, **kwargs):
        super(Fourier_Layer, self).__init__(**kwargs)
        self.fft_layer = FFT_Layer(k_max=k_max)
        self.bias_layer = Bias_Layer(self.fft_layer)
        self.k_max = k_max

    def call(self, inputs):
        fft_layer = self.fft_layer(inputs)
        bias_layer = self.bias_layer(inputs)
        added_layers = layers.Add() ([fft_layer, bias_layer])
        return layers.Activation('relu') (added_layers)
    
    def get_config(self):
        config = super().get_config()
        config["k_max"] = self.k_max
        return config
    
def FNO(INPUTDIM, OUTPUTDIM, p_dim, n, k_max=None, verbose=False, model_name='FNO', dropout=0.0, kernel_reg=0.0):
    input_layer = layers.Input(shape = INPUTDIM, name= 'input_layer')
    P_layer = layers.Dense(p_dim, activation='relu', kernel_regularizer = regularizers.l2(kernel_reg), name='P_layer') (input_layer)
    P_layer = layers.Dropout(dropout) (P_layer)
    # Repeat the custom module 'n' times
    for i in range(n):
        if verbose:
            print('Creating Fourier Layer ' +str(i))
        if i ==0:
            fourier_module_output = Fourier_Layer(name='fourier_layer_'+str(i), k_max=k_max)(P_layer)
        else:
            fourier_module_output = Fourier_Layer(name='fourier_layer_'+str(i), k_max=k_max)(fourier_module_output)
    output_layer = layers.Dense(OUTPUTDIM[0], activation='linear', kernel_regularizer = regularizers.l2(kernel_reg), name='output_layer') (fourier_module_output)
    output_layer= layers.Dropout(dropout) (output_layer)
    if verbose:
        print('-------------------------------------------------------')
    model = tf.keras.Model(inputs=input_layer, outputs = output_layer, name = model_name)
    if verbose:
        model.summary()
    return model

@tf.keras.utils.register_keras_serializable()
class FFT_Layer_2D(tf.keras.layers.Layer):
    def __init__(self, k_max=None, **kwargs):
        super(FFT_Layer_2D, self).__init__(**kwargs)
        self._fft_shape = None
        self._ifft_shape = None
        self.k_max = k_max

    def build(self, input_shape):
        if self.k_max == None:
            self._fft_shape = tf.convert_to_tensor(input_shape[-1] // 2 + 1, dtype=tf.int32)
            self._ifft_shape = tf.multiply(tf.convert_to_tensor(input_shape[-1] // 2, dtype=tf.int32), 2)
        else:
            self._fft_shape = tf.convert_to_tensor(self.k_max, dtype=tf.int32)
            self._ifft_shape = tf.multiply(tf.convert_to_tensor(self.k_max-1, dtype=tf.int32), 2)
        print('fft_shape set:', self._fft_shape.numpy())
        print('ifft_shape set:', self._ifft_shape.numpy())

        self.kernel = self.add_weight(
            name="kernel",
            shape=(self._fft_shape, self._fft_shape),
            initializer="glorot_uniform",
            trainable=True
        )

    def call(self, inputs):
        fft = tf.signal.rfft2d(inputs)
        if not(self.k_max==None):
            fft = fft[..., :self.k_max]
        kernel_complex = tf.complex(self.kernel, tf.zeros_like(self.kernel))
        r = tf.linalg.matmul(fft, kernel_complex)
        ifft = tf.signal.irfft2d(r)
        return ifft
    
    def get_config(self):
        config = super().get_config()
        config["k_max"] = self.k_max
        return config
    
    @property
    def fft_shape(self):
        return self._fft_shape

    @property
    def ifft_shape(self):
        return self._ifft_shape

@tf.keras.utils.register_keras_serializable()
class Bias_Layer_2D(tf.keras.layers.Layer):
    def __init__(self, fft_layer_object, **kwargs):
        super(Bias_Layer_2D, self).__init__(**kwargs)
        self.fft_layer_object = fft_layer_object

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_shape[-1], self.fft_layer_object.ifft_shape),
            initializer="glorot_uniform",
            trainable=True
        )
        print('Bias layer has shape: '+str(self.fft_layer_object.ifft_shape.numpy()))

    def call(self, inputs):
        bias = tf.linalg.matmul(inputs, self.kernel)
        return bias

    def get_config(self):
        config = super().get_config()
        config["fft_layer_object"] = self.fft_layer_object
        return config
    
@tf.keras.utils.register_keras_serializable()
class Fourier_Layer_2D(tf.keras.layers.Layer):
    def __init__(self, k_max=None, **kwargs):
        super(Fourier_Layer_2D, self).__init__(**kwargs)
        self.fft_layer = FFT_Layer_2D(k_max=k_max)
        self.bias_layer = Bias_Layer_2D(self.fft_layer)
        self.k_max = k_max

    def call(self, inputs):
        fft_layer = self.fft_layer(inputs)
        bias_layer = self.bias_layer(inputs)
        added_layers = layers.Add() ([fft_layer, bias_layer])
        return layers.Activation('relu') (added_layers)
    
    def get_config(self):
        config = super().get_config()
        config["k_max"] = self.k_max
        return config
    
def FNO2D(INPUTDIM, OUTPUTDIM, p_dim, n, k_max=None, verbose=False, model_name='FNO2D', dropout=0.0, kernel_reg=0.0):
    input_layer = layers.Input(shape = INPUTDIM, name= 'input_layer')
    input_layer_flat = layers.Reshape((INPUTDIM[0]*INPUTDIM[1],)) (input_layer)
    P_layer = layers.Dense(p_dim**2, activation='relu', kernel_regularizer = regularizers.l2(kernel_reg), name='P_layer') (input_layer_flat)
    P_layer = layers.Dropout(dropout) (P_layer)
    P_layer = layers.Reshape((p_dim, p_dim)) (P_layer)
    # Repeat the custom module 'n' times
    for i in range(n):
        if verbose:
            print('Creating Fourier Layer ' +str(i))
        if i ==0:
            fourier_module_output = Fourier_Layer_2D(name='fourier_layer_'+str(i), k_max=k_max)(P_layer)
        else:
            fourier_module_output = Fourier_Layer_2D(name='fourier_layer_'+str(i), k_max=k_max)(fourier_module_output)
    fourier_module_output = layers.Reshape((p_dim*2*(k_max-1),)) (fourier_module_output)
    output_layer_flat = layers.Dense(OUTPUTDIM[0]*OUTPUTDIM[1], activation='linear', kernel_regularizer = regularizers.l2(kernel_reg), name='output_layer') (fourier_module_output)
    output_layer_flat = layers.Dropout(dropout) (output_layer_flat)
    output_layer = layers.Reshape(OUTPUTDIM) (output_layer_flat)
    if verbose:
        print('-------------------------------------------------------')
    model = tf.keras.Model(inputs=input_layer, outputs = output_layer, name = model_name)
    if verbose:
        model.summary()
    return model


