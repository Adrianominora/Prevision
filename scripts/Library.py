import numpy as np
import matplotlib.pyplot as plt
from abc import ABC
from filterpy.kalman import EnsembleKalmanFilter as EnKF_model

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

        
class Data_Simulator():
    def __init__(self, dim_x, dim_z, dt=1):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dt = dt
