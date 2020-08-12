"""
===============================================================
Inference with Linear Operator Construction with Kalman Filter
===============================================================
This module implements the LOCK
for Linear-Gaussian state space models
"""

import math

import numpy as np

from .utils import array1d, array2d
from .util_functions import _parse_observations, _last_dims, \
    _determine_dimensionality


class LOCK(object) :
    """Implements the LOCK
    This class implements the LOCK
    for a Linear Gaussian model specified by,
    .. math::
        x_{t+1}   &= F_{t} x_{t} + b_{t} + v_{t} \\
        y_{t}     &= H_{t} x_{t} + d_{t} + w_{t} \\
        [v_{t}, w_{t}]^T &\sim N(0, [[Q_{t}, O], [O, R_{t}]])
    The Kalman Filter is an algorithm designed to estimate
    :math:`P(x_t | y_{0:t})`.  As all state transitions and observations are
    linear with Gaussian distributed noise, these distributions can be
    represented exactly as Gaussian distributions with mean
    `x_filt[t]` and covariances `V_filt[t]`.
    Similarly, the Kalman Smoother is an algorithm designed to estimate
    :math:`P(x_t | y_{0:T-1})`.

    Args:
        observation [n_time, n_dim_obs] {numpy-array, float}
            also known as :math:`y`. observation value
        initial_mean [n_dim_sys] {float} 
            also known as :math:`\mu_0`. initial state mean
        initial_covariance [n_dim_sys, n_dim_sys] {numpy-array, float} 
            also known as :math:`\Sigma_0`. initial state covariance
        transition_matrix [n_dim_sys, n_dim_sys] 
            or [n_dim_sys, n_dim_sys]{numpy-array, float}
            also known as :math:`F`. transition matrix from x_{t-1} to x_{t}
        observation_matrix [n_time, n_dim_sys, n_dim_obs] or [n_dim_sys, n_dim_obs]
             {numpy-array, float}
            also known as :math:`H`. observation matrix from x_{t} to y_{t}
        transition_covariance [n_time - 1, n_dim_noise, n_dim_noise]
             or [n_dim_sys, n_dim_noise]
            {numpy-array, float}
            also known as :math:`Q`. system transition covariance for times
        observation_covariance [n_time, n_dim_obs, n_dim_obs] {numpy-array, float} 
            also known as :math:`R`. observation covariance for times.
        update_interval {int}
            : interval of update transition matrix F
        eta (in (0,1])
            : update rate for transition matrix F
        cutoff {float}
            : cutoff distance for update transition matrix F
        store_transition_matrices_on {bool}
            : if True, store transition matrices regarding all update.
            you can extract the matrices via a function "get_transition_matrices"
        calculate_variance {bool} (under construction)
            : if True, calculate variance of estimator
        n_dim_sys {int}
            : dimension of system transition variable
        n_dim_obs {int}
            : dimension of observation variable
        dtype {type}
            : data type of numpy-array
        use_gpu {bool}
            wheather use gpu and cupy.
            if True, you need install package `cupy`.
            if False, set `numpy` for calculation.

    Attributes:
        y : `observation`
        F : `transition_matrix`
        Q : `transition_covariance`
        H : `observation_matrix`
        R : `observation_covariance`
    """

    def __init__(self, observation = None,
                initial_mean = None, initial_covariance = None,
                transition_matrix = None, observation_matrix = None,
                transition_covariance = None, observation_covariance = None,
                estimation_length = 10, estimation_interval = 1,
                eta = 1., cutoff = 10., estimation_mode = "backward",
                store_transition_matrices_on = True,
                calculate_variance = False,
                n_dim_sys = None, n_dim_obs = None, dtype = "float32",
                use_gpu = False):
        """Setup initial parameters.
        """
        self.use_gpu = use_gpu
        if use_gpu:
            try:
                import cupy
                self.xp = cupy
                self.xp_type = "cupy"
            except:
                self.xp = np
                self.xp_type = "numpy"
                self.use_gpu = False
        else:
            self.xp = np
            self.xp_type = "numpy"

        # determine dimensionality
        self.n_dim_sys = _determine_dimensionality(
            [(transition_matrix, array2d, -2),
             (initial_mean, array1d, -1),
             (initial_covariance, array2d, -2),
             (observation_matrix, array2d, -1)],
            n_dim_sys,
            self.xp_type
        )

        self.n_dim_obs = _determine_dimensionality(
            [(observation_matrix, array2d, -2),
             (observation_covariance, array2d, -2)],
            n_dim_obs,
            self.xp_type
        )

        # self.y = _parse_observations(observation)
        self.y = self.xp.asarray(observation).copy()

        if initial_mean is None:
            self.initial_mean = self.xp.zeros(self.n_dim_sys, dtype = dtype)
        else:
            self.initial_mean = initial_mean.astype(dtype)
        
        if initial_covariance is None:
            self.initial_covariance = self.xp.eye(self.n_dim_sys, dtype = dtype)
        else:
            self.initial_covariance = initial_covariance.astype(dtype)

        if transition_matrix is None:
            self.F = self.xp.eye(self.n_dim_sys, dtype = dtype)
        else:
            self.F = transition_matrix.astype(dtype)

        if transition_covariance is not None:
            self.Q = transition_covariance.astype(dtype)
        else:
            self.Q = self.xp.eye(self.n_dim_sys, dtype = dtype)

        if observation_matrix is None:
            self.H = self.xp.eye(self.n_dim_obs, self.n_dim_sys, dtype = dtype)
        else:
            self.H = observation_matrix.astype(dtype)
        self.HI = self.xp.linalg.pinv(self.H)
        
        if observation_covariance is None:
            self.R = self.xp.eye(self.n_dim_obs, dtype = dtype)
        else:
            self.R = observation_covariance.astype(dtype)

        if estimation_mode in ["forward", "middle", "backward"]:
            self.estimation_mode = estimation_mode
        else:
            raise ValueError("\"estimation_mode\" must be choosen from \"forward\","
                            + " \"middle\", or \"backward\".")

        if estimation_length < self.n_dim_obs:
            raise ValueError("\"estimation_length\" must be larger than"
                            + " or equal to the observation dimension")
        else:
            if self.estimation_mode in ["forward", "backward"]:
                self.tau = int(estimation_length)
                self.tau2 = int((estimation_length - 1) / 2)
            else:
                self.tau2 = int((estimation_length - 1) / 2)
                self.tau = 2 * self.tau2 + 1

        self.I = estimation_interval

        if calculate_variance:
            self.calculate_variance = True
            self.store_transition_matrices_on = True
        else:
            self.store_transition_matrices_on = store_transition_matrices_on
            self.calculate_variance = False

        if store_transition_matrices_on:
            self.store_transition_matrices_count = 1
            self.Fs = self.xp.zeros(((len(self.y) - self.tau)//self.I,
                                self.F.shape[0], self.F.shape[1]))
            if calculate_variance:
                self.FV = self.xp.zeros((len(self.y-1)//self.update_interval+1,
                                self.F.shape[0], self.F.shape[1]))

        self.eta = eta
        self.cutoff = cutoff
        self.dtype = dtype


    def forward(self):
        """Calculate prediction and filter for observation times.

        Attributes:
            T {int}
                : length of data y
            x_pred [n_time, n_dim_sys] {numpy-array, float}
                : mean of hidden state at time t given observations
                 from times [0...t-1]
            V_pred [n_time, n_dim_sys, n_dim_sys] {numpy-array, float}
                : covariance of hidden state at time t given observations
                 from times [0...t-1]
            x_filt [n_time, n_dim_sys] {numpy-array, float}
                : mean of hidden state at time t given observations from times [0...t]
            V_filt [n_time, n_dim_sys, n_dim_sys] {numpy-array, float}
                : covariance of hidden state at time t given observations
                 from times [0...t]
        """

        T = self.y.shape[0]
        self.x_pred = self.xp.zeros((T, self.n_dim_sys), dtype = self.dtype)
        self.V_pred = self.xp.zeros((T, self.n_dim_sys, self.n_dim_sys),
             dtype = self.dtype)
        self.x_filt = self.xp.zeros((T, self.n_dim_sys), dtype = self.dtype)
        self.V_filt = self.xp.zeros((T, self.n_dim_sys, self.n_dim_sys),
             dtype = self.dtype)

        # calculate prediction and filter for every time
        for t in range(T) :
            # visualize calculating time
            print("\r filter calculating... t={}".format(t) + "/" + str(T), end="")

            if t == 0:
                # initial setting
                self.x_pred[0] = self.initial_mean
                self.V_pred[0] = self.initial_covariance
                self._calculate_initial_transition_matrix()
            else:
                if t >= 2 and t < T-self.tau+1 and (t-1)%self.I==0 and self.estimation_mode=="forward":
                    self._update_transition_matrix(t)
                elif t >= self.tau+1 and (t-self.tau)%self.I==0 and self.estimation_mode=="backward":
                    self._update_transition_matrix(t)
                elif t >= self.tau2+2 and t < T-self.tau2 and (t-self.tau2-1)%self.I==0 and self.estimation_mode=="middle":
                    self._update_transition_matrix(t)
                self._predict_update(t)
            
            self._filter_update(t)


    def _predict_update(self, t):
        """Calculate fileter update

        Args:
            t {int} : observation time
        """
        # extract parameters for time t-1
        Q = _last_dims(self.Q, t - 1, 2, self.xp_type)

        # calculate predicted distribution for time t
        self.x_pred[t] = self.F @ self.x_filt[t-1]
        self.V_pred[t] = self.F @ self.V_filt[t-1] @ self.F.T + Q


    def _filter_update(self, t):
        """Calculate fileter update without noise

        Args:
            t {int} : observation time

        Attributes:
            K [n_dim_sys, n_dim_obs] {numpy-array, float}
                : Kalman gain matrix for time t
        """
        # extract parameters for time t
        R = _last_dims(self.R, t, 2, self.xp_type)

        # calculate filter step
        K = self.V_pred[t] @ (
            self.H.T @ self.xp.linalg.pinv(self.H @ (self.V_pred[t] @ self.H.T) + R)
            )
        self.x_filt[t] = self.x_pred[t] + K @ (
            self.y[t] - (self.H @ self.x_pred[t])
            )
        self.V_filt[t] = self.V_pred[t] - K @ (self.H @ self.V_pred[t])


    def _calculate_initial_transition_matrix(self):
        self.F = self.HI @ self.y[1:self.tau+1].T \
                @ self.xp.linalg.pinv(self.y[:self.tau].T) @ self.H
        if self.store_transition_matrices_on:
            self.Fs[0] = self.F.copy()


    def _update_transition_matrix(self, t):
        """Update transition matrix

        Args:
            t {int} : observation time
        """
        if self.estimation_mode=="forward":
            Fh = self.HI @ self.y[t:t+self.tau].T \
                    @ self.xp.linalg.pinv(self.y[t-1:t+self.tau-1].T) @ self.H
        elif self.estimation_mode=="middle":
            Fh = self.HI @ self.y[t-self.tau2:t+self.tau2+1].T \
                    @ self.xp.linalg.pinv(self.y[t-self.tau2-1:t+self.tau2].T) @ self.H
        elif self.estimation_mode=="backward":
            Fh = self.HI @ self.y[t-self.tau+1:t+1].T \
                    @ self.xp.linalg.pinv(self.y[t-self.tau:t].T) @ self.H
        self.F = self.F - self.eta * self.xp.minimum(self.xp.maximum(-self.cutoff, self.F - Fh), self.cutoff)

        if self.store_transition_matrices_on:
            self.Fs[self.store_transition_matrices_count] = self.F.copy()
            self.store_transition_matrices_count += 1


    def _update_transition_matrix_with_variance(self, t):
        """Update transition matrix with variance of transition matrix
        (this function is under construction)

        Args:
            t {int} : observation time
        """
        H  = _last_dims(self.H, t, 2)
        R = _last_dims(self.R, t, 2)
        Rb = _last_dims(self.R, t-1, 2)

        Gh = self.y[t-self.update_interval+1:t+1].T \
                @ self.xp.linalg.pinv(self.y[t-self.update_interval:t].T) 
        Fh = self.xp.linalg.pinv(H) @ Gh @ H
        self.F = self.F - self.eta * self.xp.minimum(self.xp.maximum(-self.cutoff, self.F - Fh), self.cutoff)
        self.Fs[t//self.update_interval] = self.F

        # Calculate variance of observation transition
        Sh = H @ self.V_pred[t] @ H.T + R - Gh @ (H @ self.V_filt[t-1] @ H.T + Rb) @ Gh.T

        # Calculate variance of transition matrix
        nu = self.xp.zeros(self.n_dim_sys * self.update_interval, dtype=self.dtype)
        kappa = self.xp.zeros(self.n_dim_sys * self.update_interval, dtype=self.dtype)
        X = self.xp.zeros((self.n_dim_sys * self.update_interval, self.n_dim_sys**2), dtype=self.dtype)

        for i,s in enumerate(range(t-self.update_interval, t)):
            Q = _last_dims(self.Q, s, 2)
            R = _last_dims(self.R, s, 2)

            X0 = self.xp.square(self.x_filt[s]) + self.xp.diag(self.V_filt[s])
            for j in range(self.n_dim_sys):
                X[self.n_dim_sys*i+j, self.n_dim_sys*j:self.n_dim_sys*(j+1)] = X0
            nu0 = self.xp.diag(self.V_pred[s+1])
            nu[self.n_dim_sys*i:self.n_dim_sys*(i+1)] = nu0
            kappa0 = self.xp.diag(self.xp.linalg.pinv(H) @ (Gh @ (H @ self.V_filt[s] @ H.T) @ Gh.T \
                        + Sh - R) @ self.xp.linalg.pinv(H.T) - Q)
            kappa[self.n_dim_sys*i:self.n_dim_sys*(i+1)] = kappa0

        self.FV[t//self.update_interval] = (self.xp.linalg.pinv(X) @ (kappa - nu))\
                                            .reshape(self.n_dim_sys, self.n_dim_sys)


    def get_predicted_value(self, dim = None):
        """Get predicted value

        Args:
            dim {int} : dimensionality for extract from predicted result

        Returns (numpy-array, float)
            : mean of hidden state at time t given observations
            from times [0...t-1]
        """
        # if not implement `forward`, implement `forward`
        try :
            self.x_pred[0]
        except :
            self.forward()

        if dim is None:
            return self.x_pred
        elif dim <= self.x_pred.shape[1]:
            return self.x_pred[:, int(dim)]
        else:
            raise ValueError('The dim must be less than '
                 + self.x_pred.shape[1] + '.')


    def get_filtered_value(self, dim = None):
        """Get filtered value

        Args:
            dim {int} : dimensionality for extract from filtered result

        Returns (numpy-array, float)
            : mean of hidden state at time t given observations
            from times [0...t]
        """
        # if not implement `forward`, implement `forward`
        try :
            self.x_filt[0]
        except :
            self.forward()

        if dim is None:
            return self.x_filt
        elif dim <= self.x_filt.shape[1]:
            return self.x_filt[:, int(dim)]
        else:
            raise ValueError('The dim must be less than '
                 + self.x_filt.shape[1] + '.')


    def get_transition_matrices(self, ids = None):
        """Get transition matrices
        
        Args:
            ids {numpy-array, int} : ids of transition matrices

        Returns {numpy-array, float}:
            : transition matrices
        """
        if self.store_transition_matrices_on:
            if ids is None:
                return self.Fs
            else:
                return self.Fs[ids]
        else:
            return self.F


    def get_variance_of_transition_matrix(self, ids = None):
        """Get transition matrices
        
        Args:
            ids {numpy-array, int} : ids of transition matrices

        Returns {numpy-array, float}:
            : transition matrices
        """
        if self.calculate_variance:
            if ids is None:
                return self.FV
            else:
                return self.FV[ids]
        else:
            print("Need to be \"calculate_variance\" on.")


    def smooth(self):
        """Calculate RTS smooth for times.

        Args:
            T : length of data y
            x_smooth [n_time, n_dim_sys] {numpy-array, float}
                : mean of hidden state distributions for times
                 [0...n_times-1] given all observations
            V_smooth [n_time, n_dim_sys, n_dim_sys] {numpy-array, float}
                : covariances of hidden state distributions for times
                 [0...n_times-1] given all observations
            A [n_dim_sys, n_dim_sys] {numpy-array, float}
                : fixed interval smoothed gain
        """

        # if not implement `forward`, implement `forward`
        try :
            self.x_pred[0]
        except :
            self.forward()

        T = self.y.shape[0]
        self.x_smooth = self.xp.zeros((T, self.n_dim_sys), dtype = self.dtype)
        self.V_smooth = self.xp.zeros((T, self.n_dim_sys, self.n_dim_sys),
             dtype = self.dtype)
        A = self.xp.zeros((self.n_dim_sys, self.n_dim_sys), dtype = self.dtype)

        self.x_smooth[-1] = self.x_filt[-1]
        self.V_smooth[-1] = self.V_filt[-1]

        # t in [0, T-2] (notice t range is reversed from 1~T)
        for t in reversed(range(T - 1)) :
            # visualize calculating times
            print("\r smooth calculating... t={}".format(T - t)
                 + "/" + str(T), end="")

            # calculate fixed interval smoothing gain
            A = self.xp.dot(self.V_filt[t], self.xp.dot(self.F.T, self.xp.linalg.pinv(self.V_pred[t + 1])))
            
            # fixed interval smoothing
            self.x_smooth[t] = self.x_filt[t] \
                + self.xp.dot(A, self.x_smooth[t + 1] - self.x_pred[t + 1])
            self.V_smooth[t] = self.V_filt[t] \
                + self.xp.dot(A, self.xp.dot(self.V_smooth[t + 1] - self.V_pred[t + 1], A.T))

            
    def get_smoothed_value(self, dim = None):
        """Get RTS smoothed value

        Args:
            dim {int} : dimensionality for extract from RTS smoothed result

        Returns (numpy-array, float)
            : mean of hidden state at time t given observations
            from times [0...T]
        """
        # if not implement `smooth`, implement `smooth`
        try :
            self.x_smooth[0]
        except :
            self.smooth()

        if dim is None:
            return self.x_smooth
        elif dim <= self.x_smooth.shape[1]:
            return self.x_smooth[:, int(dim)]
        else:
            raise ValueError('The dim must be less than '
                 + self.x_smooth.shape[1] + '.')
