"""
=======================================
Inference with Unscented Kalman Filter
=======================================
This module implements the Unscented Kalman Filter and Unscented Kalman Smoother
for state space models
"""

import math

import numpy as np

from .utils import array1d, array2d
from .util_functions import _parse_observations, _last_dims, \
    _determine_dimensionality


class UnscentedKalmanFilter(object) :
    """Implements the Kalman Filter, Kalman Smoother, and EM algorithm.
    This class implements the Kalman Filter, Kalman Smoother, and EM Algorithm
    for a Linear Gaussian model specified by,
    .. math::
        x_{t+1}   &= F_{t} x_{t} + b_{t} + G_{t} v_{t} \\
        y_{t}     &= H_{t} x_{t} + d_{t} + w_{t} \\
        [v_{t}, w_{t}]^T &\sim N(0, [[Q_{t}, S_{t}], [S_{t}, R_{t}]])
    The Kalman Filter is an algorithm designed to estimate
    :math:`P(x_t | y_{0:t})`.  As all state transitions and observations are
    linear with Gaussian distributed noise, these distributions can be
    represented exactly as Gaussian distributions with mean
    `x_filt[t]` and covariances `V_filt[t]`.
    Similarly, the Kalman Smoother is an algorithm designed to estimate
    :math:`P(x_t | y_{0:T-1})`.
    The EM algorithm aims to find for
    :math:`\theta = (F, b, H, d, Q, R, \mu_0, \Sigma_0)`
    .. math::
        \max_{\theta} P(y_{0:T-1}; \theta)
    If we define :math:`L(x_{0:T-1},\theta) = \log P(y_{0:T-1}, x_{0:T-1};
    \theta)`, then the EM algorithm works by iteratively finding,
    .. math::
        P(x_{0:T-1} | y_{0:T-1}, \theta_i)
    then by maximizing,
    .. math::
        \theta_{i+1} = \arg\max_{\theta}
            \mathbb{E}_{x_{0:T-1}} [
                L(x_{0:T-1}, \theta)| y_{0:T-1}, \theta_i
            ]

    Args:
        observation [n_time, n_dim_obs] {numpy-array, float}
            also known as :math:`y`. observation value
        initial_mean [n_dim_sys] {float} 
            also known as :math:`\mu_0`. initial state mean
        initial_covariance [n_dim_sys, n_dim_sys] {numpy-array, float} 
            also known as :math:`\Sigma_0`. initial state covariance
        transition_matrices [n_time - 1, n_dim_sys, n_dim_sys] 
            or [n_dim_sys, n_dim_sys]{numpy-array, float}
            also known as :math:`F`. transition matrix from x_{t-1} to x_{t}
        n_dim_sys {int}
            : dimension of system transition variable
        n_dim_obs {int}
            : dimension of observation variable
        dtype {type}
            : data type of numpy-array

    Attributes:
        y : `observation`
        F : `transition_matrices`
        x_pred [n_time+1, n_dim_sys] {numpy-array, float} 
            mean of predicted distribution
        V_pred [n_time+1, n_dim_sys, n_dim_sys] {numpy-array, float}
            covariance of predicted distribution
        x_filt [n_time+1, n_dim_sys] {numpy-array, float}
            mean of filtered distribution
        V_filt [n_time+1, n_dim_sys, n_dim_sys] {numpy-array, float}
            covariance of filtered distribution
        x_smooth [n_time, n_dim_sys] {numpy-array, float}
            mean of RTS smoothed distribution
        V_smooth [n_time, n_dim_sys, n_dim_sys] {numpy-array, float}
            covariance of RTS smoothed distribution
        filter_update {function}
            update function from x_{t} to x_{t+1}
    """

    def __init__(self, observation = None,
                initial_mean = None, initial_covariance = None,
                transition_functions = None, observation_functions = None,
                n_dim_sys = None, n_dim_obs = None, use_gpu = True,dtype = "float32"):
        """Setup initial parameters.
        """
        if use_gpu:
            import cupy
            self.xp = cupy
        else:
            self.xp = np

        self.y = observation.copy()
        self.n_dim_sys = n_dim_sys
        self.n_dim_obs = n_dim_obs

        if initial_mean is None:
            self.initial_mean = self.xp.zeros(self.n_dim_sys, dtype = dtype)
        else:
            self.initial_mean = initial_mean.astype(dtype)
        
        if initial_covariance is None:
            self.initial_covariance = self.xp.eye(self.n_dim_sys, dtype = dtype)
        else:
            self.initial_covariance = initial_covariance.astype(dtype)

        if transition_functions is None:
            self.f = [lambda x,v : x]
        else:
            self.F = transition_matrices.astype(dtype)

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
            K [n_dim_sys, n_dim_obs] {numpy-array, float}
                : Kalman gain matrix for time t
        """

        T = self.y.shape[0]
        self.x_pred = self.xp.zeros((T, self.n_dim_sys), dtype = self.dtype)
        self.V_pred = self.xp.zeros((T, self.n_dim_sys, self.n_dim_sys),
             dtype = self.dtype)
        self.x_filt = self.xp.zeros((T, self.n_dim_sys), dtype = self.dtype)
        self.V_filt = self.xp.zeros((T, self.n_dim_sys, self.n_dim_sys),
             dtype = self.dtype)
        K = self.xp.zeros((self.n_dim_sys, self.n_dim_obs), dtype = self.dtype)

        # calculate prediction and filter for every time
        for t in range(T) :
            # visualize calculating time
            print("\r filter calculating... t={}".format(t) + "/" + str(T), end="")

            if t == 0:
                # initial setting
                self.x_pred[0] = self.initial_mean
                self.V_pred[0] = self.initial_covariance
            else:
                self._predict_update(t)
            
            # If y[t] has any mask, skip filter calculation
            if (mask and self.xp.any(self.xp.ma.getmask(self.y[t]))) or ((not mask) and self.xp.any(self.xp.isnan(self.y[t]))) :
                self.x_filt[t] = self.x_pred[t]
                self.V_filt[t] = self.V_pred[t]
            else :
                # extract parameters for time t
                H = _last_dims(self.H, t, 2)
                R = _last_dims(self.R, t, 2)
                d = _last_dims(self.d, t, 1)

                # calculate filter step
                K = self.V_pred[t] @ (
                    H.T @ linalg.pinv(H @ (self.V_pred[t] @ H.T) + R)
                    )
                self.x_filt[t] = self.x_pred[t] + K @ (
                    self.y[t] - (H @ self.x_pred[t] + d)
                    )
                self.V_filt[t] = self.V_pred[t] - K @ (H @ self.V_pred[t])
    

    def _predict_update(self, t):
        """Calculate fileter update without noise

        Args:
            t {int} : observation time
        """
        # extract parameters for time t-1
        F = _last_dims(self.F, t - 1, 2)
        Q = _last_dims(self.Q, t - 1, 2)
        b = _last_dims(self.b, t - 1, 1)

        # calculate predicted distribution for time t
        self.x_pred[t] = F @ self.x_filt[t-1] + b
        self.V_pred[t] = F @ self.V_filt[t-1] @ F.T + Q



    def get_predicted_value(self, dim = None):
        """Get predicted value

        Args:
            dim {int} : dimensionality for extract from predicted result

        Returns (numpy-array, float)
            : mean of hidden state at time t given observations
            from times [0...t-1]
        """
        # if not implement `filter`, implement `filter`
        try :
            self.x_pred[0]
        except :
            self.filter()

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
        # if not implement `filter`, implement `filter`
        try :
            self.x_filt[0]
        except :
            self.filter()

        if dim is None:
            return self.x_filt
        elif dim <= self.x_filt.shape[1]:
            return self.x_filt[:, int(dim)]
        else:
            raise ValueError('The dim must be less than '
                 + self.x_filt.shape[1] + '.')


    def smooth(self):
        """Calculate RTS smooth for times.

        Args:
            T : length of data y (時系列の長さ)
            x_smooth [n_time, n_dim_sys] {numpy-array, float}
                : mean of hidden state distributions for times
                 [0...n_times-1] given all observations
            V_smooth [n_time, n_dim_sys, n_dim_sys] {numpy-array, float}
                : covariances of hidden state distributions for times
                 [0...n_times-1] given all observations，
            A [n_dim_sys, n_dim_sys] {numpy-array, float}
                : fixed interval smoothed gain
        """

        # if not implement `filter`, implement `filter`
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

            # extract parameters for time t
            F = _last_dims(self.F, t, 2)

            # calculate fixed interval smoothing gain
            A = self.xp.dot(self.V_filt[t], self.xp.dot(F.T, linalg.pinv(self.V_pred[t + 1])))
            
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