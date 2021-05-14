"""
=============================
Inference with Kalman Filter
=============================
This module implements the Kalman Filter, Kalman Smoother, and
EM Algorithm for Linear-Gaussian state space models
"""

import math
import time

import numpy as np

from .utils import array1d, array2d
from .util_functions import _parse_observations, _last_dims, \
    _determine_dimensionality

# Dimensionality of each Kalman Filter parameter for a single time step
DIM = {
    'transition_matrices': 2,
    'transition_offsets': 1,
    'observation_matrices': 2,
    'observation_offsets': 1,
    'transition_covariance': 2,
    'observation_covariance': 2,
    'initial_mean': 1,
    'initial_covariance': 2,
}


class KalmanFilter(object) :
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
        transition_noise_matrices [n_time - 1, n_dim_sys, n_dim_noise]
            or [n_dim_sys, n_dim_noise] {numpy-array, float}
            also known as :math:`G`. transition noise matrix
        observation_matrices [n_time, n_dim_sys, n_dim_obs] or [n_dim_sys, n_dim_obs]
             {numpy-array, float}
            also known as :math:`H`. observation matrix from x_{t} to y_{t}
        transition_covariance [n_time - 1, n_dim_noise, n_dim_noise]
             or [n_dim_sys, n_dim_noise]
            {numpy-array, float}
            also known as :math:`Q`. system transition covariance for times
        observation_covariance [n_time, n_dim_obs, n_dim_obs] {numpy-array, float} 
            also known as :math:`R`. observation covariance for times.
        transition_offsets [n_time - 1, n_dim_sys] or [n_dim_sys],
            {numpy-array, float} 
            also known as :math:`b`. system offset for times.
        observation_offsets [n_time, n_dim_obs] or [n_dim_obs] {numpy-array, float}
            also known as :math:`d`. observation offset for times.
        transition_observation_covariance [n_time, n_dim_obs, n_dim_sys]
            or [n_dim_obs, n_dim_sys], {numpy-array, float}
            also known as :math:`S`. covariance between system transition
            and observation for times.
        em_vars {list, string}
            variable name list for EM algorithm. subset of ['transition_matrices', \
            'observation_matrices', 'transition_offsets', 'observation_offsets', \
            'transition_covariance', 'observation_covariance', 'initial_mean', \
            'initial_covariance']
        transition_covariance_structure {str}
            covariance structure for system transition. select from ['all', \
            'triD1', 'triD2']. If `all`, optimize all element of transition matrix.
            If `triD1`, optimize tridiagonal element when 1 dimension lattice space
        observation_covariance_structure {str}
            : covariance structure for observation. select from ['all', \
            'triD1', 'triD2']. If `all`, optimize all element of transition matrix.
            If `triD1`, optimize tridiagonal element when 1 dimension lattice space
        transition_vh_length {list or numpy-array, int}
            : if think 2d lattice space, this shows number of vertical lattice
            points and number of horizontal lattice points, of transition
        observation_vh_length {list or numpy-array, int}
            : if think 2d lattice space, this shows number of vertical lattice
            points and number of horizontal lattice points, of observation
        n_dim_sys {int}
            : dimension of system transition variable
        n_dim_obs {int}
            : dimension of observation variable
        dtype {type}
            : data type of numpy-array

    Attributes:
        y : `observation`
        F : `transition_matrices`
        Q : `transition_covariance`. also includes `transition_noise_matrices`
        b : `transition_offsets`
        H : `observation_matrices`
        R : `observation_covariance`
        d : `observation_offsets`
        S : `transition_observation_covariance`
        transition_cs : `transition_covariance_structure`
        observation_cs : `observation_covariance_structure`
        transition_v : `transition_vh_length`
        observation_v : `observation_vh_length`
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
                transition_matrices = None, observation_matrices = None,
                transition_covariance = None, observation_covariance = None,
                transition_noise_matrices = None,
                transition_offsets = None, observation_offsets = None,
                transition_observation_covariance = None,
                em_vars = ['transition_covariance', 'observation_covariance',
                    'initial_mean', 'initial_covariance'],
                transition_covariance_structure = 'all',
                observation_covariance_structure = 'all',
                transition_vh_length = None,
                observation_vh_length = None, 
                n_dim_sys = None, n_dim_obs = None, dtype = "float32",
                use_gpu = False):
        """Setup initial parameters.
        """
        if use_gpu:
            import cupy
            self.xp = cupy
        else:
            self.xp = np

        # determine dimensionality
        self.n_dim_sys = _determine_dimensionality(
            [(transition_matrices, array2d, -2),
             (transition_offsets, array1d, -1),
             (transition_noise_matrices, array2d, -2),
             (initial_mean, array1d, -1),
             (initial_covariance, array2d, -2),
             (observation_matrices, array2d, -1),
             (transition_observation_covariance, array2d, -2)],
            n_dim_sys
        )

        self.n_dim_obs = _determine_dimensionality(
            [(observation_matrices, array2d, -2),
             (observation_offsets, array1d, -1),
             (observation_covariance, array2d, -2),
             (transition_observation_covariance, array2d, -1)],
            n_dim_obs
        )
 
        if transition_noise_matrices is None :
            self.n_dim_noise = _determine_dimensionality(
                    [(transition_covariance, array2d, -2)],
                    self.n_dim_sys
                )
            transition_noise_matrices = self.xp.eye(self.n_dim_noise, dtype = dtype)
        else :
            self.n_dim_noise = _determine_dimensionality(
                    [(transition_noise_matrices, array2d, -1),
                     (transition_covariance, array2d, -2)]
                )

        # self.y = _parse_observations(observation)
        self.y = observation.copy()

        if initial_mean is None:
            self.initial_mean = self.xp.zeros(self.n_dim_sys, dtype = dtype)
        else:
            self.initial_mean = initial_mean.astype(dtype)
        
        if initial_covariance is None:
            self.initial_covariance = self.xp.eye(self.n_dim_sys, dtype = dtype)
        else:
            self.initial_covariance = initial_covariance.astype(dtype)

        if transition_matrices is None:
            self.F = self.xp.eye(self.n_dim_sys, dtype = dtype)
        else:
            self.F = transition_matrices.astype(dtype)

        if transition_covariance is not None:
            if transition_noise_matrices is not None:
                self.Q = self._calc_transition_covariance(
                    transition_noise_matrices,
                    transition_covariance
                    ).astype(dtype)
            else:
                self.Q = transition_covariance.astype(dtype)
        else:
            self.Q = self.xp.eye(self.n_dim_sys, dtype = dtype)

        if transition_offsets is None :
            self.b = self.xp.zeros(self.n_dim_sys, dtype = dtype)
        else :
            self.b = transition_offsets.astype(dtype)

        if observation_matrices is None:
            self.H = self.xp.eye(self.n_dim_obs, self.n_dim_sys, dtype = dtype)
        else:
            self.H = observation_matrices.astype(dtype)
        
        if observation_covariance is None:
            self.R = self.xp.eye(self.n_dim_obs, dtype = dtype)
        else:
            self.R = observation_covariance.astype(dtype)

        if observation_offsets is None :
            self.d = self.xp.zeros(self.n_dim_obs, dtype = dtype)
        else :
            self.d = observation_offsets.astype(dtype)

        if transition_observation_covariance is None:
            self.predict_update = self._predict_update_no_noise
        else:
            self.S = transition_observation_covariance
            self.predict_update = self._predict_update_noise

        self.em_vars = em_vars
        if transition_covariance_structure == 'triD2':
            if transition_vh_length is None:
                raise ValueError('you should iself.xput transition_vh_length.')
            elif transition_vh_length[0] * transition_vh_length[1] != self.n_dim_sys:
                raise ValueError('you should confirm transition_vh_length.')
            else:
                self.transition_v = transition_vh_length[0]
                self.transition_cs = transition_covariance_structure
        elif transition_covariance_structure in ['all', 'triD1']:
            self.transition_cs = transition_covariance_structure
        else:
            raise ValueError('you should confirm transition_covariance_structure.')

        if observation_covariance_structure == 'triD2':
            if observation_vh_length is None:
                raise ValueError('you should iself.xput observation_vh_length.')
            elif observation_vh_length[0]*observation_vh_length[1] != self.n_dim_obs:
                raise ValueError('you should confirm observation_vh_length.')
            else:
                self.observation_v = observation_vh_length[0]
                self.observation_cs = observation_covariance_structure
        elif observation_covariance_structure in ['all', 'triD1']:
            self.observation_cs = observation_covariance_structure
        else:
            raise ValueError('you should confirm observation_covariance_structure.')

        self.dtype = dtype
        self.times = self.xp.zeros(3)


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
                self.predict_update(t)
            
            # If y[t] has any mask, skip filter calculation
            # if (mask and self.xp.any(self.xp.ma.getmask(self.y[t]))) or ((not mask) and self.xp.any(self.xp.isnan(self.y[t]))) :
            if self.xp.any(self.xp.isnan(self.y[t])):
                self.x_filt[t] = self.x_pred[t]
                self.V_filt[t] = self.V_pred[t]
            else :
                # extract parameters for time t
                H = _last_dims(self.H, t, 2)
                R = _last_dims(self.R, t, 2)
                d = _last_dims(self.d, t, 1)

                # calculate filter step
                K = self.V_pred[t] @ (
                    H.T @ self.xp.linalg.pinv(H @ (self.V_pred[t] @ H.T) + R)
                    )
                # print("K: ", K)
                self.x_filt[t] = self.x_pred[t] + K @ (
                    self.y[t] - (H @ self.x_pred[t] + d)
                    )
                self.V_filt[t] = self.V_pred[t] - K @ (H @ self.V_pred[t])
    

    def _predict_update_no_noise(self, t):
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


    def _predict_update_noise(self, t):
        """Calculate fileter update without noise

        Args:
            t {int} : observation time
        """
        if self.xp.any(self.xp.ma.getmask(self.y[t-1])) :
            self._predict_update_no_noise(t)
        else:
            # extract parameters for time t-1
            F = _last_dims(self.F, t - 1, 2)
            Q = _last_dims(self.Q, t - 1, 2)
            b = _last_dims(self.b, t - 1, 1)
            H = _last_dims(self.H, t - 1, 2)
            d = _last_dims(self.d, t - 1, 1)
            S = _last_dims(self.S, t - 1, 2)
            R = _last_dims(self.R, t - 1, 2)

            # calculate predicted distribution for time t
            SR = S @ self.xp.linalg.pinv(R)
            F_SRH = F - SR @ H
            self.x_pred[t] = F_SRH @ self.x_filt[t-1] + b + SR @ (self.y[t-1] - d)
            self.V_pred[t] = F_SRH @ self.V_filt[t-1] @ F_SRH.T + Q - SR @ S.T


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
            A = self.xp.dot(self.V_filt[t], self.xp.dot(F.T, self.xp.linalg.pinv(self.V_pred[t + 1])))
            
            # fixed interval smoothing
            self.x_smooth[t] = self.x_filt[t] \
                + self.xp.dot(A, self.x_smooth[t + 1] - self.x_pred[t + 1])
            self.V_smooth[t] = self.V_filt[t] \
                + self.xp.dot(A, self.xp.dot(self.V_smooth[t + 1] - self.V_pred[t + 1], A.T))


    def fixed_lag_smooth(self, L=5):
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
            print("\r fixed lag smooth calculating... t={}".format(t) + "/" + str(T), end="")

            if t == 0:
                # initial setting
                self.x_pred[0] = self.initial_mean
                self.V_pred[0] = self.initial_covariance
            else:
                self._predict_update_lag(t, L)
            
            # If y[t] has any mask, skip filter calculation
            self.x_filt[t] = self.x_pred[t].copy()
            self.V_filt[t] = self.V_pred[t].copy()
            if not self.xp.any(self.xp.isnan(self.y[t])):
                # extract parameters for time t
                self._filter_update_lag(t, L)


    def _predict_update_lag(self, t, L):
        """Calculate fileter update without noise

        Args:
            t {int} : observation time
        """
        # extract parameters for time t-1
        F = _last_dims(self.F, t - 1, 2)
        Q = _last_dims(self.Q, t - 1, 2)
        b = _last_dims(self.b, t - 1, 1)

        # calculate predicted distribution for time t
        low = max(0, t-L)
        self.x_pred[t] = F @ self.x_filt[t-1] + b
        self.V_pred[t] = F @ self.V_filt[t-1] @ F.T + Q
        self.V_pred[low:t] = self.V_filt[low:t] @ F.T


    def _filter_update_lag(self, t, L):
        H = _last_dims(self.H, t, 2)
        R = _last_dims(self.R, t, 2)
        d = _last_dims(self.d, t, 1)

        # K = self.xp.zeros((L, self.n_dim_sys, self.n_dim_obs), dtype = self.dtype)

        # calculate filter step
        low = max(0, t-L)
        K = self.V_pred[low:t+1] @ (
            H.T @ self.xp.linalg.pinv(H @ (self.V_pred[t] @ H.T) + R)
            )
        self.x_filt[low:t+1] = self.x_filt[low:t+1] + K @ (
            self.y[t] - (H @ self.x_pred[t] + d)
            )
        self.V_filt[low:t+1] = self.V_pred[low:t+1] - K @ (H @ self.V_pred[t])
        # only s=0

            
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


    def em(self, n_iter = 10, em_vars = None, save_name = None, save_states = [], save_em_vars = []):
        """Apply the EM algorithm to estimate all parameters specified by `em_vars`.

        Args:
            n_iter {int}
                : number of EM iterations to perform
            em_vars {list or str}
                : iterable of strings or 'all' variables to perform EM over.
                Any variable not appearing here is left untouched.
        """

        # Create dictionary of variables not to perform EM
        if em_vars is None:
            em_vars = self.em_vars

        if save_name is None:
            save_on = False
        else:
            save_on = True
            zfill_num = len(str(n_iter))
            variable_dict = {'transition_matrices': self.F,
                            'observation_matrices': self.H,
                            'transition_offsets': self.b,
                            'observation_offsets': self.d,
                            'transition_covariance': self.Q,
                            'observation_covariance': self.R,
                            'initial_mean': self.initial_mean,
                            'initial_covariance': self.initial_covariance}

            if em_vars == "all":
                variables = ['transition_matrices',
                            'observation_matrices',
                            'transition_offsets',
                            'observation_offsets',
                            'transition_covariance',
                            'observation_covariance',
                            'initial_mean',
                            'initial_covariance']
            else:
                variables = em_vars.copy()

            for variable in variables:
                if variable in save_em_vars:
                    self.xp.save(save_name + variable + "0".zfill(zfill_num) + ".npy", variable_dict[variable])


        if em_vars == 'all':
            # if `all`, not given known parameters
            given = {}
        else:
            given = {
                'transition_matrices': self.F,
                'observation_matrices': self.H,
                'transition_offsets': self.b,
                'observation_offsets': self.d,
                'transition_covariance': self.Q,
                'observation_covariance': self.R,
                'initial_mean': self.initial_mean,
                'initial_covariance': self.initial_covariance
            }
            # If `em_vars` has elements, remove them from `given`
            em_vars = set(em_vars)
            for k in list(given.keys()):
                if k in em_vars:
                    given.pop(k)

        # Actual EM iterations
        for i in range(n_iter):
            print("EM calculating... i={}".format(i+1) + "/" + str(n_iter), end="")

            # Expectation step
            start_time = time.time()
            self.forward()
            self.times[0] = time.time() - start_time
            
            # system covariance transition between time t and t-1
            start_time = time.time()
            self._sigma_pair_smooth()
            self.times[1] = time.time() - start_time

            # Maximumization step
            start_time = time.time()
            self._calc_em(given = given)
            self.times[2] = time.time() - start_time

            if save_on:
                variable_dict = {'transition_matrices': self.F,
                            'observation_matrices': self.H,
                            'transition_offsets': self.b,
                            'observation_offsets': self.d,
                            'transition_covariance': self.Q,
                            'observation_covariance': self.R,
                            'initial_mean': self.initial_mean,
                            'initial_covariance': self.initial_covariance}
                            
                for variable in variables:
                    if variable in save_em_vars:
                        self.xp.save(save_name + variable + str(i+1).zfill(zfill_num) + ".npy", variable_dict[variable])

                if "predicted" in save_states:
                    self.xp.save(save_name + "predicted" + str(i).zfill(zfill_num) + ".npy", self.x_pred)
                if "filtered" in save_states:
                    self.xp.save(save_name + "filtered" + str(i).zfill(zfill_num) + ".npy", self.x_filt)
                if "smoothed" in save_states:
                    self.xp.save(save_name + "smoothed" + str(i).zfill(zfill_num) + ".npy", self.x_smooth)
        return self


    # calculate transition covariance
    def _calc_transition_covariance(self, G, Q):
        """Calculate transition covariance

        Args:
            G [n_time - 1, n_dim_sys, n_dim_noise] or [n_dim_sys, n_dim_noise]
                {numpy-array, float}
                transition noise matrix
            Q [n_time - 1, n_dim_noise, n_dim_noise] or [n_dim_sys, n_dim_noise]
                {numpy-array, float}
                system transition covariance for times
        """
        if G.ndim == 2:
            GT = G.T
        elif G.ndim == 3:
            GT = G.transpose(0,2,1)
        else:
            raise ValueError('The ndim of transition_noise_matrices'
                + ' should be 2 or 3,' + ' but your iself.xput is ' + str(G.ndim) + '.')
        if Q.ndim == 2 or Q.ndim == 3:
            return self.xp.matmul(G, self.xp.matmul(Q, GT))
        else:
            raise ValueError('The ndim of transition_covariance should be 2 or 3,'
                + ' but your iself.xput is ' + str(Q.ndim) + '.')


    def _sigma_pair_smooth(self):
        """Calculate covariance between hidden states at time t and t-1

        Attributes:
            T {int} : length of y
            V_pair [n_time, n_dim_sys, n_dim_sys] {numpy-array, float}
                : Covariance between hidden states at times t and t-1
                 for t = [1...n_timesteps-1].  Time 0 is ignored.
        """

        T = self.y.shape[0]
        self.x_smooth = self.xp.zeros((T, self.n_dim_sys), dtype = self.dtype)
        self.V_smooth = self.xp.zeros((T, self.n_dim_sys, self.n_dim_sys),
             dtype = self.dtype)

        # pairwise covariance
        self.V_pair = self.xp.zeros((T, self.n_dim_sys, self.n_dim_sys),
             dtype = self.dtype)
        A = self.xp.zeros((self.n_dim_sys, self.n_dim_sys), dtype = self.dtype)

        self.x_smooth[-1] = self.x_filt[-1]
        self.V_smooth[-1] = self.V_filt[-1]

        # t in [0, T-2]
        for t in reversed(range(T - 1)) :
            # visualize calculating time
            print("\r self.expectation step calculating... t={}".format(T - t)
                 + "/" + str(T), end="")

            # extract parameters at time t
            F = _last_dims(self.F, t, 2)

            # calculate fixed interval smoothing gain
            A = self.xp.dot(self.V_filt[t], self.xp.dot(F.T, self.xp.linalg.pinv(self.V_pred[t + 1])))
            
            # fixed interval smoothing
            self.x_smooth[t] = self.x_filt[t] \
                + self.xp.dot(A, self.x_smooth[t + 1] - self.x_pred[t + 1])
            self.V_smooth[t] = self.V_filt[t] \
                + self.xp.dot(A, self.xp.dot(self.V_smooth[t + 1] - self.V_pred[t + 1], A.T))

            # calculate pairwise covariance
            self.V_pair[t + 1] = self.xp.dot(self.V_smooth[t + 1], A.T) # self.V_smooth[t]


    def _calc_em(self, given = {}):
        """Calculate parameters by EM algorithm

        Attributes:
            T {int} : length of observation y
        """

        # length of y
        T = self.y.shape[0]

        # update `observation_matrices`
        if 'observation_matrices' not in given:
            """math
            y_t : observation, d_t : observation_offsets
            x_t : system, H : observation_matrices

            H &= ( \sum_{t=0}^{T-1} (y_t - d_t) \mathbb{E}[x_t]^T )
             ( \sum_{t=0}^{T-1} \mathbb{E}[x_t x_t^T] )^-1
            """
            res1 = self.xp.zeros((self.n_dim_obs, self.n_dim_sys), dtype = self.dtype)
            res2 = self.xp.zeros((self.n_dim_sys, self.n_dim_sys), dtype = self.dtype)

            for t in range(T):
                # if not self.xp.any(self.xp.ma.getmask(self.y[t])):
                if not self.xp.any(self.xp.isnan(self.y[t])):
                    d = _last_dims(self.d, t, 1)
                    res1 += self.xp.outer(self.y[t] - d, self.x_smooth[t])
                    res2 += self.V_smooth[t] \
                        + self.xp.outer(self.x_smooth[t], self.x_smooth[t])

            # update `observation_matrices` or `H`
            self.H = self.xp.dot(res1, self.xp.linalg.pinv(res2))


        # update `observation_covariance`
        if 'observation_covariance' not in given:
            """math
            R : observation_covariance, H_t : observation_matrices,
            x_t : system, d_t : observation_offsets, y_t : observation

            R &= \frac{1}{T} \sum_{t=0}^{T-1}
                [y_t - H_t \mathbb{E}[x_t] - d_t]
                    [y_t - H_t \mathbb{E}[x_t] - d_t]^T
                + H_t Var(x_t) H_t^T
            """

            # y : n_time x n_obs, d : n_obs
            # H : n_obs x n_sys, x_smooth : n_time x n_sys
            # err : n_time x n_obs
            # boolm = ~self.xp.any(self.y.mask, axis=1)
            boolm = self.xp.any(self.xp.isnan(self.y), axis=1)
            err = self.y[boolm] - (self.H @ self.x_smooth[boolm].T).T \
                 - self.d.reshape(1,len(self.d))
            res1 = err.T @ err + self.H @ self.V_smooth[boolm].sum(axis=0) @ self.H.T
            n_obs = boolm.astype(self.xp.int).sum()

            if n_obs > 0:
                self.R = (1.0 / n_obs) * res1
            else:
                self.R = res1

            # divided about `covariance_structure`
            if self.observation_cs == 'triD1':
                # definite new `R`
                new_R = self.xp.zeros_like(self.R, dtype=self.dtype)

                # average diagonal elements
                self.xp.fill_diagonal(new_R, self.R.diagonal().mean())

                # average tridiagonal elements
                rho = (self.R.diagonal(1).mean() + self.R.diagonal(-1).mean()) / 2

                # unify results
                self.R = new_R + self.xp.diag(rho * self.xp.ones(self.n_dim_obs - 1), 1) \
                     + self.xp.diag(rho * self.xp.ones(self.n_dim_obs - 1), -1)
            elif self.observation_cs == 'triD2':
                # definite new `R`
                new_R = self.xp.zeros_like(self.R, dtype=self.dtype)

                # average diagonal elements
                self.xp.fill_diagonal(new_R, self.R.diagonal().mean())

                # average tridiagonal and adjacency elements
                start_time = time.time()
                td = self.xp.ones(self.n_dim_obs - 1)
                td[self.observation_v-1::self.observation_v-1] = 0
                condition = self.xp.diag(td, 1) + self.xp.diag(td, -1) \
                    + self.xp.diag(
                        self.xp.ones(self.n_dim_obs - self.observation_v),
                        self.observation_v
                        ) \
                    + self.xp.diag(
                        self.xp.ones(self.n_dim_obs - self.observation_v),
                        self.observation_v
                        )
                rho = self.R[condition.astype(bool)].mean()

                # unify results
                self.R = new_R + rho * condition.astype(self.dtype)


        # update `transition_matrices`
        if 'transition_matrices' not in given:
            """math
            F : transition_matrices, x_t : system,
            b_t : transition_offsets

            F &= ( \sum_{t=1}^{T-1} \mathbb{E}[x_t x_{t-1}^{T}]
                - b_{t-1} \mathbb{E}[x_{t-1}]^T )
             ( \sum_{t=1}^{T-1} \mathbb{E}[x_{t-1} x_{t-1}^T] )^{-1}
            """
            res1 = self.xp.zeros((self.n_dim_sys, self.n_dim_sys), dtype = self.dtype)
            res2 = self.xp.zeros((self.n_dim_sys, self.n_dim_sys), dtype = self.dtype)
            for t in range(1, T):
                b = _last_dims(self.b, t - 1, 1)
                res1 += self.V_pair[t] + self.xp.outer(
                    self.x_smooth[t], self.x_smooth[t - 1]
                    )
                res1 -= self.xp.outer(b, self.x_smooth[t - 1])            
                res2 += self.V_smooth[t - 1] \
                    + self.xp.outer(self.x_smooth[t - 1], self.x_smooth[t - 1])

            self.F = self.xp.dot(res1, self.xp.linalg.pinv(res2))


        # update `transition_covariance`
        if 'transition_covariance' not in given:
            """math
            Q : transition_covariance, x_t : system, 
            b_t : transition_offsets, F_t : transition_matrices

            Q &= \frac{1}{T-1} \sum_{t=0}^{T-2}
                (\mathbb{E}[x_{t+1}] - A_t \mathbb{E}[x_t] - b_t)
                    (\mathbb{E}[x_{t+1}] - A_t \mathbb{E}[x_t] - b_t)^T
                + F_t Var(x_t) F_t^T + Var(x_{t+1})
                - Cov(x_{t+1}, x_t) F_t^T - F_t Cov(x_t, x_{t+1})
            """

            # x_smooth : n_time x n_sys, F : n_sys x n_sys
            # V_pair, V_smooth : n_time x n_sys x n_sys
            # err : n_time x n_sys, Vt1t_F : n_sys x n_sys
            err = self.x_smooth[1:] \
                - self.x_smooth[:-1] @ self.F.T \
                - self.b.reshape(1,len(self.b))
            Vt1t_F = self.V_pair[1:].sum(axis=0) @ self.F.T
            res1 = err.T @ err \
                + self.F @ (self.V_smooth[:-1] @ self.F.T).sum(axis=0) \
                + self.V_smooth[1:].sum(axis=0) - Vt1t_F - Vt1t_F.T

            self.Q = (1.0 / (T - 1)) * res1

            # devided about `covariance_structure`
            if self.transition_cs == 'triD1':
                # definite new `Q`
                new_Q = self.xp.zeros_like(self.Q, dtype=self.dtype)

                # average diagonal elements
                self.xp.fill_diagonal(new_Q, self.Q.diagonal().mean())

                # average tridiagonal elements
                rho = (self.Q.diagonal(1).mean() + self.Q.diagonal(-1).mean()) / 2

                # unify results
                self.Q = new_Q + self.xp.diag(rho * self.xp.ones(self.n_dim_sys - 1), 1)\
                     + self.xp.diag(rho * self.xp.ones(self.n_dim_sys - 1), -1)
            elif self.transition_cs == 'triD2':
                # definite new `R`
                new_Q = self.xp.zeros_like(self.Q, dtype=self.dtype)

                # average diagonal elements
                self.xp.fill_diagonal(new_Q, self.Q.diagonal().mean())

                # average tridiagonal and adjacency elements
                td = self.xp.ones(self.n_dim_sys - 1)
                td[self.transition_v-1::self.transition_v-1] = 0
                condition = self.xp.diag(td, 1) + self.xp.diag(td, -1) \
                    + self.xp.diag(
                        self.xp.ones(self.n_dim_sys - self.transition_v),
                        self.transition_v
                        ) \
                    + self.xp.diag(
                        self.xp.ones(self.n_dim_sys - self.transition_v),
                        self.transition_v
                        )
                rho = self.Q[condition.astype(bool)].mean()

                # unify results
                self.Q = new_Q + rho * condition.astype(self.dtype)

        # update `initial_mean`
        if 'initial_mean' not in  given:
            """math
            x_0 : system of t=0
                \mu_0 = \mathbb{E}[x_0]
            """
            tmp = self.initial_mean
            self.initial_mean = self.x_smooth[0]


        # update `initial_covariance`
        if 'initial_covariance' not in given:
            """math
            mu_0 : system of t=0
                \Sigma_0 = \mathbb{E}[x_0, x_0^T] - \mu_0 \mu_0^T
            """
            x0 = self.x_smooth[0]
            x0_x0 = self.V_smooth[0] + self.xp.outer(x0, x0)

            self.initial_covariance = x0_x0 - self.xp.outer(self.initial_mean, x0)
            self.initial_covariance += - self.xp.outer(x0, self.initial_mean)\
                 + self.xp.outer(self.initial_mean, self.initial_mean)


        # update `transition_offsets`
        if 'transition_offsets' not in given:
            """math
            b : transition_offsets, x_t : system
            F_t : transition_matrices
                b = \frac{1}{T-1} \sum_{t=1}^{T-1}
                        \mathbb{E}[x_t] - F_{t-1} \mathbb{E}[x_{t-1}]
            """
            self.b = self.xp.zeros(self.n_dim_sys, dtype = self.dtype)

            if T > 1:
                for t in range(1, T):
                    F = _last_dims(self.F, t - 1)
                    self.b += self.x_smooth[t] - self.xp.dot(F, self.x_smooth[t - 1])
                self.b *= (1.0 / (T - 1))


        # update `observation_offsets`
        if 'observation_offsets' not in given:
            """math
            d : observation_offsets, y_t : observation
            H_t : observation_matrices, x_t : system
                d = \frac{1}{T} \sum_{t=0}^{T-1} y_t - H_{t} \mathbb{E}[x_{t}]
            """
            self.d = self.xp.zeros(self.n_dim_obs, dtype = self.dtype)
            n_obs = 0
            for t in range(T):
                # if not self.xp.any(self.xp.ma.getmask(self.y[t])):
                if not self.xp.any(self.xp.isnan(self.y[t])):
                    H = _last_dims(self.H, t)
                    self.d += self.y[t] - self.xp.dot(H, self.x_smooth[t])
                    n_obs += 1
            if n_obs > 0:
                self.d *= (1.0 / n_obs)

