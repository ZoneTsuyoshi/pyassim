"""
=====================================================================
Inference with Local Linear Operator Construction with Kalman Filter
=====================================================================
This module implements the Local LOCK
for Linear-Gaussian state space models
"""
from logging import getLogger, StreamHandler, DEBUG
logger = getLogger("llock")
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

import os
import math
import time
import multiprocessing as mp
import itertools

import numpy as np

from .utils import array1d, array2d
from .util_functions import _parse_observations, _last_dims, \
    _determine_dimensionality


def _local_calculation(i, j, A, y):
    local_A = A[i] | A[j]
    local_node_number_i = len(np.where(local_A[:i])[0])
    local_node_number_j = len(np.where(local_A[:j])[0])
    global_node_number = np.where(local_A)[0]
    Gh = y[1:, global_node_number].T \
            @ np.linalg.pinv(y[:-1, global_node_number].T)
    return Gh[local_node_number_i, local_node_number_j]


class LocalLOCK(object) :
    """Implements the Local LOCK.
    This class implements the LLOCK,
    for a Linear Gaussian model specified by,
    .. math::
        x_{t+1}   &= F_{t} x_{t} + b_{t} + v_{t} \\
        y_{t}     &= H_{t} x_{t} + d_{t} + w_{t} \\
        [v_{t}, w_{t}]^T &\sim N(0, [[Q_{t}, O], [O, R_{t}]])
    The LLOCK is an algorithm designed to estimate
    :math:`P(x_t | y_{0:t})` and :math:`F` in real-time. 
    As all state transitions and observations are
    linear with Gaussian distributed noise, these distributions can be
    represented exactly as Gaussian distributions with mean
    `x_filt[t]` and covariances `V_filt`.

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
        observation_matrix [n_dim_sys, n_dim_obs] {numpy-array, float}
            also known as :math:`H`. observation matrix from x_{t} to y_{t}
        transition_covariance [n_time-1, n_dim_sys, n_dim_sys]
             or [n_dim_sys, n_dim_sys]
            {numpy-array, float}
            also known as :math:`Q`. system transition covariance
        observation_covariance [n_time, n_dim_obs, n_dim_obs] {numpy-array, float} 
            also known as :math:`R`. observation covariance
        adjacency_matrix [n_dim_sys, n_dim_sys] {numpy-array, float}
            also known as :math:`A`. adjacency matrix, 
            if there is a link between i and j, A[i,j]=1, else A[i,j]=0.
            Besides, you should A[i,i]=1 forall i.
        method {string}
            : method for localized calculation
            "elementwise": calculation for each element of transition matrix
            "local-average": average calculation for specific two observation dimenstions
            "all-average": average calculation for each observation dimenstions
        update_interval {int}
            interval of update transition matrix F
        eta (in (0,1])
            update rate for update transition matrix F
        cutoff
            cutoff distance for update transition matrix F
        save_dir {str, directory-like}
            directory for saving transition matrices and filtered states.
            if this variable is `None`, cannot save them.
        advance_mode {bool}
            if True, calculate transition matrix before filtering.
            if False, calculate the matrix after filtering.
        n_dim_sys {int}
            dimension of system transition variable
        n_dim_obs {int}
            dimension of observation variable
        dtype {type}
            data type of numpy-array
        use_gpu {bool}
            wheather use gpu and cupy.
            if True, you need install package `cupy`.
            if False, set `numpy` for calculation.
        num_cpu {int} or `all`
            number of cpus duaring calculating transition matrix.
            you can set `all` or positive integer.


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
                adjacency_matrix = None, method = "elementwise",
                estimation_length = 10, estimation_interval = 1,
                eta = 1., cutoff = 10., 
                estimation_mode = "backward",
                save_dir = None,
                n_dim_sys = None, n_dim_obs = None, dtype = "float32",
                use_gpu = False, num_cpu = "all"):
        """Setup initial parameters.
        """
        if use_gpu:
            try:
                import cupy
                self.xp = cupy
                self.use_gpu = True
            except:
                self.xp = np
                self.use_gpu = False
        else:
            self.xp = np
            self.use_gpu = False

        # determine dimensionality
        self.n_dim_sys = _determine_dimensionality(
            [(transition_matrix, array2d, -2),
             (initial_mean, array1d, -1),
             (initial_covariance, array2d, -2),
             (observation_matrix, array2d, -1)],
            n_dim_sys,
            self.use_gpu
        )

        self.n_dim_obs = _determine_dimensionality(
            [(observation_matrix, array2d, -2),
             (observation_covariance, array2d, -2),
             (adjacency_matrix, array2d, -2)],
            n_dim_obs,
            self.use_gpu
        )

        # self.y = _parse_observations(observation)
        self.y = self.xp.asarray(observation).copy()

        if initial_mean is None:
            self.initial_mean = self.xp.zeros(self.n_dim_sys, dtype = dtype)
        else:
            self.initial_mean = self.xp.asarray(initial_mean, dtype = dtype)
        
        if initial_covariance is None:
            self.initial_covariance = self.xp.eye(self.n_dim_sys, dtype = dtype)
        else:
            self.initial_covariance = self.xp.asarray(initial_covariance, dtype = dtype)

        if transition_matrix is None:
            self.F = self.xp.eye(self.n_dim_sys, dtype = dtype)
        else:
            self.F = self.xp.asarray(transition_matrix, dtype = dtype)

        if transition_covariance is not None:
            self.Q = self.xp.asarray(transition_covariance, dtype = dtype)
        else:
            self.Q = self.xp.eye(self.n_dim_sys, dtype = dtype)

        if observation_matrix is None:
            self.H = self.xp.eye(self.n_dim_obs, self.n_dim_sys, dtype = dtype)
        else:
            self.H = self.xp.asarray(observation_matrix, dtype = dtype)
        self.HI = self.xp.linalg.pinv(self.H)
        
        if observation_covariance is None:
            self.R = self.xp.eye(self.n_dim_obs, dtype = dtype)
        else:
            self.R = self.xp.asarray(observation_covariance, dtype = dtype)

        if adjacency_matrix is None:
            self.A = self.xp.eye(dtype=bool)
        else:
            self.A = self.xp.asarray(adjacency_matrix, dtype = bool)

        if method in ["elementwise", "local-average", "all-average"]:
            self.method = method
        else:
            raise ValueError("Variable \"method\" only allows \"elementwise\", \"local-average\" "
                + "or \"all-average\". So, your setting \"{}\" need to be changed.".format(method))

        if estimation_mode in ["forward", "middle", "backward"]:
            self.estimation_mode = estimation_mode
        else:
            raise ValueError("\"estimation_mode\" must be choosen from \"forward\","
                            + " \"middle\", or \"backward\".")

        if self.estimation_mode in ["forward", "backward"]:
            self.tau = int(estimation_length)
            self.tau2 = int((estimation_length - 1) / 2)
        else:
            self.tau2 = int((estimation_length - 1) / 2)
            self.tau = 2 * self.tau2 + 1

        self.I = estimation_interval
        self.tm_count = 1

        if save_dir is None:
            self.save_change = False
        else:
            self.save_change = True
            self.save_dir = save_dir
            self.fillnum = len(str(int(self.y.shape[0] / self.I)))
            self.xp.save(os.path.join(self.save_dir, "transition_matrix_" + str(0).zfill(self.fillnum) + ".npy"), self.F)

        if num_cpu == "all":
            self.num_cpu = mp.cpu_count()
        else:
            self.num_cpu = num_cpu

        self.eta = eta
        self.cutoff = cutoff
        self.dtype = dtype
        self.times = self.xp.zeros(5)


    def forward(self):
        """Calculate prediction and filter for observation times.

        Attributes:
            T {int}
                : length of data y
            x_pred [n_time, n_dim_sys] {numpy-array, float}
                : mean of hidden state at time t given observations
                 from times [0...t-1]
            V_pred [n_dim_sys, n_dim_sys] {numpy-array, float}
                : covariance of hidden state at time t given observations
                 from times [0...t-1]
            x_filt [n_time, n_dim_sys] {numpy-array, float}
                : mean of hidden state at time t given observations from times [0...t]
            V_filt [n_dim_sys, n_dim_sys] {numpy-array, float}
                : covariance of hidden state at time t given observations
                 from times [0...t]
        """

        T = self.y.shape[0]
        self.x_pred = self.xp.zeros((T, self.n_dim_sys), dtype = self.dtype)
        self.x_filt = self.xp.zeros((T, self.n_dim_sys), dtype = self.dtype)

        # calculate prediction and filter for every time
        for t in range(T):
            # visualize calculating time
            print("\r filter calculating... t={}".format(t) + "/" + str(T), end="")

            if t == 0:
                # initial setting
                self.x_pred[0] = self.initial_mean
                self.V_pred = self.initial_covariance.copy()
                self._update_transition_matrix(self.tau)
            else:
                if t >= 2 and t < T-self.tau+1 and (t-1)%self.I==0 and self.estimation_mode=="forward":
                    self._update_transition_matrix(t+self.tau-1)
                elif t >= self.tau+1 and (t-self.tau)%self.I==0 and self.estimation_mode=="backward":
                    self._update_transition_matrix(t)
                elif t >= self.tau2+2 and t < T-self.tau2 and (t-self.tau2-1)%self.I==0 and self.estimation_mode=="middle":
                    self._update_transition_matrix(t+self.tau2)
                start_time = time.time()
                self._predict_update(t)
                self.times[0] += time.time() - start_time
            
            if self.xp.any(self.xp.isnan(self.y[t])):
                self.x_filt[t] = self.x_pred[t]
                self.V_filt = self.V_pred
            else :
                start_time = time.time()
                self._filter_update(t)
                self.times[1] += time.time() - start_time

        if self.save_change:
            self.xp.save(os.path.join(self.save_dir, "states.npy"), self.x_filt)


    def _predict_update(self, t):
        """Calculate fileter update

        Args:
            t {int} : observation time
        """
        # extract parameters for time t-1
        Q = _last_dims(self.Q, t - 1, 2, self.use_gpu)

        # calculate predicted distribution for time t
        self.x_pred[t] = self.F @ self.x_filt[t-1]
        self.V_pred = self.F @ self.V_filt @ self.F.T + Q


    def _filter_update(self, t):
        """Calculate fileter update without noise

        Args:
            t {int} : observation time

        Attributes:
            K [n_dim_sys, n_dim_obs] {numpy-array, float}
                : Kalman gain matrix for time t
        """
        # extract parameters for time t
        R = _last_dims(self.R, t, 2, self.use_gpu)

        # calculate filter step
        K = self.V_pred @ (
            self.H.T @ self.xp.linalg.inv(self.H @ (self.V_pred @ self.H.T) + R)
            )
        self.x_filt[t] = self.x_pred[t] + K @ (
            self.y[t] - (self.H @ self.x_pred[t])
            )
        self.V_filt = self.V_pred - K @ (self.H @ self.V_pred)


    def _update_transition_matrix(self, t):
        """Update transition matrix

        Args:
            t {int} : observation time
        """
        G = self.xp.zeros((self.n_dim_obs, self.n_dim_obs), dtype=self.dtype)

        start_time = time.time()

        if self.method=="elementwise": # elementwise
            if self.use_gpu:
                A = self.A.get()
                y = self.y[t-self.tau:t+1].get()
            else:
                A = self.A
                y = self.y[t-self.tau:t+1]
            where_is_A = np.where(A)

            p = mp.Pool(self.num_cpu)
            G_local = p.starmap(_local_calculation, zip(where_is_A[0],
                                                        where_is_A[1],
                                                        itertools.repeat(A),
                                                        itertools.repeat(y)))
            p.close()
            G[A] = G_local
        elif self.method=="local-average": # local-average
            for i in range(self.n_dim_obs):
                local_node_number = len(self.xp.where(self.A[i][:i])[0]) #LA
                global_node_number = self.xp.where(self.A[i])[0]
                Gh = self.y[t-self.tau+1:t+1, global_node_number].T \
                        @ self.xp.linalg.pinv(self.y[t-self.tau:t, global_node_number].T)
                G[i, global_node_number] += Gh[local_node_number] #LA
                G[global_node_number, i] += Gh[:, local_node_number] #LA
            G /= 2.0 #LA
        elif self.method=="all-average": #all-average
            C = self.xp.zeros((self.n_dim_obs, self.n_dim_obs), dtype=self.dtype) #AA
            for i in range(self.n_dim_obs):
                global_node_number = self.xp.where(self.A[i])[0]
                Gh = self.y[t-self.tau+1:t+1, global_node_number].T \
                        @ self.xp.linalg.pinv(self.y[t-self.tau:t, global_node_number].T)
                G[self.xp.ix_(global_node_number, global_node_number)] += Gh #AA
                C[self.xp.ix_(global_node_number, global_node_number)] += 1 #AA
            C[C==0] = 1 #AA
            G /= C #AA

        if self.tm_count==1:
            self.F = self.HI @ G @ self.H
        else:
            self.times[2] += time.time() - start_time
            Fh = self.HI @ G @ self.H
            self.times[3] += time.time() - start_time
            self.times[4] += 1

            self.F = self.F - self.eta * self.xp.minimum(self.xp.maximum(-self.cutoff, self.F - Fh), self.cutoff)

        if self.save_change:
            self.xp.save(os.path.join(self.save_dir, "transition_matrix_" + str(self.tm_count).zfill(self.fillnum) + ".npy"), self.F)
        self.tm_count += 1


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


    # def smooth(self):
    #     """Calculate RTS smooth for times.

    #     Args:
    #         T : length of data y
    #         x_smooth [n_time, n_dim_sys] {numpy-array, float}
    #             : mean of hidden state distributions for times
    #              [0...n_times-1] given all observations
    #         V_smooth [n_time, n_dim_sys, n_dim_sys] {numpy-array, float}
    #             : covariances of hidden state distributions for times
    #              [0...n_times-1] given all observations
    #         A [n_dim_sys, n_dim_sys] {numpy-array, float}
    #             : fixed interval smoothed gain
    #     """

    #     # if not implement `filter`, implement `filter`
    #     try :
    #         self.x_pred[0]
    #     except :
    #         self.filter()

    #     T = self.y.shape[0]
    #     self.x_smooth = self.xp.zeros((T, self.n_dim_sys), dtype = self.dtype)
    #     self.V_smooth = self.xp.zeros((T, self.n_dim_sys, self.n_dim_sys),
    #          dtype = self.dtype)
    #     A = self.xp.zeros((self.n_dim_sys, self.n_dim_sys), dtype = self.dtype)

    #     self.x_smooth[-1] = self.x_filt[-1]
    #     self.V_smooth[-1] = self.V_filt[-1]

    #     # t in [0, T-2] (notice t range is reversed from 1~T)
    #     for t in reversed(range(T - 1)) :
    #         # visualize calculating times
    #         print("\r smooth calculating... t={}".format(T - t)
    #              + "/" + str(T), end="")

    #         # extract parameters for time t
    #         F = _last_dims(self.F, t, 2)

    #         # calculate fixed interval smoothing gain
    #         A = self.xp.dot(self.V_filt[t], self.xp.dot(F.T, self.xp.linalg.pinv(self.V_pred[t + 1])))
            
    #         # fixed interval smoothing
    #         self.x_smooth[t] = self.x_filt[t] \
    #             + self.xp.dot(A, self.x_smooth[t + 1] - self.x_pred[t + 1])
    #         self.V_smooth[t] = self.V_filt[t] \
    #             + self.xp.dot(A, self.xp.dot(self.V_smooth[t + 1] - self.V_pred[t + 1], A.T))

            
    # def get_smoothed_value(self, dim = None):
    #     """Get RTS smoothed value

    #     Args:
    #         dim {int} : dimensionality for extract from RTS smoothed result

    #     Returns (numpy-array, float)
    #         : mean of hidden state at time t given observations
    #         from times [0...T]
    #     """
    #     # if not implement `smooth`, implement `smooth`
    #     try :
    #         self.x_smooth[0]
    #     except :
    #         self.smooth()

    #     if dim is None:
    #         return self.x_smooth
    #     elif dim <= self.x_smooth.shape[1]:
    #         return self.x_smooth[:, int(dim)]
    #     else:
    #         raise ValueError('The dim must be less than '
    #              + self.x_smooth.shape[1] + '.')
