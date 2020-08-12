"""
=============================
Inference with Kalman Filter
=============================
This module implements the Kalman Filter, Kalman Smoother, and
EM Algorithm for Linear-Gaussian state space models

This code is inference for real-time calculation of Kalman Filter.
"""

import time
import math
import os
import itertools

import numpy as np
# try:
#     import cupy
#     xp = cupy
#     from cupy import linalg
# except:
#     xp = np
#     from numpy import linalg
    # from scipy import sparse
# from scipy import linalg

from utils import array1d, array2d
from util_functions import _determine_dimensionality



class ParticleKalmanFilterRealtime(object) :
    """Implements the Kalman Filter, Kalman Smoother, and EM algorithm.
    This class implements the Kalman Filter, Kalman Smoother, and EM Algorithm
    for a Linear Gaussian model specified by,
    .. math::
        x_{t+1}   &= F_{t} x_{t} + b_{t} + G_{t} v_{t} \\
        y_{t}     &= H_{t} x_{t} + d_{t} + w_{t} \\
        [v_{t}, w_{t}]^T &\sim N(0, [[Q_{t}, O], [O, R_{t}]])
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
        initial_mean [n_dim_sys] {float} 
            : also known as :math:`\mu_0`. initial state mean
            初期状態分布の期待値[状態変数軸]
        initial_covariance [n_dim_sys, n_dim_sys] {numpy-array, float} 
            : also known as :math:`\Sigma_0`. initial state covariance
            初期状態分布の共分散行列[状態変数軸，状態変数軸]
        transition_matrix [n_dim_sys, n_dim_sys]{numpy-array, float}
            : also known as :math:`F`. transition matrix from x_{t-1} to x_{t}
            システムモデルの変換行列[状態変数軸，状態変数軸]
        observation_matrix [n_dim_sys, n_dim_obs]
             {numpy-array, float}
            : also known as :math:`H`. observation matrix from x_{t} to y_{t}
            観測行列 [状態変数軸，観測変数軸]
        transition_covariance [n_dim_sys, n_dim_noise]
            {numpy-array, float}
            : also known as :math:`Q`. system transition covariance for times
            システムノイズの共分散行列[ノイズ変数軸，ノイズ変数軸]
        observation_covariance [n_time, n_dim_obs, n_dim_obs] {numpy-array, float} 
            : also known as :math:`R`. observation covariance for times.
            観測ノイズの共分散行列[観測変数軸，観測変数軸]
        transition_offset [n_dim_sys], {numpy-array, float} 
            : also known as :math:`b`. system offset for times.
            システムモデルの切片（バイアス，オフセット） [状態変数軸]
        observation_offset [n_dim_obs] {numpy-array, float}
            : also known as :math:`d`. observation offset for times.
            観測モデルの切片 [観測変数軸]
        n_dim_sys {int}
            : dimension of system transition variable
            システム変数の次元
        n_dim_obs {int}
            : dimension of observation variable
            観測変数の次元
        dtype {type}
            : data type of numpy-array
            numpy のデータ形式
        save_path {str, path-like}
            : directory for save state and covariance.
        em_vars {list, string}
            : variable name list for EM algorithm. subset of ['transition_matrices', \
            'observation_matrices', 'transition_offsets', 'observation_offsets', \
            'transition_covariance', 'observation_covariance', 'initial_mean', \
            'initial_covariance']
            EMアルゴリズムで最適化する変数リスト
        em_mode {boolean}
            : if true, set EM mode.

    Attributes:
        F : `transition_matrix`
        Q : `transition_covariance`
        b : `transition_offset`
        H : `observation_matrix`
        R : `observation_covariance`
        d : `observation_offset`
        x [n_dim_sys] {numpy-array, float} 
            : mean of predicted or filtered distribution
            予測，フィルタ分布の平均 [時間軸，状態変数軸]
        V [n_dim_sys, n_dim_sys] {numpy-array, float}
            : covariance of predicted or filtered distribution
            予測，フィルタ分布の共分散行列 [時間軸，状態変数軸，状態変数軸]
    """

    def __init__(self, initial_mean = None, initial_covariance = None,
                transition_matrices = None, observation_matrix = None,
                transition_covariance = None, observation_covariance = None,
                transition_offset = None, observation_offset = None,
                n_dim_sys = None, n_dim_obs = None, lag = 5,
                dtype = "float32",
                save_path = None, mode = "filter",
                cpu_number = "all",
                use_gpu = True):
        """Setup initial parameters.
        """
        self.use_gpu = use_gpu
        if use_gpu:
            import cupy
            self.xp = cupy
            # cupy.cuda.set_allocator(cupy.cuda.MemoryPool().malloc)
            # from cupy import linalg
        else:
            self.xp = np
            # from numpy import linalg

        # determine dimensionality
        self.n_dim_sys = _determine_dimensionality(
            [(transition_matrices, array2d, -2),
             (transition_offset, array1d, -1),
             (transition_covariance, array2d, -2),
             (initial_mean, array1d, -1),
             (initial_covariance, array2d, -2),
             (observation_matrix, array2d, -1)],
            n_dim_sys
        )

        self.n_dim_obs = _determine_dimensionality(
            [(observation_matrix, array2d, -2),
             (observation_offset, array1d, -1),
             (observation_covariance, array2d, -2)],
            n_dim_obs
        )

        self.lag_on = False
        if mode=="filter":
            self.forward_update = self._forward_update
        elif mode=="lag_smooth":
            self.forward_update = self._fixed_lag_smooth
            self.L = int(lag)
            self.lag_on = True

        if self.lag_on:
            if initial_mean is None:
                self.x = self.xp.zeros((self.L+1, self.n_dim_sys), dtype = dtype)
            else:
                self.x = self.xp.zeros((self.L+1, self.n_dim_sys), dtype = dtype)
                self.x[:] = self.xp.asarray(initial_mean, dtype = dtype)
        else:
            if initial_mean is None:
                self.x = self.xp.zeros(self.n_dim_sys, dtype = dtype)
            else:
                self.x = self.xp.asarray(initial_mean, dtype = dtype)
        
        if self.lag_on:
            if initial_covariance is None:
                self.V = self.xp.zeros((self.L+1, self.n_dim_sys, self.n_dim_sys), dtype = dtype)
                self.V[:] = self.xp.eye(self.n_dim_sys, dtype = dtype)
            else:
                self.V = self.xp.zeros((self.L+1, self.n_dim_sys, self.n_dim_sys), dtype = dtype)
                self.V[:] = self.xp.asarray(initial_covariance, dtype = dtype)
        else:
            if initial_covariance is None:
                self.V = self.xp.eye(self.n_dim_sys, dtype = dtype)
            else:
                self.V = self.xp.asarray(initial_covariance, dtype = dtype)

        if transition_matrices is None:
            # self.F = sparse.eye(self.n_dim_sys, dtype = dtype)
            self.F = self.xp.asarray([
                    self.xp.eye(self.n_dim_sys, dtype = dtype),
                    self.xp.ones((self.n_dim_sys, self.n_dim_sys), dtype = dtype)
                ])
        else:
            self.F = self.xp.asarray(transition_matrices, dtype = dtype)
        self.n_candidate = len(self.F)

        if transition_covariance is None:
            # self.Q = sparse.eye(self.n_dim_sys, dtype = dtype)
            self.Q = self.xp.eye(self.n_dim_sys, dtype = dtype)
        else:
            self.Q = self.xp.asarray(transition_covariance, dtype = dtype)

        if transition_offset is None :
            self.b = self.xp.zeros(self.n_dim_sys, dtype = dtype)
        else :
            self.b = self.xp.asarray(transition_offset, dtype = dtype)

        if observation_matrix is None:
            self.H = self.xp.eye(self.n_dim_obs, self.n_dim_sys, dtype = dtype)
        else:
            self.H = self.xp.asarray(observation_matrix, dtype = dtype)
        
        if observation_covariance is None:
            self.R = self.xp.eye(self.n_dim_obs, dtype = dtype)
        else:
            self.R = self.xp.asarray(observation_covariance, dtype = dtype)
        self.R_inv = self.xp.linalg.pinv(self.R)

        if observation_offset is None:
            self.d = self.xp.zeros(self.n_dim_obs, dtype = dtype)
        else :
            self.d = self.xp.asarray(observation_offset, dtype = dtype)

        self.dtype = dtype
        self.times = self.xp.zeros(3)

        if save_path is not None:
            self.save_path = save_path
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)

        # if cpu_number == "all":
        #     self.cpu_number = multi.cpu_count()
        # else:
        #     self.cpu_number = cpu_number



    def _forward_update(self, t, y, weighted=False, F=None, H=None, Q=None, R=None, b=None, d=None, on_save=False,
                    save_path=None, fillnum = 3):
        """Calculate prediction and filter regarding arguments.

        Args:
            t {int}
                : time for calculating.
            y [n_dim_obs] {numpy-array, float}
                : also known as :math:`y`. observation value
                観測値[観測変数軸]
            F [n_dim_sys, n_dim_sys]{numpy-array, float}
                : also known as :math:`F`. transition matrix from x_{t-1} to x_{t}
                システムモデルの変換行列[状態変数軸，状態変数軸]
            H [n_dim_sys, n_dim_obs] {numpy-array, float}
                : also known as :math:`H`. observation matrix from x_{t} to y_{t}
                観測行列 [状態変数軸，観測変数軸]
            Q [n_dim_sys, n_dim_noise] {numpy-array, float}
                : also known as :math:`Q`. system transition covariance for times
                システムノイズの共分散行列[ノイズ変数軸，ノイズ変数軸]
            R [n_time, n_dim_obs, n_dim_obs] {numpy-array, float} 
                : also known as :math:`R`. observation covariance for times.
                観測ノイズの共分散行列[観測変数軸，観測変数軸]
            b [n_dim_sys], {numpy-array, float} 
                : also known as :math:`b`. system offset for times.
                システムモデルの切片（バイアス，オフセット） [状態変数軸]
            d [n_dim_obs] {numpy-array, float}
                : also known as :math:`d`. observation offset for times.
                観測モデルの切片 [観測変数軸]
            on_save {boolean}
                : if true, save state x and covariance V.
            fillnum {int}
                : number of filling for zfill.

        Attributes:
            K [n_dim_sys, n_dim_obs] {numpy-array, float}
                : Kalman gain matrix for time t [状態変数軸，観測変数軸]
                カルマンゲイン
        """

        if F is None:
            F = self.F
        if H is None:
            H = self.H
        if Q is None:
            Q = self.Q
        if R is None:
            R = self.R
            R_inv = self.R_inv
        else:
            R_inv = self.xp.linalg.pinv(R)
        if b is None:
            b = self.b
        if d is None:
            d = self.d
        if save_path is None and on_save:
            save_path = self.save_path
        # tnums = self.xp.zeros(6)
        # tnums[0] = time.time()

        # calculate predicted distribution for time t
        selected_index = 0
        if t != 0:
            self.x = F @ self.x + b # n_candidate x n_dim_sys
            innovation = (y - ((H @ self.x.T).T + d)) # n_candidate x n_dim_obs
            
            if weighted:
                weight = -2 * self.xp.diag(innovation @ R_inv @ innovation.T)
                weight = weight - self.xp.max(weight)
                weight = self.xp.exp(weight)
                selected_index = weight / weight.sum()
                F_selected = self.xp.moveaxis(F, 0, -1) @ selected_index
                self.x = self.x.T @ selected_index
            else:
                selected_index = self.xp.argmin(self.xp.diag(innovation @ R_inv @ innovation.T))
                F_selected = F[selected_index]
                self.x = self.x[selected_index] # n_dim_sys

            self.V = F_selected @ self.V @ F_selected.T + Q

            if on_save:
                self.xp.save(os.path.join(save_path, "predictive_mean_"
                                                    + str(t).zfill(fillnum) + ".npy"),
                        self.x)
                self.xp.save(os.path.join(save_path, "predictive_covariance_"
                                                    + str(t).zfill(fillnum) + ".npy"),
                        self.V)

        # calculate filter step
        K = self.V @ ( H.T @ self.xp.linalg.inv(H @ (self.V @ H.T) + R) )
        self.x = self.x + K @ ( y - (H @ self.x + d) )
        self.V = self.V - K @ (H @ self.V)
        # tnums[2] = time.time()

        if on_save:
            self.xp.save(os.path.join(save_path, "filtered_mean_"
                                            + str(t).zfill(fillnum) + ".npy"),
                    self.x)
            self.xp.save(os.path.join(save_path, "filtered_covariance_"
                                            + str(t).zfill(fillnum) + ".npy"),
                    self.V)

        return self.x, selected_index


    def _fixed_lag_smooth(self, t, y, F=None, H=None, Q=None, R=None, b=None, d=None, on_save=False,
                    save_path=None, fillnum = 3, return_on=False):
        """Calculate prediction and filter regarding arguments.

        Args:
            t {int}
                : time for calculating.
            y [n_dim_obs] {numpy-array, float}
                : also known as :math:`y`. observation value
                観測値[観測変数軸]
            F [n_dim_sys, n_dim_sys]{numpy-array, float}
                : also known as :math:`F`. transition matrix from x_{t-1} to x_{t}
                システムモデルの変換行列[状態変数軸，状態変数軸]
            H [n_dim_sys, n_dim_obs] {numpy-array, float}
                : also known as :math:`H`. observation matrix from x_{t} to y_{t}
                観測行列 [状態変数軸，観測変数軸]
            Q [n_dim_sys, n_dim_noise] {numpy-array, float}
                : also known as :math:`Q`. system transition covariance for times
                システムノイズの共分散行列[ノイズ変数軸，ノイズ変数軸]
            R [n_time, n_dim_obs, n_dim_obs] {numpy-array, float} 
                : also known as :math:`R`. observation covariance for times.
                観測ノイズの共分散行列[観測変数軸，観測変数軸]
            b [n_dim_sys], {numpy-array, float} 
                : also known as :math:`b`. system offset for times.
                システムモデルの切片（バイアス，オフセット） [状態変数軸]
            d [n_dim_obs] {numpy-array, float}
                : also known as :math:`d`. observation offset for times.
                観測モデルの切片 [観測変数軸]
            on_save {boolean}
                : if true, save state x and covariance V.
            fillnum {int}
                : number of filling for zfill.

        Attributes:
            K [n_dim_sys, n_dim_obs] {numpy-array, float}
                : Kalman gain matrix for time t [状態変数軸，観測変数軸]
                カルマンゲイン
        """

        if F is None:
            F = self.F
        if H is None:
            H = self.H
        if Q is None:
            Q = self.Q
        if R is None:
            R = self.R
        if b is None:
            b = self.b
        if d is None:
            d = self.d
        if save_path is None and on_save:
            save_path = self.save_path
        # tnums = self.xp.zeros(6)
        # tnums[0] = time.time()

        # calculate predicted distribution for time t
        low = max(t-self.L, 0)
        if t != 0:
            self.x[-1] = F @ self.x[-2] + b
            self.V[-1] = F @ self.V[-2] @ F.T + Q
            self.V[:-1] = self.V[:-1] @ F.T

            if on_save:
                self.xp.save(os.path.join(save_path, "predictive_mean_"
                                                    + str(t).zfill(fillnum) + ".npy"),
                        self.x)
                self.xp.save(os.path.join(save_path, "predictive_covariance_"
                                                    + str(t).zfill(fillnum) + ".npy"),
                        self.V)

        # tnums[1] = time.time()

        # print("predicted time : {}".format(t2-t1))
            
        # If y[t] has any mask, skip filter calculation
        # if not self.xp.any(self.xp.isnan(y)):
        # calculate filter step
        K = self.V @ ( H.T @ self.xp.linalg.inv(H @ (self.V[-1] @ H.T) + R) )
        self.x = self.x + K @ ( y - (H @ self.x[-1] + d) )
        self.V = self.V - K @ (H @ self.V[-1])
        # tnums[2] = time.time()

        if on_save:
            self.xp.save(os.path.join(save_path, "filtered_mean_"
                                            + str(t).zfill(fillnum) + ".npy"),
                    self.x)
            self.xp.save(os.path.join(save_path, "filtered_covariance_"
                                            + str(t).zfill(fillnum) + ".npy"),
                    self.V)

        # for i in range(2):
        #     self.times[i] += tnums[i+1] - tnums[i]
        # print("filtered time : {}".format(t3-t2))
        if return_on:
            return self.x