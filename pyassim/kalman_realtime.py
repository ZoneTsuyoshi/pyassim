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

from multiprocessing import Pool
import multiprocessing as multi

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



def global_to_local(adjacency_matrix, global_variable):
    """Map global to local

    Args:
        adjacency_matrix [n_dim] {numpy-array, float}
            : adjacency matrix from global to local
            大域変数を局所変数に写像するための隣接行列
        global_variable [n_dim] {numpy-array, float}
            : global variable
            大域変数
    """
    return global_variable[adjacency_matrix][:, adjacency_matrix]


def local_multi_processing(resf1, resf2, i, adjacency_matrix):
    """Parallel processing for local calculation
    """
    xp = cupy.get_array_module(adjacency_matrix)
    result = xp.zeros((len(adjacency_matrix), len(adjacency_matrix))).astype(resf1.dtype)
    result[adjacency_matrix][:, adjacency_matrix] = resf1 @ xp.linalg.pinv(resf2)
    return result


class KalmanFilterRealtime(object) :
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
                transition_matrix = None, observation_matrix = None,
                transition_covariance = None, observation_covariance = None,
                transition_offset = None, observation_offset = None,
                adjacency_matrix = None,
                n_dim_sys = None, n_dim_obs = None, lag = 5,
                dtype = "float32",
                save_path = None, mode = "filter",
                em_vars = ["transition_matrices", "observation_matrices",
                    "transition_covariance", "observation_covariance",
                    "initial_mean", "initial_covariance"],
                em_mode = False, cpu_number = "all",
                use_gpu = True):
        """Setup initial parameters.
        """
        self.use_gpu = use_gpu
        if use_gpu:
            import cupy
            self.xp = cupy
            self.xp_type = "cupy"
            # cupy.cuda.set_allocator(cupy.cuda.MemoryPool().malloc)
            # from cupy import linalg
        else:
            self.xp = np
            self.xp_type = "numpy"
            # from numpy import linalg

        # determine dimensionality
        self.n_dim_sys = _determine_dimensionality(
            [(transition_matrix, array2d, -2),
             (transition_offset, array1d, -1),
             (transition_covariance, array2d, -2),
             (initial_mean, array1d, -1),
             (initial_covariance, array2d, -2),
             (observation_matrix, array2d, -1)],
            n_dim_sys,
            self.xp_type
        )

        self.n_dim_obs = _determine_dimensionality(
            [(observation_matrix, array2d, -2),
             (observation_offset, array1d, -1),
             (observation_covariance, array2d, -2)],
            n_dim_obs,
            self.xp_type
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

        if transition_matrix is None:
            # self.F = sparse.eye(self.n_dim_sys, dtype = dtype)
            self.F = self.xp.eye(self.n_dim_sys, dtype = dtype)
        else:
            self.F = self.xp.asarray(transition_matrix, dtype = dtype)

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

        if observation_offset is None:
            self.d = self.xp.zeros(self.n_dim_obs, dtype = dtype)
        else :
            self.d = self.xp.asarray(observation_offset, dtype = dtype)

        if adjacency_matrix is None:
            # if adjacency matrix is None, then not localization mode.
            self.lmode_on = False
        else:
            self.A = self.xp.asarray(adjacency_matrix, dtype = bool)
            self.lmode_on = True

        self.dtype = dtype
        self.times = self.xp.zeros(3)

        if save_path is not None:
            self.save_path = save_path
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)

        self.em_mode = em_mode
        if em_mode:
            self.setup_em_calculation(em_vars)

        if cpu_number == "all":
            self.cpu_number = multi.cpu_count()
        else:
            self.cpu_number = cpu_number



    def _forward_update(self, t, y, F=None, H=None, Q=None, R=None, b=None, d=None, on_save=False,
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
        if t != 0:
            self.x = F @ self.x + b
            self.V = F @ self.V @ F.T + Q

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
        # print(H.shape, self.V.shape, R.shape)
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

        # for i in range(2):
        #     self.times[i] += tnums[i+1] - tnums[i]
        # print("filtered time : {}".format(t3-t2))
        if return_on:
            return self.x


    def forward_update_v1_t0(self, y, H):
        K = self.V @ ( H.T @ self.xp.linalg.inv(H @ (self.V @ H.T) + self.R) )
        self.x = self.x + K @ ( y - (H @ self.x + self.d) )
        self.V = self.V - K @ (H @ self.V)
        return self.x


    def forward_update_v1(self, y, H=None):
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
        # calculate predicted distribution for time t
        self.x = self.F @ self.x + self.b
        self.V = self.F @ self.V @ self.F.T + self.Q

        K = self.V @ ( H.T @ self.xp.linalg.inv(H @ (self.V @ H.T) + self.R) )
        self.x = self.x + K @ ( y - (H @ self.x + self.d) )
        self.V = self.V - K @ (H @ self.V)
        return self.x


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



    def backward_update(self, t, T, y, F=None, on_save=False, save_path=None, fillnum = 3):
        """Calculate smoothing regarding arguments.

        Args:
            t {int}
                : time for calculating.
            T {int}
                : last time for forward step.
            y [n_dim_obs] {numpy-array, float}
                : also known as :math:`y`. observation value
                観測値[観測変数軸]
            F [n_dim_sys, n_dim_sys]{numpy-array, float}
                : also known as :math:`F`. transition matrix from x_{t-1} to x_{t}
                システムモデルの変換行列[状態変数軸，状態変数軸]
            on_save {boolean}
                : if true, save state x and covariance V.
            fillnum {int}
                : number of filling for zfill.
            em_mode {boolean}
                : if true, set EM mode, i.e., calculate pairwise covariance and EM values.

        Attributes:
            A [n_dim_sys, n_dim_obs] {numpy-array, float}
                : fixed-interval smooth gain matrix for time t [状態変数軸，観測変数軸]
                固定区間平滑化ゲイン
        """

        if F is None:
            F = self.F
        if save_path is None:
            save_path = self.save_path

        if t==T-2 and on_save:
            self.xp.save(os.path.join(save_path, "smoothed_mean_"
                                            + str(t+1).zfill(fillnum) + ".npy"),
                    self.x)
            self.xp.save(os.path.join(save_path, "smoothed_covariance_"
                                            + str(t+1).zfill(fillnum) + ".npy"),
                    self.V)

        # load forward result
        start_time = time.time()
        x_pred = self.xp.load(os.path.join(save_path, "predictive_mean_"
                                                + str(t+1).zfill(fillnum) + ".npy"))
        V_pred = self.xp.load(os.path.join(save_path, "predictive_covariance_"
                                                + str(t+1).zfill(fillnum) + ".npy"))
        x_filt = self.xp.load(os.path.join(save_path, "filtered_mean_"
                                                + str(t).zfill(fillnum) + ".npy"))
        V_filt = self.xp.load(os.path.join(save_path, "filtered_covariance_"
                                                + str(t).zfill(fillnum) + ".npy"))
        self.times[0] += time.time() - start_time

        # calculate smoothed value
        start_time = time.time()
        A = V_filt @ (F.T @ self.xp.linalg.inv(V_pred)) # pinv
        self.x = x_filt + A @ (self.x - x_pred)
        self.V = V_filt + A @ (self.V - V_pred) @ A.T

        if self.em_mode:
            V_pair = self.V @ A.T
        self.times[1] += time.time() - start_time

        start_time = time.time()
        if on_save:
            self.xp.save(os.path.join(save_path, "smoothed_mean_"
                                            + str(t).zfill(fillnum) + ".npy"),
                    self.x)
            self.xp.save(os.path.join(save_path, "smoothed_covariance_"
                                            + str(t).zfill(fillnum) + ".npy"),
                    self.V)
            if self.em_mode:
                self.xp.save(os.path.join(save_path, "pairwise_covariance_"
                                            + str(t).zfill(fillnum) + ".npy"),
                        V_pair)
        self.times[2] += time.time() - start_time



    def setup_em_calculation(self, em_vars):
        """Setup for eself.xpectation maximization calculation.

        Args:
            em_vars {list, string}
                : variable name list for EM algorithm. subset of ['transition_matrices', \
                'observation_matrices', 'transition_offsets', 'observation_offsets', \
                'transition_covariance', 'observation_covariance', 'initial_mean', \
                'initial_covariance']
                EMアルゴリズムで最適化する変数リスト
        """
        self.em_vars = []
        if "F" in em_vars or "transition_matrix" in em_vars or "transition_matrices" in em_vars:
            self.em_vars.append("F")
            self.resf1 = self.xp.zeros((self.n_dim_sys, self.n_dim_sys), dtype = self.dtype)
            self.resf2 = self.xp.zeros((self.n_dim_sys, self.n_dim_sys), dtype = self.dtype)
        if "H" in em_vars or "observation_matrix" in em_vars or "observation_matrices" in em_vars:
            self.em_vars.append("H")
            self.resh1 = self.xp.zeros((self.n_dim_obs, self.n_dim_sys), dtype = self.dtype)
            self.resh2 = self.xp.zeros((self.n_dim_sys, self.n_dim_sys), dtype = self.dtype)
        if "Q" in em_vars or "transition_covariance" in em_vars:
            self.em_vars.append("Q")
            self.resq = self.xp.zeros((self.n_dim_sys, self.n_dim_sys), dtype = self.dtype)
        if "R" in em_vars or "observation_covariance" in em_vars:
            self.em_vars.append("R")
            self.resr = self.xp.zeros((self.n_dim_sys, self.n_dim_sys), dtype = self.dtype)
        if "x0" in em_vars or "initial_mean" in em_vars:
            self.em_vars.append("x0")
        if "V0" in em_vars or "initial_covariance" in em_vars:
            self.em_vars.append("V0")
        if "b" in em_vars or "transition_offset" in em_vars or "transition_offsets" in em_vars:
            self.em_vars.append("b")
            self.resb = self.xp.zeros(self.n_dim_sys, dtype = self.dtype)
        if "d" in em_vars or "observation_offset" in em_vars or "observation_offsets" in em_vars:
            self.em_vars.append("d")
            self.resd = self.xp.zeros(self.n_dim_obs, dtype = self.dtype)


    def em_forward_FH(self, t, y, b=None, d=None, save_path=None, fillnum=3):
        """Calculate maximization step for EM algorithm.

        Args:
            t {int}
                : time for calculating.
            y [n_dim_obs] {numpy-array, float}
                : also known as :math:`y`. observation value
                観測値[観測変数軸]
            b [n_dim_sys], {numpy-array, float} 
                : also known as :math:`b`. system offset for times.
                システムモデルの切片（バイアス，オフセット） [状態変数軸]
            d [n_dim_obs] {numpy-array, float}
                : also known as :math:`d`. observation offset for times.
                観測モデルの切片 [観測変数軸]
        """
        if b is None:
            b = self.b
        if d is None:
            d = self.d
        if save_path is None:
            save_path = self.save_path

        # load backward result
        x_smooth = self.xp.load(os.path.join(save_path, "smoothed_mean_"
                                        + str(t).zfill(fillnum) + ".npy"))
        
        if "H" in self.em_vars:
            V_smooth = self.xp.load(os.path.join(save_path, "smoothed_covariance_"
                                            + str(t).zfill(fillnum) + ".npy"))
            self.resh1 += self.xp.outer(y - d, x_smooth)
            self.resh2 += V_smooth + self.xp.outer(x_smooth, x_smooth)

        if "F" in self.em_vars:
            if t!=0:
                x_smooth_old = self.xp.load(os.path.join(save_path, "smoothed_mean_"
                                                    + str(t-1).zfill(fillnum) + ".npy"))
                V_pair = self.xp.load(os.path.join(save_path, "pairwise_covariance_"
                                                    + str(t-1).zfill(fillnum) + ".npy"))
                V_smooth = self.xp.load(os.path.join(save_path, "smoothed_covariance_"
                                                    + str(t-1).zfill(fillnum) + ".npy"))
                self.resf1 += V_pair + self.xp.outer(x_smooth, x_smooth_old) - self.xp.outer(b, x_smooth_old)
                self.resf2 += V_smooth + self.xp.outer(x_smooth_old, x_smooth_old)


    def em_update_FH(self):
        """
        """
        if "H" in self.em_vars:
            self.H = self.resh1 @ self.xp.linalg.pinv(self.resh2)
        
        if "F" in self.em_vars:
            if self.lmode_on and self.use_gpu:
                self.F = self.xp.zeros((self.n_dim_sys, self.n_dim_sys), dtype = self.dtype)
                # resf1_local = self._parallel_global_to_local(self.resf1, self.n_dim_sys, self.A)
                # resf2_local = self._parallel_global_to_local(self.resf2, self.n_dim_sys, self.A)

                # # local processing
                # p = Pool(self.cpu_number)
                # self.F += p.starmap(local_multi_processing, zip(resf1_local, resf2_local, 
                #     range(self.n_dim_sys), self.A))
                # p.close()

                for i in range(self.n_dim_sys):
                    # self.F[self.xp.ix_(self.A[i], self.A[i])] = self.resf1[self.xp.ix_(self.A[i],self.A[i])] \
                    #                 @ self.xp.linalg.pinv(self.resf2[self.xp.ix_(self.A[i], self.A[i])])
                    Fh = self.resf1[self.xp.ix_(self.A[i],self.A[i])] \
                         @ self.xp.linalg.pinv(self.resf2[self.xp.ix_(self.A[i], self.A[i])])
                    local_node_number = len(self.xp.where(self.A[i][:i])[0])
                    global_node_number = self.xp.where(self.A[i])[0]
                    self.F[i, global_node_number] += Fh[local_node_number]
                    self.F[global_node_number, i] += Fh[:, local_node_number]

                # self.F += self.xp.diag(self.xp.diag(self.F))
                self.F /= 2.0
            else:
                self.F = self.resf1 @ self.xp.linalg.pinv(self.resf2)


    def em_forward_QR(self, t, T, F=None, save_path=None, fillnum=3):
        """Calculate maximization step for EM algorithm.

        Args:
            t {int}
                : time for calculating.
            T {int}
                : last time for forward step.
            F [n_dim_sys, n_dim_sys]{numpy-array, float}
                : also known as :math:`F`. transition matrix from x_{t-1} to x_{t}
                システムモデルの変換行列[状態変数軸，状態変数軸]
        """
        if F is None:
            F = self.F
        if save_path is None:
            save_path = self.save_path

        V_smooth = self.xp.load(os.path.join(save_path, "smoothed_covariance_"
                                            + str(t).zfill(fillnum) + ".npy"))

        if "R" in self.em_vars:
            self.resr += V_smooth

        if "Q" in self.em_vars:
            if t!=0:
                V_pair = self.xp.load(os.path.join(save_path, "pairwise_covariance_"
                                                    + str(t-1).zfill(fillnum) + ".npy"))
                self.resq += V_smooth
                self.resq -= V_pair @ F.T + (V_pair @ F.T).T
            elif t!=T:
                self.resq += F @ V_smooth @ F.T


    def em_update_QR(self, T, y, F=None, H=None, b=None, d=None, save_path=None, fillnum=3):
        """

        Args:
            T {int}
                : last time for forward step.
            y [n_time, n_dim_obs] {numpy-array, float}
                : also known as :math:`y`. observation value
                観測値[時間軸,観測変数軸]
            F [n_dim_sys, n_dim_sys]{numpy-array, float}
                : also known as :math:`F`. transition matrix from x_{t-1} to x_{t}
                システムモデルの変換行列[状態変数軸，状態変数軸]
            H [n_dim_sys, n_dim_obs] {numpy-array, float}
                : also known as :math:`H`. observation matrix from x_{t} to y_{t}
                観測行列 [状態変数軸，観測変数軸]
            b [n_dim_sys], {numpy-array, float} 
                : also known as :math:`b`. system offset for times.
                システムモデルの切片（バイアス，オフセット） [状態変数軸]
            d [n_dim_obs] {numpy-array, float}
                : also known as :math:`d`. observation offset for times.
                観測モデルの切片 [観測変数軸]

        """
        if F is None:
            F = self.F
        if H is None:
            H = self.H
        if b is None:
            b = self.b
        if d is None:
            d = self.d
        if save_path is None:
            save_path = self.save_path

        x_smooth = self.xp.zeros((T, self.n_dim_sys), dtype = self.dtype)
        for t in range(T):
            x_smooth[t] = self.xp.load(os.path.join(save_path, "smoothed_mean_"
                                        + str(t).zfill(fillnum) + ".npy"))
        if "R" in  self.em_vars:
            err = y - (H @ x_smooth.T).T - d.reshape(1, len(d))
            res = err.T @ err + H @ self.resr @ H.T
            self.R = res / T

        if "Q" in self.em_vars:
            err = x_smooth[1:] - x_smooth[:-1] @ F.T - b.reshape(1, len(b))
            res = err.T @ err + self.resq
            self.Q = res / (T - 1)



    def em_forward_bd(self, t, y, F=None, H=None, save_path=None, fillnum=3):
        """Calculate maximization step for EM algorithm.

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
        """
        if F is None:
            F = self.F
        if H is None:
            H = self.H
        if save_path is None:
            save_path = self.save_path

        # load backward result
        x_smooth = self.xp.load(os.path.join(save_path, "smoothed_mean_"
                                        + str(t).zfill(fillnum) + ".npy"))

        if "d" in self.em_vars:
            self.resd += y - H @ x_smooth

        if "b" in self.em_vars:
            if t!=0:
                x_smooth_old = self.xp.load(os.path.join(save_path, "smoothed_mean_"
                                                    + str(t-1).zfill(fillnum) + ".npy"))
                self.resb += x_smooth - F @ x_smooth_old


    def em_update_bd(self, T):
        """
        Args:
            T {int}
                : last time for forward step.
        """
        if "d" in self.em_vars:
            self.d = self.resd / T
        
        if "b" in self.em_vars:
            self.b = self.resb / (T - 1)


    def _parallel_global_to_local(self, variable, n_dim, adjacency_matrix):
        """Parallel calculation for global to local

        Args:
            variable : input variable which transited from global to local
            n_dim : dimension of variables
            adjacency_matrix : adjacency matrix for transition
        """
        p = Pool(self.cpu_number)
        new_variable = p.starmap(global_to_local,
            zip(adjacency_matrix, itertools.repeat(variable, n_dim)))
        p.close()
        return new_variable
