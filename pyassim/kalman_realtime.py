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

import numpy as np
try:
    import cupy
    print("successfully import cupy.")
    xp = cupy
    from cupy import linalg, sparse
except:
    xp = np
    from numpy import linalg
    from scipy import sparse
# from scipy import linalg

from .utils import array1d, array2d
from .util_functions import _determine_dimensionality


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
                n_dim_sys = None, n_dim_obs = None, dtype = xp.float32):
        """Setup initial parameters.
        """

        # determine dimensionality
        self.n_dim_sys = _determine_dimensionality(
            [(transition_matrix, array2d, -2),
             (transition_offset, array1d, -1),
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

        if initial_mean is None:
            self.x = xp.zeros(self.n_dim_sys, dtype = dtype)
        else:
            self.x = initial_mean.astype(dtype)
        
        if initial_covariance is None:
            self.V = xp.eye(self.n_dim_sys, dtype = dtype)
        else:
            self.V = initial_covariance.astype(dtype)

        if transition_matrix is None:
            # self.F = sparse.eye(self.n_dim_sys, dtype = dtype)
            self.F = xp.eye(self.n_dim_sys, dtype = dtype)
        else:
            self.F = transition_matrix.astype(dtype)

        if transition_covariance is not None:
            self.Q = transition_covariance.astype(dtype)
        else:
            # self.Q = sparse.eye(self.n_dim_sys, dtype = dtype)
            self.Q = xp.eye(self.n_dim_sys, dtype = dtype)

        if transition_offset is None :
            self.b = xp.zeros(self.n_dim_sys, dtype = dtype)
        else :
            self.b = transition_offset.astype(dtype)

        if observation_matrix is None:
            self.H = xp.eye(self.n_dim_obs, self.n_dim_sys, dtype = dtype)
        else:
            self.H = observation_matrix.astype(dtype)
        
        if observation_covariance is None:
            self.R = xp.eye(self.n_dim_obs, dtype = dtype)
        else:
            self.R = observation_covariance.astype(dtype)

        if observation_offset is None :
            self.d = xp.zeros(self.n_dim_obs, dtype = dtype)
        else :
            self.d = observation_offset.astype(dtype)

        self.dtype = dtype
        self.times = xp.zeros(2)


    def forward_update(self, y, F=None, H=None, Q=None, R=None, b=None, d=None):
        """Calculate prediction and filter regarding arguments.

        Args:
            y [n_time, n_dim_obs] {numpy-array, float}
                : also known as :math:`y`. observation value
                観測値[時間軸,観測変数軸]
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
        tnums = xp.zeros(6)
        tnums[0] = time.time()

        # calculate predicted distribution for time t
        self.x = F @ self.x + b
        self.V = F @ self.V @ F.T + Q

        tnums[1] = time.time()

        # print("predicted time : {}".format(t2-t1))
            
        # If y[t] has any mask, skip filter calculation
        # if not xp.any(xp.isnan(y)):
        # calculate filter step
        # K = linalg.inv(H @ (self.V @ H.T) + R)
        # L = linalg.cholesky(H @ (self.V @ H.T) + R)
        # L = linalg.inv(L)
        # K = L.T @ L
        # K = self.V @ (H.T @ K)
        K = self.V @ ( H.T @ linalg.inv(H @ (self.V @ H.T) + R) )
        self.x = self.x + K @ ( y - (H @ self.x + d) )
        self.V = self.V - K @ (H @ self.V)
        tnums[2] = time.time()

        for i in range(2):
            self.times[i] += tnums[i+1] - tnums[i]
        # print("filtered time : {}".format(t3-t2))
        return self.x