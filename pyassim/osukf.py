"""
=============================
Inference with Sequential Update Kalman Filter
=============================
This module implements the Sequential Update Kalman Filter
and Kalman Smoother for Linear-Gaussian state space models
"""

import math

import numpy as np

from .utils import array1d, array2d
from .util_functions import _parse_observations, _last_dims, \
    _determine_dimensionality


class OffsetsSequentialUpdateKalmanFilter(object) :
    """Implements the Kalman Filter, Kalman Smoother, and EM algorithm.
    This class implements the Kalman Filter, Kalman Smoother, and EM Algorithm
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
            観測値[時間軸,観測変数軸]
        initial_mean [n_dim_sys] {float} 
            also known as :math:`\mu_0`. initial state mean
            初期状態分布の期待値[状態変数軸]
        initial_covariance [n_dim_sys, n_dim_sys] {numpy-array, float} 
            also known as :math:`\Sigma_0`. initial state covariance
            初期状態分布の共分散行列[状態変数軸，状態変数軸]
        transition_matrices [n_dim_sys, n_dim_sys] 
            or [n_dim_sys, n_dim_sys]{numpy-array, float}
            also known as :math:`F`. transition matrix from x_{t-1} to x_{t}
            システムモデルの変換行列[状態変数軸，状態変数軸]
        observation_matrices [n_time, n_dim_sys, n_dim_obs] or [n_dim_sys, n_dim_obs]
             {numpy-array, float}
            also known as :math:`H`. observation matrix from x_{t} to y_{t}
            観測行列[時間軸，状態変数軸，観測変数軸] or [状態変数軸，観測変数軸]
        transition_covariance [n_time - 1, n_dim_noise, n_dim_noise]
             or [n_dim_sys, n_dim_noise]
            {numpy-array, float}
            also known as :math:`Q`. system transition covariance for times
            システムノイズの共分散行列[時間軸，ノイズ変数軸，ノイズ変数軸]
        observation_covariance [n_time, n_dim_obs, n_dim_obs] {numpy-array, float} 
            also known as :math:`R`. observation covariance for times.
            観測ノイズの共分散行列[時間軸，観測変数軸，観測変数軸]
        update_interval {int}
            : interval of update transition matrix F
        eta (in (0.1))
            : update rate for transition matrix F
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
        y : `observation`
        F : `transition_matrices`
        Q : `transition_covariance`
        H : `observation_matrices`
        R : `observation_covariance`
        transition_cs : `transition_covariance_structure`
        observation_cs : `observation_covariance_structure`
        transition_v : `transition_vh_length`
        observation_v : `observation_vh_length`
        x_pred [n_time+1, n_dim_sys] {numpy-array, float} 
            mean of predicted distribution
            予測分布の平均 [時間軸，状態変数軸]
        V_pred [n_time+1, n_dim_sys, n_dim_sys] {numpy-array, float}
            covariance of predicted distribution
            予測分布の共分散行列 [時間軸，状態変数軸，状態変数軸]
        x_filt [n_time+1, n_dim_sys] {numpy-array, float}
            mean of filtered distribution
            フィルタ分布の平均 [時間軸，状態変数軸]
        V_filt [n_time+1, n_dim_sys, n_dim_sys] {numpy-array, float}
            covariance of filtered distribution
            フィルタ分布の共分散行列 [時間軸，状態変数軸，状態変数軸]
        x_smooth [n_time, n_dim_sys] {numpy-array, float}
            mean of RTS smoothed distribution
            固定区間平滑化分布の平均 [時間軸，状態変数軸]
        V_smooth [n_time, n_dim_sys, n_dim_sys] {numpy-array, float}
            covariance of RTS smoothed distribution
            固定区間平滑化の共分散行列 [時間軸，状態変数軸，状態変数軸]
        filter_update {function}
            update function from x_{t} to x_{t+1}
            フィルター更新関数
    """

    def __init__(self, observation = None,
                initial_mean = None, initial_covariance = None,
                transition_matrices = None, observation_matrices = None,
                transition_covariance = None, observation_covariance = None,
                transition_offsets = None, observation_offsets = None,
                update_interval = 1, eta = 0.1, cutoff = 0.1, 
                save_transition_change = True, method = "at-same-time",
                n_dim_sys = None, n_dim_obs = None, dtype = "float32",
                xp = "numpy"):
        """Setup initial parameters.
        """
        if xp=="numpy":
            self.xp = np
        elif xp=="cupy":
            import cupy
            self.xp = cupy

        # determine dimensionality
        self.n_dim_sys = _determine_dimensionality(
            [(transition_matrices, array2d, -2),
             (transition_offsets, array1d, -1),
             (initial_mean, array1d, -1),
             (initial_covariance, array2d, -2),
             (observation_matrices, array2d, -1)],
            n_dim_sys
        )

        self.n_dim_obs = _determine_dimensionality(
            [(observation_matrices, array2d, -2),
             (observation_covariance, array2d, -2),
             (observation_offsets, array1d, -1)],
            n_dim_obs
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

        if transition_matrices is None:
            self.F = self.xp.eye(self.n_dim_sys, dtype = dtype)
        else:
            self.F = transition_matrices.astype(dtype)

        if transition_covariance is None:
            self.Q = self.xp.eye(self.n_dim_sys, dtype = dtype)
        else:
            self.Q = transition_covariance.astype(dtype)

        if transition_offsets is None:
            self.b = self.xp.zeros(self.n_dim_sys, dtype = dtype)
        else:
            self.b = transition_offsets.astype(dtype)

        if observation_matrices is None:
            self.H = self.xp.eye(self.n_dim_obs, self.n_dim_sys, dtype = dtype)
        else:
            self.H = observation_matrices.astype(dtype)
        
        if observation_covariance is None:
            self.R = self.xp.eye(self.n_dim_obs, dtype = dtype)
        else:
            self.R = observation_covariance.astype(dtype)

        if observation_offsets is None:
            self.d = self.xp.zeros(self.n_dim_obs, dtype = dtype)
        else:
            self.d = observation_offsets.astype(dtype)

        if method in ["leap-flog", "at-same-time"]:
            self.method = method
        else:
            raise ValueError("Variable \"method\" needs to be selected from \"reap-flog\", "
                + "or \"at-same-time\". However, your choice is {}".format(method))

        self.update_interval = int(update_interval)
        self.save_transition_change = save_transition_change

        if save_transition_change:
            if self.method=="leap-flog":
                self.Fs = self.xp.zeros((math.floor(len(self.y)/(2*self.update_interval))+1,
                                    self.F.shape[0], self.F.shape[1]))
                self.bs = self.xp.zeros((math.floor(len(self.y)/(self.update_interval))//2+1,
                                    len(self.b)))
            elif self.method=="at-same-time":
                self.Fs = self.xp.zeros(((len(self.y)-1)//self.update_interval+1,
                                    self.F.shape[0], self.F.shape[1]))
                self.bs = self.xp.zeros(((len(self.y)-1)//self.update_interval+1,
                                    len(self.b)))
            self.Fs[0] = self.F
            self.bs[0] = self.b

        self.G = self.xp.eye(self.n_dim_obs, dtype=dtype)
        self.c = self.xp.zeros(self.n_dim_obs, dtype=dtype)

        self.eta = eta
        self.cutoff = cutoff
        self.dtype = dtype


    def forward(self):
        """Calculate prediction and filter for observation times.

        Attributes:
            T {int}
                : length of data y （時系列の長さ）
            x_pred [n_time, n_dim_sys] {numpy-array, float}
                : mean of hidden state at time t given observations
                 from times [0...t-1]
                時刻 t における状態変数の予測期待値 [時間軸，状態変数軸]
            V_pred [n_time, n_dim_sys, n_dim_sys] {numpy-array, float}
                : covariance of hidden state at time t given observations
                 from times [0...t-1]
                時刻 t における状態変数の予測共分散 [時間軸，状態変数軸，状態変数軸]
            x_filt [n_time, n_dim_sys] {numpy-array, float}
                : mean of hidden state at time t given observations from times [0...t]
                時刻 t における状態変数のフィルタ期待値 [時間軸，状態変数軸]
            V_filt [n_time, n_dim_sys, n_dim_sys] {numpy-array, float}
                : covariance of hidden state at time t given observations
                 from times [0...t]
                時刻 t における状態変数のフィルタ共分散 [時間軸，状態変数軸，状態変数軸]
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
            else:
                self._predict_update(t)
            
            # If y[t] has any mask, skip filter calculation
            # if self.xp.any(self.xp.ma.getmask(self.y[t])) :
            # if self.xp.any(self.xp.isnan(self.y[t])) :
            #     self.x_filt[t] = self.x_pred[t]
            #     self.V_filt[t] = self.V_pred[t]
            # else :
            self._filter_update(t)
            # ToDo : if there exists nan, more consider this part
            if t>0 and t%self.update_interval==0:
                if self.method == "leap-flog":
                    if (t//self.update_interval)%2==0:
                        self._update_transition_matrix(t)
                    else:
                        self._update_transition_offset(t)
                elif self.method == "at-same-time":
                    self._update_transition_matrix(t)
                    self._update_transition_offset(t)


    def _predict_update(self, t):
        """Calculate fileter update

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


    def _filter_update(self, t):
        """Calculate fileter update without noise

        Args:
            t {int} : observation time

        Attributes:
            K [n_dim_sys, n_dim_obs] {numpy-array, float}
                : Kalman gain matrix for time t [状態変数軸，観測変数軸]
                カルマンゲイン
        """
        # extract parameters for time t
        H = _last_dims(self.H, t, 2)
        R = _last_dims(self.R, t, 2)
        d = _last_dims(self.d, t, 1)

        # calculate filter step
        K = self.V_pred[t] @ (
            H.T @ self.xp.linalg.pinv(H @ (self.V_pred[t] @ H.T) + R)
            )
        target = self.xp.isnan(self.y[t])
        self.y[t][target] = (H @ self.x_pred[t])[target]
        self.x_filt[t] = self.x_pred[t] + K @ (
            self.y[t] - (H @ self.x_pred[t] + d)
            )
        self.y[t][target] = (H @ self.x_filt[t])[target]
        self.V_filt[t] = self.V_pred[t] - K @ (H @ self.V_pred[t])


    def _update_transition_matrix(self, t):
        """Update transition matrix

        Args:
            t {int} : observation time
        """
        self.G = (self.y[t-self.update_interval+1:t+1].T - self.c.reshape(-1,1)) \
                @ self.xp.linalg.pinv(self.y[t-self.update_interval:t].T) 
        Fh = self.xp.linalg.pinv(self.H) \
                @ (self.G @ self.H \
                    - self.xp.tile(self.H @ self.b - self.c, (self.update_interval, 1)).T \
                        @ self.xp.linalg.pinv(self.x_filt[t-self.update_interval:t].T))
        # self.F = (1 - self.eta) * self.F + self.eta * Fh
        self.F = self.F - self.eta * self.xp.minimum(self.xp.maximum(-self.cutoff, self.F - Fh), self.cutoff)

        if self.save_transition_change:
            if self.method == "leap-flog":
                self.Fs[t//(2*self.update_interval)] = self.F
            elif self.method == "at-same-time":
                self.Fs[t//self.update_interval] = self.F


    def _update_transition_offset(self, t):
        """Update transition offset

        Args:
            t {int} : observation time
        """
        C = self.y[t-self.update_interval+1:t+1].T - self.G @ self.y[t-self.update_interval:t].T
        self.c = self.xp.mean(C, axis=1)
        B = self.xp.linalg.pinv(self.H) @ ((self.G @ self.H - self.H @ self.F) \
                                            @ self.x_filt[t-self.update_interval:t].T + C)
        self.b = self.b - self.eta * self.xp.minimum(self.xp.maximum(-self.cutoff, 
                                                    self.b - self.xp.mean(B, axis=1)), self.cutoff)

        if self.save_transition_change:
            if self.method == "leap-flog":
                self.bs[t//(2*self.update_interval)+1] = self.b
            elif self.method == "at-same-time":
                self.bs[t//self.update_interval] = self.b


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


    def get_transition_matrices(self, ids = None):
        """Get transition matrices
        
        Args:
            ids {numpy-array, int} : ids of transition matrices

        Returns {numpy-array, float}:
            : transition matrices
        """
        if self.save_transition_change:
            if ids is None:
                return self.Fs
            else:
                return self.Fs[ids]
        else:
            return self.F


    def get_transition_offsets(self, ids = None):
        """Get transition matrices
        
        Args:
            ids {numpy-array, int} : ids of transition matrices

        Returns {numpy-array, float}:
            : transition offsets
        """
        if self.save_transition_change:
            if ids is None:
                return self.bs
            else:
                return self.bs[ids]
        else:
            return self.b


    def smooth(self):
        """Calculate RTS smooth for times.

        Args:
            T : length of data y (時系列の長さ)
            x_smooth [n_time, n_dim_sys] {numpy-array, float}
                : mean of hidden state distributions for times
                 [0...n_times-1] given all observations
                時刻 t における状態変数の平滑化期待値 [時間軸，状態変数軸]
            V_smooth [n_time, n_dim_sys, n_dim_sys] {numpy-array, float}
                : covariances of hidden state distributions for times
                 [0...n_times-1] given all observations
                時刻 t における状態変数の平滑化共分散 [時間軸，状態変数軸，状態変数軸]
            A [n_dim_sys, n_dim_sys] {numpy-array, float}
                : fixed interval smoothed gain
                固定区間平滑化ゲイン [時間軸，状態変数軸，状態変数軸]
        """

        # if not implement `filter`, implement `filter`
        try :
            self.x_pred[0]
        except :
            self.filter()

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
