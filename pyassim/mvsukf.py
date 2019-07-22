"""
=============================================================
Inference with Missing Value Sequential Update Kalman Filter
=============================================================
This module implements the Sequential Update Kalman Filter
and Kalman Smoother for Linear-Gaussian state space models
"""

import math

import numpy as np

from .utils import array1d, array2d
from .util_functions import _parse_observations, _last_dims, \
    _determine_dimensionality


class MissingValueSequentialUpdateKalmanFilter(object) :
    """Implements the Kalman Filter, Kalman Smoother, and EM algorithm.
    This class implements the Kalman Filter, Kalman Smoother, and EM Algorithm
    for a Linear Gaussian model specified by,
    .. math::
        x_{t+1}   &= F_{t} x_{t} + v_{t} \\
        y_{t}     &= x_{t} + w_{t} \\
        [v_{t}, w_{t}]^T &\sim N(0, [[Q_{t}, O], [O, R_{t}]])
    The Kalman Filter is an algorithm designed to estimate
    :math:`P(x_t | y_{0:t})`.  As all state transitions and observations are
    linear with Gaussian distributed noise, these distributions can be
    represented exactly as Gaussian distributions with mean
    `x_filt[t]` and covariances `V_filt[t]`.
    Similarly, the Kalman Smoother is an algorithm designed to estimate
    :math:`P(x_t | y_{0:T-1})`.

    Args:
        observation [n_time, n_dim_sys] {numpy-array, float}
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
        observation_matrices [n_time, n_dim_sys, n_dim_sys] or [n_dim_sys, n_dim_sys]
             {numpy-array, float}
            also known as :math:`H`. observation matrix from x_{t} to y_{t}
            観測行列[時間軸，状態変数軸，観測変数軸] or [状態変数軸，観測変数軸]
        transition_covariance [n_time - 1, n_dim_noise, n_dim_noise]
             or [n_dim_sys, n_dim_noise]
            {numpy-array, float}
            also known as :math:`Q`. system transition covariance for times
            システムノイズの共分散行列[時間軸，ノイズ変数軸，ノイズ変数軸]
        observation_covariance [n_time, n_dim_sys, n_dim_sys] {numpy-array, float} 
            also known as :math:`R`. observation covariance for times.
            観測ノイズの共分散行列[時間軸，観測変数軸，観測変数軸]
        update_interval {int}
            : interval of update transition matrix F
        eta (in (0.1))
            : update rate for transition matrix F
        n_dim_sys {int}
            : dimension of system transition variable
            システム変数の次元
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
                transition_matrices = None,
                transition_covariance = None, observation_covariance = None,
                update_interval = 1, eta = 0.1, cutoff = 0.1, 
                mvmode = "filter",
                save_transition_matrix_change = True, calculate_variance = False,
                n_dim_sys = None, dtype = "float32",
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
             (initial_mean, array1d, -1),
             (initial_covariance, array2d, -2),
             (observation_covariance, array2d, -2)],
            n_dim_sys
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

        if transition_covariance is not None:
            self.Q = transition_covariance.astype(dtype)
        else:
            self.Q = self.xp.eye(self.n_dim_sys, dtype = dtype)
        
        if observation_covariance is None:
            self.R = self.xp.eye(self.n_dim_sys, dtype = dtype)
        else:
            self.R = observation_covariance.astype(dtype)

        self.update_interval = int(update_interval)

        if mvmode in ["filter", "smooth"]:
            self.mvmode = mvmode
            if mvmode=="smooth":
                self.bnP = self.xp.isnan(self.y[0])
        else:
            raise ValueError("Variable \"mvmode\" only allows \"filter\" or \"smooth\"."
                + "However, your choice is \"{}\".".format(mvmode))

        if calculate_variance:
            self.calculate_variance = True
            self.save_transition_matrix_change = True
        else:
            self.save_transition_matrix_change = save_transition_matrix_change
            self.calculate_variance = False

        if save_transition_matrix_change:
            self.Fs = self.xp.zeros(((len(self.y)-1)//self.update_interval+1,
                                self.F.shape[0], self.F.shape[1]))
            self.Fs[0] = self.F

            if calculate_variance:
                self.FV = self.xp.zeros(((len(self.y)-1)//self.update_interval+1,
                                self.F.shape[0], self.F.shape[1]))

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
            
            self._filter_update(t)
            # ToDo : if there exists nan, more consider this part
            if t>0 and t%self.update_interval==0:
                if self.calculate_variance:
                    self._update_transition_matrix_with_variance(t)
                else:
                    self._update_transition_matrix(t)


    def _predict_update(self, t):
        """Calculate fileter update

        Args:
            t {int} : observation time
        """
        # extract parameters for time t-1
        F = _last_dims(self.F, t - 1, 2)
        Q = _last_dims(self.Q, t - 1, 2)

        # calculate predicted distribution for time t
        self.x_pred[t] = F @ self.x_filt[t-1]
        self.V_pred[t] = F @ self.V_filt[t-1] @ F.T + Q


    def _filter_update(self, t):
        """Calculate fileter update without noise

        Args:
            t {int} : observation time

        Attributes:
            K [n_dim_obs_per_timestep, n_dim_sys] {numpy-array, float}
                : Kalman gain matrix for time t [状態変数軸，観測変数軸]
                カルマンゲイン
        """
        # extract parameters for time t
        R = _last_dims(self.R, t, 2)
        P = ~self.xp.isnan(self.y[t])

        # calculate filter step
        K = self.V_pred[t][:,P] @ self.xp.linalg.pinv(self.V_pred[t][self.xp.ix_(P,P)] + R[self.xp.ix_(P,P)])
        self.x_filt[t] = self.x_pred[t] + K @ (self.y[t][P] - self.x_pred[t][P])
        self.V_filt[t] = self.V_pred[t] - K @ self.V_pred[t][P,:]


    def _update_transition_matrix(self, t):
        """Update transition matrix

        Args:
            t {int} : observation time
        """
        if self.mvmode == "filter":
            for t in range(t-self.update_interval, t+1):
                nP = self.xp.isnan(self.y[t])
                self.y[t][nP] = self.x_filt[t][nP]
        elif self.mvmode == "smooth":
            self._smooth(t)

        Fh = self.y[t-self.update_interval+1:t+1].T \
                @ self.xp.linalg.pinv(self.y[t-self.update_interval:t].T)
        self.F = self.F - self.eta * self.xp.minimum(self.xp.maximum(-self.cutoff, self.F - Fh), self.cutoff)

        if self.save_transition_matrix_change:
            self.Fs[t//self.update_interval] = self.F


    def _smooth(self, t):
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
        x_smooth = self.xp.zeros((self.update_interval+1, self.n_dim_sys), dtype = self.dtype)
        V_smooth = self.xp.zeros((self.update_interval+1, self.n_dim_sys, self.n_dim_sys),
             dtype = self.dtype)
        A = self.xp.zeros((self.n_dim_sys, self.n_dim_sys), dtype = self.dtype)

        x_smooth[-1] = self.x_filt[t]
        V_smooth[-1] = self.V_filt[t]

        # t in [0, T-2] (notice t range is reversed from 1~T)
        for i,s in zip(reversed(range(self.update_interval)), reversed(range(t-self.update_interval, t))):
            # extract parameters for time t
            F = _last_dims(self.F, s, 2)

            # calculate fixed interval smoothing gain
            A = self.V_filt[s] @ F.T @ self.xp.linalg.pinv(self.V_pred[s + 1])
            
            # fixed interval smoothing
            x_smooth[i] = self.x_filt[s] \
                + A @ (x_smooth[i + 1] - self.x_pred[s + 1])
            V_smooth[i] = self.V_filt[s] \
                + A @ (V_smooth[i + 1] - self.V_pred[s + 1]) @ A.T

            nP = self.xp.isnan(self.y[s])
            self.y[s][nP] = x_smooth[i][nP]

            if i==0:
                self.y[s][self.bnP] = x_smooth[i][self.bnP]
                self.bnP = self.xp.isnan(self.y[t])

        self.y[t][self.bnP] = self.x_filt[t][self.bnP] # before not P


    def _update_transition_matrix_with_variance(self, t):
        """Update transition matrix with variance of transition matrix

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
        if self.save_transition_matrix_change:
            if ids is None:
                return self.Fs
            else:
                return self.Fs[ids]
        else:
            return self.F


    def get_variance_of_transition_matrices(self, ids = None):
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
