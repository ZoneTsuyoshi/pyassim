"""
=========================================
Inference with Ensemble Kalman Filter
=========================================
This module implements the Ensemble Kalman Filter and Ensemble Kalman Smoother,
for Nonlinear Semi-Gaussian state space models
"""

import math

import numpy as np
try:
    import cupy
    print("successfully import cupy at enkf.")
    xp = cupy
    import cupy.random as rd
    from cupy import linalg
except:
    xp = np
    import numpy.random as rd
    # from numpy import linalg
    from scipy import linalg
import pandas as pd

from .utils import array1d, array2d, check_random_state, get_params, \
    preprocess_arguments, check_random_state
from .util_functions import _parse_observations, _last_dims, \
    _determine_dimensionality


class EnsembleKalmanFilter(object):
    """Implements the Ensemble Kalman Filter and Ensemble Kalman Smoother.
    This class implements the Ensemble Kalman Filter and Ensemble Kalman Smoother
    for a Nonlinear Semi-Gaussian model specified by,
    .. math::
        x_{t+1}   &= f_{t}(x_{t}) + v_{t}, v_{t} &\sim p(v_{t}) \\
        y_{t}     &= H_{t} x_{t} + w_{t}, w_{t} &\sim N(0, R_{t}) \\
    The Enseble Kalman Filter is an algorithm designed to estimate
    :math:`P(x_t | y_{0:t})`.  All state transitions are nonlinear with
    non-Gaussian distributed noise, observations are linear with Gaussian
    distributed noise.
    Similarly, the Ensemble Kalman Smoother is an algorithm designed to estimate
    :math:`P(x_t | y_{0:T-1})`.

    Args:
        y, observation [n_time, n_dim_obs] {numpy-array, float}
            also known as :math:`y`. observation value
            観測値 [時間軸,観測変数軸]
        initial_mean [n_dim_sys] {float} 
            also known as :math:`\mu_0`. initial state mean
            初期状態分布の期待値 [状態変数軸]
        f, transition_functions [n_time] {function}
            also known as :math:`f`. transition function from x_{t-1} to x_{t}
            システムモデルの遷移関数 [時間軸] or []
        H, observation_matrices [n_time, n_dim_sys, n_dim_obs] {numpy-array, float}
            also known as :math:`H`. observation matrices from x_{t} to y_{t}
            観測行列 [時間軸，状態変数軸，観測変数軸] or [状態変数軸，観測変数軸]
        q, transition_noise [n_time - 1] {(method, parameters)}
            also known as :math:`v` and `p(v)`. transition noise for v_{t}
            システムノイズの発生方法とパラメータ [時間軸]
            サイズは指定できる形式
        R, observation_covariance [n_time, n_dim_obs, n_dim_obs]
            or [n_dim_obs, n_dim_obs] {numpy-array, float} 
            also known as :math:`R`. covariance of observation normal noise
            観測の共分散行列 [時間軸，観測変数軸，観測変数軸] or [観測変数軸，観測変数軸]
        n_particle {int}
            : number of particles or ensemble members
            粒子数
        n_dim_sys {int}
            : dimension of system variable
            システム変数の次元
        n_dim_obs {int}
            : dimension of observation variable
            観測変数の次元
        dtype {xp.dtype}
            : dtype of numpy-array
            numpy のデータ型
        seed {int}
            : random seed
            ランダムシード
    """

    def __init__(self, observation = None, transition_functions = None,
                observation_matrices = None, initial_mean = None,
                transition_noise = None, observation_covariance = None,
                n_particle = 100, n_dim_sys = None, n_dim_obs = None,
                dtype = xp.float32, seed = 10) :

        # check order of tensor and mask missing value
        self.y = _parse_observations(observation)

        # determine dimensionality
        self.n_dim_sys = _determine_dimensionality(
            [(initial_mean, array1d, -1)],
            n_dim_sys
        )

        self.n_dim_obs = _determine_dimensionality(
            [(observation_covariance, array2d, -2)],
            n_dim_obs
        )

        # transition_functions
        # None -> system + noise
        if transition_functions is None:
            self.f = [lambda x, v: x + v]
        else:
            self.f = transition_functions

        # observation_matrices
        # None -> xp.eye
        if observation_matrices is None:
            self.H = xp.eye(self.n_dim_obs, self.n_dim_sys, dtype = dtype)
        else:
            self.H = observation_matrices.astype(dtype)

        # transition_noise
        # None -> standard normal distribution
        if transition_noise is None:
            self.q = (rd.multivariate_normal,
                [xp.zeros(self.n_dim_sys, dtype = dtype),
                xp.eye(self.n_dim_sys, dtype = dtype)])
        else:
            self.q = transition_noise

        # observation_covariance
        # None -> xp.eye
        if observation_covariance is None:
            self.R = xp.eye(self.n_dim_obs, dtype = dtype)
        else:
            self.R = observation_covariance.astype(dtype)

        # initial_mean
        # None -> xp.zeros
        if initial_mean is None:
            self.initial_mean = xp.zeros(self.n_dim_sys, dtype = dtype)
        else:
            self.initial_mean = initial_mean.astype(dtype)

        self.n_particle = n_particle
        xp.random.seed(seed)
        self.seed = seed
        self.dtype = dtype


    def forward(self):
        """Calculate prediction and filter for observation times.

        Attributes (self):
            T {int}
                : length of data y
                時系列の長さ
            x_pred_mean [n_time+1, n_dim_sys] {numpy-array, float}
                : mean of `x_pred` regarding to particles at time t
                時刻 t における x_pred の粒子平均 [時間軸，状態変数軸]
            V_pred [n_time+1, n_dim_sys, n_dim_sys] {numpy-array, float}
                : covariance of hidden state at time t given observations
                 from times [0...t-1]
                時刻 t における状態変数の予測共分散 [時間軸，状態変数軸，状態変数軸]
            x_filt [n_time+1, n_particle, n_dim_sys] {numpy-array, float}
                : hidden state at time t given observations for each particle
                状態変数のフィルタアンサンブル [時間軸，粒子軸，状態変数軸]
            x_filt_mean [n_time+1, n_dim_sys] {numpy-array, float}
                : mean of `x_filt` regarding to particles
                時刻 t における状態変数のフィルタ平均 [時間軸，状態変数軸]
            X5 [n_time, n_dim_sys, n_dim_obs] {numpy-array, float}
                : right operator for filter, smooth calulation
                filter, smoothing 計算で用いる各時刻の右作用行列

        Attributes (local):
            x_pred [n_particle, n_dim_sys] {numpy-array, float}
                : hidden state at time t given observations for each particle
                状態変数の予測アンサンブル [粒子軸，状態変数軸]
            x_pred_center [n_particle, n_dim_sys] {numpy-array, float}
                : centering of `x_pred`
                x_pred の中心化 [粒子軸，状態変数軸]
            w_ensemble [n_particle, n_dim_obs] {numpy-array, float}
                : observation noise ensemble
                観測ノイズのアンサンブル [粒子軸，観測変数軸]
            Inovation [n_dim_obs, n_particle] {numpy-array, float}
                : Innovation from observation [観測変数軸，粒子軸]
                観測と予測のイノベーション
        """

        # lenght of time-series
        T = self.y.shape[0]

        ## definition of array
        # initial setting
        self.x_pred_mean = xp.zeros((T + 1, self.n_dim_sys), dtype = self.dtype)
        self.x_filt = xp.zeros((T + 1, self.n_particle, self.n_dim_sys),
             dtype = self.dtype)
        self.x_filt[0, :] = self.initial_mean
        self.x_filt_mean = xp.zeros((T + 1, self.n_dim_sys), dtype = self.dtype)
        self.X5 = xp.zeros((T + 1, self.n_particle, self.n_particle),
             dtype = self.dtype)
        self.X5[:] = xp.eye(self.n_particle, dtype = self.dtype)

        self.x_pred_mean[0] = self.initial_mean
        self.x_filt_mean[0] = self.initial_mean

        x_pred = xp.zeros((self.n_particle, self.n_dim_sys), dtype = self.dtype)
        x_pred_center = xp.zeros((T + 1, self.n_particle, self.n_dim_sys),
             dtype = self.dtype)
        w_ensemble = xp.zeros((self.n_particle, self.n_dim_obs), dtype = self.dtype)

        # observation innovation
        Inovation = xp.zeros((self.n_dim_obs, self.n_particle), dtype = self.dtype)

        # calculate prediction and filtering step
        for t in range(T):
            # visualize calculationing time
            print("\r filter calculating... t={}".format(t+1) + "/" + str(T), end="")

            ## filter update
            # calculate prediction step
            f = _last_dims(self.f, t, 1)[0]

            # raise system noise parametric
            v = self.q[0](*self.q[1], size = self.n_particle)

            # calculate ensemble prediction
            # x_pred = f(*xp.transpose([self.x_filt[t], v], (0, 2, 1))).T
            x_pred = f(self.x_filt[t].T, v.T).T

            # calculate `x_pred_mean`
            self.x_pred_mean[t + 1] = xp.mean(x_pred, axis = 0)

            # treat missing values
            if xp.any(xp.ma.getmask(self.y[t])):
                self.x_filt[t + 1] = x_pred
            else:
                # calculate `x_pred_center`
                x_pred_center = x_pred - self.x_pred_mean[t + 1]

                # raise observation noise ensemble
                R = _last_dims(self.R, t)
                w_ensemble = rd.multivariate_normal(xp.zeros(self.n_dim_obs), R,
                    size = self.n_particle)

                # calculate innovation
                H = _last_dims(self.H, t)
                Inovation.T[:] = self.y[t]
                Inovation += w_ensemble.T - H @ x_pred.T

                # calculate singular value decomposition
                U, s, _ = linalg.svd(H @ x_pred_center.T + w_ensemble.T, False)

                # calculate right work matrix, following Evansen method
                X1 = xp.diag(1 / (s * s)) @ U.T
                X2 = X1 @ Inovation
                X3 = U @ X2
                X4 = (H @ x_pred_center.T).T @ X3
                self.X5[t + 1] = xp.eye(self.n_particle, dtype = self.dtype) + X4

                # calculate ensemble member for filtering
                self.x_filt[t + 1] = self.X5[t + 1].T @ x_pred

            # calculate filtering ensemble
            self.x_filt_mean[t + 1] = xp.mean(self.x_filt[t + 1], axis = 0)


    def get_predicted_value(self, dim = None) :
        """Get predicted value

        Args:
            dim {int} : dimensionality for extract from predicted result

        Returns (numpy-array, float)
            : mean of hidden state at time t given observations
            from times [0...t-1]
        """
        # if not implement `filter`, implement `filter`
        try :
            self.x_pred_mean[0]
        except :
            self.filter()

        if dim is None:
            return self.x_pred_mean[1:]
        elif dim <= self.x_pred_mean.shape[1]:
            return self.x_pred_mean[1:, int(dim)]
        else:
            raise ValueError("The dim must be less than "
                 + self.x_pred_mean.shape[1] + ".")


    def get_filtered_value(self, dim = None) :
        """Get filtered value

        Args:
            dim {int} : dimensionality for extract from filtered result

        Returns (numpy-array, float)
            : mean of hidden state at time t given observations
            from times [0...t]
        """
        # if not implement `filter`, implement `filter`
        try :
            self.x_filt_mean[0]
        except :
            self.filter()

        if dim is None:
            return self.x_filt_mean[1:]
        elif dim <= self.x_filt_mean.shape[1]:
            return self.x_filt_mean[1:, int(dim)]
        else:
            raise ValueError("The dim must be less than " 
                + self.x_filt_mean.shape[1] + ".")


    def smooth(self, lag = 10):
        """calculate fixed lag smooth

        Args:
            lag {int}
                : smoothing lag
                平滑化ラグ

        Attributes:
            x_smooth [n_time+1, n_particle, n_dim_sys] {numpy-array, float}
                : hidden state at time s given observations
                 from times [0...t] for each particle
                時刻 s における状態変数の平滑化値 [時間軸，粒子軸，状態変数軸]
            x_smooth_mean [n_time+1, n_dim_sys] {numpy-array, float}
                : mean of `x_smooth`
                時刻 s における平滑化 x_smooth の平均値
            V_smooth [n_dim_sys, n_dim_sys] {numpy-array, float}
                : covariance of hidden state at times 
                given observations from times [0...t]
                時刻 s における状態変数の予測共分散 [時間軸，状態変数軸，状態変数軸]
        """

        # length of time-series
        T = self.y.shape[0]

        # definite arrays
        x_smooth = xp.zeros((self.n_particle, self.n_dim_sys), dtype = self.dtype)
        self.x_smooth_mean = xp.zeros((T + 1, self.n_dim_sys), dtype = self.dtype)

        x_smooth[0] = self.initial_mean
        self.x_smooth_mean[0] = self.initial_mean

        for t in range(T + 1):
            # visualize calculating times
            print("\r smooth calculating... t={}".format(t+1) + "/" + str(T), end="")
            x_smooth = self.x_filt[t]

            # determine range for smoothing
            if t > T - lag:
                s_range = range(t + 1, T + 1)
            else:
                s_range = range(t + 1, t + lag + 1)

            for s in s_range:
                x_smooth = self.X5[s] @ x_smooth
            
            # calculate mean of smoothing
            self.x_smooth_mean[t] = xp.mean(x_smooth, axis = 0)


    def get_smoothed_value(self, dim = None) :
        """Get smoothed value

        Args:
            dim {int} : dimensionality for extract from RTS smoothed result

        Returns (numpy-array, float)
            : mean of hidden state at time t given observations
            from times [0...T]
        """
        # if not implement `smooth`, implement `smooth`
        try :
            self.x_smooth_mean[0]
        except :
            self.smooth()

        if dim is None:
            return self.x_smooth_mean[1:]
        elif dim <= self.x_smooth_mean.shape[1]:
            return self.x_smooth_mean[1:, int(dim)]
        else:
            raise ValueError("The dim must be less than "
             + self.x_smooth_mean.shape[1] + ".")



class NonlinearEnsembleKalmanFilter(object):
    def __init__(self, observation = None, transition_functions = None,
                observation_functions = None, initial_mean = None,
                transition_noise = None, observation_covariance = None,
                n_particle = 100, n_dim_sys = None, n_dim_obs = None,
                dtype = xp.float32, seed = 10) :

        # determine dimensionality
        self.n_dim_sys = _determine_dimensionality(
            [(initial_mean, array1d, -1)],
            n_dim_sys
        )

        self.n_dim_obs = _determine_dimensionality(
            [(observation_covariance, array2d, -2)],
            n_dim_obs
        )

        if observation_functions is not None:
            self.h = observation_functions
        else:
            self.h = [lambda x : x]

        # expanded system
        self.f = transition_functions
        expand_f = []
        expand_H = xp.hstack([xp.zeros((self.n_dim_obs, self.n_dim_sys)), xp.eye(self.n_dim_obs)])
        expand_initial_mean = xp.hstack([initial_mean, self.h[0](initial_mean)])
        for t in range(len(observation)):
            f = _last_dims(self.f, t, 1)[0]
            h = _last_dims(self.h, t, 1)[0]
            if t==0 or len(self.f)>1 or len(self.h)>1:
                expand_f.append(lambda expand_x, v : xp.vstack([f(expand_x[:self.n_dim_sys], v),
                                                        h(f(expand_x[:self.n_dim_sys], v))]))

        # set normal EnKF
        self.enkf = EnsembleKalmanFilter(observation, expand_f, expand_H, expand_initial_mean,
            transition_noise, observation_covariance, n_particle, self.n_dim_sys+self.n_dim_obs,
            self.n_dim_obs, dtype, seed)


    def forward(self):
        """Calculate prediction and filter for observation times."""
        self.enkf.forward()


    def get_predicted_value(self, dim = None) :
        """Get predicted value

        Args:
            dim {int} : dimensionality for extract from predicted result

        Returns (numpy-array, float)
            : mean of hidden state at time t given observations
            from times [0...t-1]
        """
        if dim is None:
            return self.enkf.get_predicted_value()[:self.n_dim_sys]
        elif dim <= self.n_dim_sys:
            return self.enkf.get_predicted_value(dim)
        else:
            raise ValueError("The dim must be less than "
                 + self.n_dim_sys + ".")


    def get_filtered_value(self, dim = None) :
        """Get filtered value

        Args:
            dim {int} : dimensionality for extract from filtered result

        Returns (numpy-array, float)
            : mean of hidden state at time t given observations
            from times [0...t]
        """
        if dim is None:
            return self.enkf.get_filtered_value()[:self.n_dim_sys]
        elif dim <= self.n_dim_sys:
            return self.enkf.get_filtered_value(dim)
        else:
            raise ValueError("The dim must be less than "
                 + self.n_dim_sys + ".")


    def smooth(self, lag = 10):
        """calculate fixed lag smooth

        Args:
            lag {int}
                : smoothing lag
                平滑化ラグ
        """
        self.enkf.smooth(lag)


    def get_smoothed_value(self, dim = None) :
        """Get smoothed value
        
        Args:
            dim {int} : dimensionality for extract from RTS smoothed result

        Returns (numpy-array, float)
            : mean of hidden state at time t given observations
            from times [0...T]
        """
        if dim is None:
            return self.enkf.get_smoothed_value()[:self.n_dim_sys]
        elif dim <= self.n_dim_sys:
            return self.enkf.get_smoothed_value(dim)
        else:
            raise ValueError("The dim must be less than "
                 + self.n_dim_sys + ".")

            