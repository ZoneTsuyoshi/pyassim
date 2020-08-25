"""
=========================================
Inference with Ensemble Kalman Filter
=========================================
This module implements the Ensemble Kalman Filter and Ensemble Kalman Smoother,
for Nonlinear Semi-Gaussian state space models
"""

import math

import numpy as np
# import pandas as pd

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
        n_particles {int}
            : number of particles or ensemble members
            粒子数
        n_dim_sys {int}
            : dimension of system variable
            システム変数の次元
        n_dim_obs {int}
            : dimension of observation variable
            観測変数の次元
        dtype {self.xp.dtype}
            : dtype of numpy-array
            numpy のデータ型
        seed {int}
            : random seed
            ランダムシード
    """

    def __init__(self, observation = None, transition_functions = None,
                observation_matrices = None, initial_mean = None,
                initial_covariance = None,
                transition_noise = None, observation_covariance = None,
                n_particles = 100, n_dim_sys = None, n_dim_obs = None,
                use_gpu = False,
                dtype = "float32", seed = 10):

        self.use_gpu = use_gpu
        if use_gpu:
            import cupy
            self.xp = cupy
        else:
            self.xp = np

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
        # None -> self.xp.eye
        if observation_matrices is None:
            self.H = self.xp.eye(self.n_dim_obs, self.n_dim_sys, dtype = dtype)
        else:
            self.H = observation_matrices.astype(dtype)

        # transition_noise
        # None -> standard normal distribution
        if transition_noise is None:
            self.q = (rd.multivariate_normal,
                [self.xp.zeros(self.n_dim_sys, dtype = dtype),
                self.xp.eye(self.n_dim_sys, dtype = dtype)])
        else:
            self.q = transition_noise

        # observation_covariance
        # None -> self.xp.eye
        if observation_covariance is None:
            self.R = self.xp.eye(self.n_dim_obs, dtype = dtype)
        else:
            self.R = observation_covariance.astype(dtype)

        # initial_mean
        # None -> self.xp.zeros
        if initial_mean is None:
            self.initial_mean = self.xp.zeros(self.n_dim_sys, dtype = dtype)
        else:
            self.initial_mean = initial_mean.astype(dtype)

        if initial_covariance is None:
            self.initial_covariance = self.xp.eye(self.n_dim_sys, dtype = dtype)
        else:
            self.initial_covariance = initial_covariance.astype(dtype)

        self.n_particles = n_particles
        self.xp.random.seed(seed)
        self.seed = seed
        self.dtype = dtype

        self.x_pred = self.xp.zeros((self.y.shape[0], self.n_particles, self.n_dim_sys),
             dtype = self.dtype)
        if self.initial_mean.ndim==2:
            self.n_particles = self.initial_mean.shape[0]
            self.x_pred[0] = self.initial_mean
        else:
            self.x_pred[0] = self.xp.random.multivariate_normal(self.initial_mean, self.initial_covariance,
                size=self.n_particles)






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
            x_filt [n_time+1, n_particles, n_dim_sys] {numpy-array, float}
                : hidden state at time t given observations for each particle
                状態変数のフィルタアンサンブル [時間軸，粒子軸，状態変数軸]
            x_filt_mean [n_time+1, n_dim_sys] {numpy-array, float}
                : mean of `x_filt` regarding to particles
                時刻 t における状態変数のフィルタ平均 [時間軸，状態変数軸]
            X5 [n_time, n_dim_sys, n_dim_obs] {numpy-array, float}
                : right operator for filter, smooth calulation
                filter, smoothing 計算で用いる各時刻の右作用行列

        Attributes (local):
            x_pred [n_particles, n_dim_sys] {numpy-array, float}
                : hidden state at time t given observations for each particle
                状態変数の予測アンサンブル [粒子軸，状態変数軸]
            x_pred_center [n_particles, n_dim_sys] {numpy-array, float}
                : centering of `x_pred`
                x_pred の中心化 [粒子軸，状態変数軸]
            w_ensemble [n_particles, n_dim_obs] {numpy-array, float}
                : observation noise ensemble
                観測ノイズのアンサンブル [粒子軸，観測変数軸]
            Inovation [n_dim_obs, n_particles] {numpy-array, float}
                : Innovation from observation [観測変数軸，粒子軸]
                観測と予測のイノベーション
        """

        # lenght of time-series
        T = self.y.shape[0]

        ## definition of array
        # initial setting
        self.x_pred_mean = self.xp.zeros((T, self.n_dim_sys), dtype = self.dtype)
        self.x_filt = self.xp.zeros((T, self.n_particles, self.n_dim_sys),
             dtype = self.dtype)
        self.x_filt_mean = self.xp.zeros((T, self.n_dim_sys), dtype = self.dtype)
        self.X5 = self.xp.zeros((T, self.n_particles, self.n_particles),
             dtype = self.dtype)
        self.X5[:] = self.xp.eye(self.n_particles, dtype = self.dtype)

        self.x_pred_mean[0] = self.x_pred[0].mean(axis=0)
        # self.x_filt_mean[0] = self.initial_mean

        # observation innovation
        Inovation = self.xp.zeros((self.n_dim_obs, self.n_particles), dtype = self.dtype)

        # calculate prediction and filtering step
        for t in range(T):
            # visualize calculationing time
            print("\r filter calculating... t={}".format(t+1) + "/" + str(T), end="")

            if t!=0:
                ## filter update
                # calculate prediction step
                f = _last_dims(self.f, t-1, 1)[0]

                # raise system noise parametric
                v = self.q[0](*self.q[1], size=self.n_particles)

                # calculate ensemble prediction
                self.x_pred[t] = f(self.x_filt[t-1].T, v.T).T # (np,Dx)

                # calculate `x_pred_mean`
                self.x_pred_mean[t] = self.xp.mean(self.x_pred[t], axis=0)

            # treat missing values
            if self.xp.any(self.xp.ma.getmask(self.y[t])):
                self.x_filt[t] = self.x_pred[t]
            else:
                # calculate `x_pred_center`
                x_pred_center = self.x_pred[t] - self.x_pred_mean[t]

                # raise observation noise ensemble
                R = _last_dims(self.R, t)
                w_ensemble = self.xp.random.multivariate_normal(self.xp.zeros(self.n_dim_obs), R,
                                                                size=self.n_particles)

                # calculate innovation
                H = _last_dims(self.H, t)
                Inovation.T[:] = self.y[t]
                Inovation += w_ensemble.T - H @ self.x_pred[t].T

                # calculate singular value decomposition
                U, s, _ = self.xp.linalg.svd(H @ x_pred_center.T + w_ensemble.T, False)

                # calculate right work matrix, following Evensen method
                X1 = self.xp.diag(1 / (s * s)) @ U.T
                X2 = X1 @ Inovation
                X3 = U @ X2
                X4 = (H @ x_pred_center.T).T @ X3
                self.X5[t] = self.xp.eye(self.n_particles, dtype=self.dtype) + X4

                # calculate ensemble member for filtering
                self.x_filt[t] = self.X5[t].T @ self.x_pred[t]

            # calculate filtering ensemble
            self.x_filt_mean[t] = self.xp.mean(self.x_filt[t], axis=0)


    def get_predicted_value(self, dim=None, get_particles=False) :
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

        if get_particles:
            if dim is None:
                return self.x_pred
            elif dim <= self.x_pred.shape[2]:
                return self.x_pred[:,:,int(dim)]
            else:
                raise ValueError("The dim must be less than "
                     + self.x_pred.shape[2] + ".")
        else:
            if dim is None:
                return self.x_pred_mean
            elif dim <= self.x_pred_mean.shape[1]:
                return self.x_pred_mean[:, int(dim)]
            else:
                raise ValueError("The dim must be less than "
                     + self.x_pred_mean.shape[1] + ".")


    def get_filtered_value(self, dim=None, get_particles=False) :
        """Get filtered value

        Args:
            dim {int} : dimensionality for extract from filtered result

        Returns (numpy-array, float)
            : mean of hidden state at time t given observations
            from times [0...t]
        """
        try :
            self.x_filt_mean[0]
        except :
            self.filter()

        if get_particles:
            if dim is None:
                return self.x_filt
            elif dim <= self.x_filt.shape[2]:
                return self.x_filt[:,:,int(dim)]
            else:
                raise ValueError("The dim must be less than " 
                    + self.x_filt.shape[2] + ".")
        else:
            if dim is None:
                return self.x_filt_mean
            elif dim <= self.x_filt_mean.shape[1]:
                return self.x_filt_mean[:, int(dim)]
            else:
                raise ValueError("The dim must be less than " 
                    + self.x_filt_mean.shape[1] + ".")


    def smooth(self, lag=10):
        """calculate fixed lag smooth

        Args:
            lag {int}
                : smoothing lag

        Attributes:
            x_smooth [n_time+1, n_particles, n_dim_sys] {numpy-array, float}
                : hidden state at time s given observations
                 from times [0...t] for each particle
            x_smooth_mean [n_time+1, n_dim_sys] {numpy-array, float}
                : mean of `x_smooth`
            V_smooth [n_dim_sys, n_dim_sys] {numpy-array, float}
                : covariance of hidden state at times 
                given observations from times [0...t]
        """
        try:
            self.x_filt_mean[0]
        except:
            self.forward()

        # length of time-series
        T = self.y.shape[0]

        # definite arrays
        self.x_smooth = self.xp.zeros((T, self.n_particles, self.n_dim_sys), dtype = self.dtype)
        self.x_smooth_mean = self.xp.zeros((T, self.n_dim_sys), dtype = self.dtype)

        self.x_smooth[0] = self.x_pred[0]
        self.x_smooth_mean[0] = self.initial_mean

        for t in range(T):
            # visualize calculating times
            print("\r smooth calculating... t={}".format(t+1) + "/" + str(T), end="")
            self.x_smooth[t] = self.x_filt[t]

            # determine range for smoothing
            if t > T - lag - 1:
                s_range = range(t+1, T)
            else:
                s_range = range(t+1, t+lag+1)

            for s in s_range:
                self.x_smooth[t] = self.X5[s] @ self.x_smooth[t]
            
            # calculate mean of smoothing
            self.x_smooth_mean[t] = self.xp.mean(self.x_smooth[t], axis=0)


    def get_smoothed_value(self, dim=None, get_particles=False) :
        """Get smoothed value

        Args:
            dim {int} : dimensionality for extract from RTS smoothed result

        Returns (numpy-array, float)
            : mean of hidden state at time t given observations
            from times [0...T]
        """
        # if not implement `smooth`, implement `smooth`
        try:
            self.x_smooth_mean[0]
        except:
            self.smooth()

        if get_particles:
            if dim is None:
                return self.x_smooth
            elif dim <= self.x_smooth.shape[2]:
                return self.x_smooth[:,:,int(dim)]
            else:
                raise ValueError("The dim must be less than "
                 + self.x_smooth.shape[2] + ".")
        else:
            if dim is None:
                return self.x_smooth_mean
            elif dim <= self.x_smooth_mean.shape[1]:
                return self.x_smooth_mean[:, int(dim)]
            else:
                raise ValueError("The dim must be less than "
                 + self.x_smooth_mean.shape[1] + ".")



class NonlinearEnsembleKalmanFilter(object):
    def __init__(self, observation = None, transition_functions = None,
                observation_functions = None, initial_mean = None,
                initial_covariance = None,
                transition_noise = None, observation_covariance = None,
                n_particles = 100, n_dim_sys = None, n_dim_obs = None,
                use_gpu = False, dtype = "float32", seed = 10):
        self.use_gpu = use_gpu
        if use_gpu:
            import cupy
            self.xp = cupy
        else:
            self.xp = np

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

        if initial_mean is None:
            self.initial_mean = self.xp.zeros(self.n_dim_sys, dtype = dtype)
        else:
            self.initial_mean = initial_mean.astype(dtype)

        if initial_covariance is None:
            self.initial_covariance = self.xp.eye(self.n_dim_sys, dtype = dtype)
        else:
            self.initial_covariance = initial_covariance.astype(dtype)

        # expanded system
        self.f = transition_functions
        expand_f = []
        expand_H = self.xp.hstack([self.xp.zeros((self.n_dim_obs, self.n_dim_sys)), self.xp.eye(self.n_dim_obs)])
        x0 = self.xp.random.multivariate_normal(self.initial_mean, self.initial_covariance,
            size=n_particles)
        expand_initial_mean = self.xp.concatenate([x0, self.h[0](x0)], axis=1)
        for t in range(len(observation)):
            f = _last_dims(self.f, t, 1)[0]
            h = _last_dims(self.h, t, 1)[0]
            if t==0 or len(self.f)>1 or len(self.h)>1:
                expand_f.append(lambda expand_x, v : self.xp.concatenate([f(expand_x[:self.n_dim_sys], v),
                                                        h(f(expand_x[:self.n_dim_sys], v).T).T], axis=0))

        # set normal EnKF
        self.enkf = EnsembleKalmanFilter(observation, expand_f, expand_H, expand_initial_mean,
            None, transition_noise, observation_covariance, n_particles, self.n_dim_sys+self.n_dim_obs,
            self.n_dim_obs, use_gpu, dtype, seed)


    def forward(self):
        """Calculate prediction and filter for observation times."""
        self.enkf.forward()


    def get_predicted_value(self, dim=None, get_particles=False) :
        """Get predicted value

        Args:
            dim {int} : dimensionality for extract from predicted result

        Returns (numpy-array, float)
            : mean of hidden state at time t given observations
            from times [0...t-1]
        """
        result = self.enkf.get_predicted_value(dim, get_particles)
        if dim is None:
            if get_particles:
                return result[:,:,:self.n_dim_sys]
            else:
                return result[:,:self.n_dim_sys]
        else:
            return result



    def get_filtered_value(self, dim=None, get_particles=False) :
        """Get filtered value

        Args:
            dim {int} : dimensionality for extract from filtered result

        Returns (numpy-array, float)
            : mean of hidden state at time t given observations
            from times [0...t]
        """
        result = self.enkf.get_filtered_value(dim, get_particles)
        if dim is None:
            if get_particles:
                return result[:,:,:self.n_dim_sys]
            else:
                return result[:,:self.n_dim_sys]
        else:
            return result



    def smooth(self, lag = 10):
        """calculate fixed lag smooth

        Args:
            lag {int}
                : smoothing lag
                平滑化ラグ
        """
        self.enkf.smooth(lag)


    def get_smoothed_value(self, dim=None, get_particles=False) :
        """Get smoothed value
        
        Args:
            dim {int} : dimensionality for extract from RTS smoothed result

        Returns (numpy-array, float)
            : mean of hidden state at time t given observations
            from times [0...T]
        """
        result = self.enkf.get_smoothed_value(dim, get_particles)
        if dim is None:
            if get_particles:
                return result[:,:,:self.n_dim_sys]
            else:
                return result[:,:self.n_dim_sys]
        else:
            return result


            