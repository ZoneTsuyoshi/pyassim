"""
==========================================
Inference with Auxiliary Particle Filter
==========================================
This module implements the Particle Filter and Particle Smoother,
for Nonlinear Non-Gaussian state space models
"""

import numpy as np
import numpy.random as rd
from scipy import linalg

from .utils import array1d, array2d, check_random_state, get_params, \
    preprocess_arguments, check_random_state
from .util_functions import _parse_observations, _last_dims, \
    _determine_dimensionality

class AuxiliaryParticleFilter(object):
    """Implements the Particle Filter and Particle Smoother.
    This class implements the Particle Filter and Particle Smoother
    for a Nonlinear Non-Gaussian model specified by,
    .. math::
        x_{t+1}   &= f_{t}(x_{t}) + v_{t}, v_{t} &\sim p(v_{t}) \\
        y_{t}     &= h_{t}(x_{t}) + w_{t}, w_{t} &\sim N(w_{t}) \\
    The Particle Filter is an algorithm designed to estimate
    :math:`P(x_t | y_{0:t})`.  All state transitions are nonlinear with
    non-Gaussian distributed noise, observations are nonlinear with non-Gaussian
    distributed noise.
    Similarly, the Particle Smoother is an algorithm designed to estimate
    :math:`P(x_t | y_{0:T-1})`.

    Args:
        y, observation [n_time, n_dim_obs] {numpy-array, float}
            also known as :math:`y`. observation value
            観測値 [時間軸,観測変数軸]
        initial_mean [n_dim_sys] {float} 
            also known as :math:`\mu_0`. initial state mean
            初期状態分布の期待値 [状態変数軸]
        initial_covariance [n_dim_sys, n_dim_sys] {numpy-array, float} 
            also known as :math:`\Sigma_0`. initial state covariance
            初期状態分布の共分散行列[状態変数軸，状態変数軸]
        f, transition_functions [n_time] {function}
            also known as :math:`f`. transition function from x_{t-1} to x_{t}
            システムモデルの遷移関数 [時間軸] or []
        q, transition_noise [n_time - 1] {(method, parameters)}
            also known as :math:`p(v)`. method and parameters of transition
            noise. noise distribution must be parametric and need input variable
            `size`, which mean number of ensemble
            システムノイズの発生方法とパラメータ [時間軸]
            サイズは指定できる形式
        lf, likelihood_functions [n_time] or [] {function}
            also known as :math:`p(w)`. likelihood function between x_t and y_t.
            only need kernel part and not need regulation part. likelihood function
            must be parameteric and need `likelihood_function_parameters`.
            観測モデルの尤度関数 [時間軸] or []
        lfp, likelihood_function_parameters [n_time, n_param] or [n_param]
         {numpy-array, float}
            : parameters for `likelihood_functions`
            尤度関数のパラメータ群 [時間軸，パラメータ軸] or [パラメータ軸]
        likelihood_function_is_log_form {boolean}
            : which `likelihood_functions` are log form. If true,
            `likelihood_functions` mean log likelihood function. If false,
            `likelihood_functions` mean likelihood function. For example,
            if you use gaussian distribution, whose kernel are exponential
            form, then you should use log from because of overflow problem.
            尤度関数の対数形式の有無
        observation_parameters_time_invariant {boolean}
            : which observation parameters are time-invariant. If true,
            `likelihood_functions` and `likelihood_function_parameters` has
            time-invariantness. If false, they are time-variant
            観測パラメータの時不変性の有無
        eta, regularization_noise [n_time - 1] {(method, parameters)}
            : noise distribution for regularization. noise distribution
            must be parametric and need input variable `size`,
            which mean number of ensemble
            正則化のためのノイズ分布
        n_particle {int}
            : number of particles (ensembles)
            粒子数
        n_dim_sys {int}
            : dimension of system variable
            システム変数の次元
        n_dim_obs {int}
            : dimension of observation variable
            観測変数の次元
        dtype {np.dtype}
            : dtype of numpy-array
            numpy のデータ型
        seed {int}
            : random seed
            ランダムシード

    Attributes:
        regularization {boolean}
            : which particle filter has regularization. If true,
            after filtering step, add state variables to regularization noise
            because of protecting from degeneration of particle.
            If false, doesn't add regularization noise.
            正則化項の導入の有無
    """
    def __init__(self, observation = None, 
                initial_mean = None, initial_covariance = None,
                transition_functions = None, transition_noise = None,
                likelihood_functions = None, likelihood_function_parameters = None,
                likelihood_function_is_log_form = True,
                observation_parameters_time_invariant = True,
                regularization_noise = None,
                n_particle = 100, n_dim_sys = None, n_dim_obs = None,
                dtype = np.float32, seed = 10) :
        # check order of tensor and mask missing values
        self.y = _parse_observations(observation)

        # determine dimensionality
        self.n_dim_sys = _determine_dimensionality(
            [(initial_mean, array1d, -1),
            (initial_covariance, array2d, -2)],
            n_dim_sys
        )

        self.n_dim_obs = _determine_dimensionality(
            [(observation, array1d, -1)],
            n_dim_obs
        )

        # transition_functions
        # None -> system + noise
        if transition_functions is None:
            self.f = [lambda x, v: x + v]
        else:
            self.f = transition_functions

        # transition_noise
        # None -> standard normal distribution
        if transition_noise is None:
            self.q = (rd.multivariate_normal,
                [np.zeros(self.n_dim_sys, dtype = dtype),
                np.eye(self.n_dim_sys, dtype = dtype)])
        else:
            self.q = transition_noise

        # initial_mean None -> np.zeros
        if initial_mean is None:
            self.initial_mean = np.zeros(self.n_dim_sys, dtype = dtype)
        else:
            self.initial_mean = initial_mean.astype(dtype)

        # initial_covariance None -> np.eye
        if initial_covariance is None:
            self.initial_covariance = np.eye(self.n_dim_sys, dtype = dtype)
        else:
            self.initial_covariance = initial_covariance.astype(dtype)

        # likelihood_functions, likelihood_function_parameters None
        if likelihood_function_parameters is None:
            self.likelihood_function_is_log_form = likelihood_function_is_log_form
            self.observation_parameters_time_invariant \
                = observation_parameters_time_invariant
            self.lf = likelihood_functions
            self.lfp = likelihood_function_parameters
        else:
            self.likelihood_function_is_log_form = True
            self.observation_parameters_time_invariant = True
            self.lf = self._log_norm_likelihood

            # use normal likelihood, but want to change parameter R
            if likelihood_functions is None:
                self.lfp = [np.eye(self.n_dim_obs, dtype = dtype)]
            else:
                self.lfp = likelihood_function_parameters

        # regularization noise
        if regularization_noise is None:
            self.regularization = False
        else:
            self.eta = regularization_noise
            self.regularization = True

        self.n_particle = n_particle
        np.random.seed(seed)
        self.dtype = dtype
        self.log_likelihood = - np.inf
    

    def _norm_likelihood(self, y, mean, covariance):
        """calculate likelihood for Gauss distribution whose parameters
        are `mean` and `covariance`

        Args:
            y [n_dim_obs] {numpy-array, float} 
                observation point which measure likehoodness
                観測 [観測変数軸]
            mean [n_particle, n_dim_obs] {numpy-array, float}
                : mean of Gauss distribution
                各粒子に対する正規分布の平均 [粒子軸，観測変数軸]
            covariance [n_dim_obs, n_dim_obs] {numpy-array, float}
                : covariance of Gauss distribution
                正規分布の共分散 [観測変数軸]
        """
        Y = np.zeros((self.n_dim_obs, self.n_particle), dtype = self.dtype)
        Y.T[:] = y
        return np.exp((- 0.5 * (Y - mean).T @ linalg.pinv(covariance) \
                @ (Y - mean))[:, 0])
    

    def _log_norm_likelihood(self, y, mean, covariance) :
        """calculate log likelihood for Gauss distribution whose parameters
        are `mean` and `covariance`

        Args:
            y [n_dim_obs] {numpy-array, float} 
                observation point which measure likehoodness
                観測 [観測変数軸]
            mean [n_particle, n_dim_obs] {numpy-array, float}
                : mean of Gauss distribution
                各粒子に対する正規分布の平均 [粒子軸，観測変数軸]
            covariance [n_dim_obs, n_dim_obs] {numpy-array, float}
                : covariance of Gauss distribution
                正規分布の共分散 [観測変数軸]
        """
        Y = np.zeros((self.n_dim_obs, self.n_particle), dtype = self.dtype)
        Y.T[:] = y
        return (
            - 0.5 * (Y - mean).T @ linalg.pinv(covariance) @ (Y - mean)
            ).diagonal()


    def _emperical_cummulative_inv(self, w_cumsum, idx, u):
        """calculate inverse map for emperical cummulative function

        Args:
            w_cumsum [n_particle] {numpy-array, float}
                : emperical cummulative function for particles
                粒子の経験分布関数 [粒子軸]
            idx [n_particle] {numpy-array, int}
                : array of ID which are assigined to each particle
                粒子数を持つID配列
            u {float}
                : value between 0 and 1
                (0,1)間の値

        Returns (int):
            like floor function, ID number which maximize set of ID numbers,
            set are less than `u`
        """
        if np.any(w_cumsum < u) == False:
            return 0
        k = np.max(idx[w_cumsum < u])
        return k + 1
        

    def _resampling(self, weights):
        """caluclate standard resampling method
        
        Args:
            weights {numpy-array, float} [n_particle]
                : set of likelihoodness for each particle
                各粒子の尤度(重み)

        Returns:
            k_list {numpy-array, float} [n_particle]
                : index set which represent particle number remeining
        """
        w_cumsum = np.cumsum(weights)

        # generate basic labels
        idx = np.asanyarray(range(self.n_particle))

        # storage for k
        k_list = np.zeros(self.n_particle, dtype = np.int32)
        
        # get index for resampling from weights with uniform distribution
        for i, u in enumerate(rd.uniform(0, 1, size = self.n_particle)):
            k = self._emperical_cummulative_inv(w_cumsum, idx, u)
            k_list[i] = k
        return k_list


    def _stratified_resampling(self, weights):
        """caluclate stratified resampling method
        
        Args:
            weights {numpy-array, float} [n_particle]
                : set of likelihoodness for each particle
                各粒子の尤度(重み)

        Returns:
            k_list {numpy-array, float} [n_particle]
                : index set which represent particle number remeining
        """
        idx = np.asanyarray(range(self.n_particle))
        u0 = rd.uniform(0, 1 / self.n_particle)
        u = [1 / self.n_particle*i + u0 for i in range(self.n_particle)]
        w_cumsum = np.cumsum(weights)
        k = np.asanyarray([
            self._emperical_cummulative_inv(w_cumsum, idx, val) for val in u
            ])
        return k
    

    def filter(self):
        """Calculate prediction and filter for observation times.

        Attributes (self):
            x_pred_mean [n_time+1, n_dim_sys] {numpy-array, float}
                : mean of `x_pred` regarding to particles at time t
                時刻 t における x_pred の粒子平均 [時間軸，状態変数軸]
            x_filt_mean [n_time+1, n_dim_sys] {numpy-array, float}
                : mean of `x_filt` regarding to particles
                時刻 t における状態変数のフィルタ平均 [時間軸，状態変数軸]

        Attributes (local):
            T {int}
                : length of time-series
                時系列の長さ
            x_pred [n_dim_sys, n_particle]
                : hidden state at time t given observations for each particle
                状態変数の予測アンサンブル [状態変数軸，粒子軸]
            x_filt [n_dim_sys, n_particle] {numpy-array, float}
                : hidden state at time t given observations for each particle
                状態変数のフィルタアンサンブル [状態変数軸，粒子軸]
            w [n_particle] {numpy-array, float}
                : weight (likelihoodness) lambda of each particle
                各粒子の尤度 [粒子軸]
            v [n_dim_sys, n_particle] {numpy-array, float}
                : ensemble menbers of system noise
                各時刻の状態ノイズ [状態変数軸，粒子軸]
            k [n_particle] {numpy-array, float}
                : index numbers for resampling
                各時刻のリサンプリングインデックス [粒子軸]
        """

        # length of time-series data
        T = len(self.y)
        
        # initial filter, prediction
        self.x_pred_mean = np.zeros((T + 1, self.n_dim_sys), dtype = self.dtype)
        self.x_filt_mean = np.zeros((T + 1, self.n_dim_sys), dtype = self.dtype)
        x_pred = np.zeros((self.n_dim_sys, self.n_particle), dtype = self.dtype)

        # initial distribution
        x_filt = rd.multivariate_normal(self.initial_mean, self.initial_covariance, 
            size = self.n_particle).T
        
        # initial setting
        self.x_pred_mean[0] = self.initial_mean
        self.x_filt_mean[0] = self.initial_mean

        for t in range(T):
            # visualize calculating times
            print("\r filter calculating... t={}".format(t), end="")
            
            ## filter update
            # calculate prediction step
            f = _last_dims(self.f, t, 1)[0]

            # raise parametric system noise
            v = self.q[0](*self.q[1], size = self.n_particle).T

            # calculate ensemble prediction
            x_pred = f(*[x_filt, v])

            # calculate mean of ensemble prediction
            self.x_pred_mean[t + 1] = np.mean(x_pred, axis = 1)
            
            # treat missing values
            if np.any(np.ma.getmask(self.y[t])):
                x_filt = x_pred
            else:
                # (log) likelihood for each particle for y[t]
                lf = self._last_likelihood(self.lf, t)
                lfp = self._last_likelihood(self.lfp, t)
                try:
                    w = lf(self.y[t], x_pred, *lfp)
                except:
                    raise ValueError("you must check likelihood_functions" 
                        + "and parameters.")
                
                # avoid evaporation
                if self.likelihood_function_is_log_form:
                    w = np.exp(w - np.max(w))
                else:
                    w = w / np.max(w)

                # normalize weights
                w = w / np.sum(w)

                # calculate resampling
                k = self._stratified_resampling(w)
                x_filt = x_pred[:, k]

                # add regularization
                if self.regularization:
                    x_filt += self.eta[0](*self.eta[1], size = self.n_particle).T
            
            # calculate mean of filtering results
            self.x_filt_mean[t + 1] = np.mean(x_filt, axis = 1)

        
    def get_predicted_value(self, dim = None) :
        """Get predicted value

        Args:
            dim {int} : dimensionality for extract from predicted result

        Returns (numpy-array, float)
            : mean of hidden state at time t given observations
            from times [0...t]
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
        """calculate fixed lag smooth. Because of memory saving,
        also describe filtering step

        Args:
            lag {int}
                : lag of smoothing
                平滑化のためのラグ
        
        Attributes (self):
            x_pred_mean [n_time+1, n_dim_sys] {numpy-array, float}
                : mean of `x_pred` regarding to particles at time t
                時刻 t における x_pred の粒子平均 [時間軸，状態変数軸]
            x_filt_mean [n_time+1, n_dim_sys] {numpy-array, float}
                : mean of `x_filt` regarding to particles
                時刻 t における状態変数のフィルタ平均 [時間軸，状態変数軸]
            x_smooth_mean [n_time, n_dim_sys] {numpy-array, float}
                : mean of `x_smooth` regarding to particles at time t
                時刻 t における状態変数の平滑化平均 [時間軸，状態変数軸]

        Attributes (local):
            T {int}
                : length of time-series
                時系列の長さ
            x_pred [n_dim_sys, n_particle]
                : hidden state at time t given observations for each particle
                状態変数の予測アンサンブル [状態変数軸，粒子軸]
            x_filt [n_dim_sys, n_particle] {numpy-array, float}
                : hidden state at time t given observations for each particle
                状態変数のフィルタアンサンブル [状態変数軸，粒子軸]
            x_smooth [n_time, n_dim_sys, n_particle] {numpy-array, float}
                : hidden state at time t given observations[:t+lag] for each particle
                状態変数の平滑化アンサンブル [時間軸，状態変数軸，粒子軸]
            w [n_particle] {numpy-array, float}
                : weight (likelihoodness) lambda of each particle
                各粒子の尤度 [粒子軸]
            v [n_dim_sys, n_particle] {numpy-array, float}
                : ensemble menbers of system noise
                各時刻の状態ノイズ [状態変数軸，粒子軸]
            k [n_particle] {numpy-array, float}
                : index numbers for resampling
                各時刻のリサンプリングインデックス [粒子軸]
        """

        # length of time-series data
        T = len(self.y)
        
        # initial filter, prediction
        self.x_pred_mean = np.zeros((T + 1, self.n_dim_sys), dtype = self.dtype)
        self.x_filt_mean = np.zeros((T + 1, self.n_dim_sys), dtype = self.dtype)
        self.x_smooth_mean = np.zeros((T + 1, self.n_dim_sys), dtype = self.dtype)
        x_pred = np.zeros((self.n_dim_sys, self.n_particle), dtype = self.dtype)
        x_filt = np.zeros((self.n_dim_sys, self.n_particle), dtype = self.dtype)
        x_smooth = np.zeros((T + 1, self.n_dim_sys, self.n_particle),
             dtype = self.dtype)

        # initial distribution
        x_filt = rd.multivariate_normal(self.initial_mean, self.initial_covariance, 
            size = self.n_particle).T
        
        # initial setting
        self.x_pred_mean[0] = self.initial_mean
        self.x_filt_mean[0] = self.initial_mean
        self.x_smooth_mean[0] = self.initial_mean

        for t in range(T):
            print("\r filter and smooth calculating... t={}".format(t), end="")
            
            ## filter update
            # calculate prediction step
            f = _last_dims(self.f, t, 1)[0]

            # raise parametric system noise
            v = self.q[0](*self.q[1], size = self.n_particle).T

            # calculate ensemble prediction
            x_pred = f(*[x_filt, v])

            # calculate mean of predicted values
            self.x_pred_mean[t + 1] = np.mean(x_pred, axis = 1)
            
            # treat missing values
            if np.any(np.ma.getmask(self.y[t])):
                x_filt = x_pred
            else:
                # (log) likelihood for each particle for y[t]
                lf = self._last_likelihood(self.lf, t)
                lfp = self._last_likelihood(self.lfp, t)
                try:
                    w = lf(self.y[t], x_pred, *lfp)
                except:
                    raise ValueError("you must check likelihood_functions"
                        + " and parameters.")
                
                # avoid evaporation
                if self.likelihood_function_is_log_form:
                    w = np.exp(w - np.max(w))
                else:
                    w = w / np.max(w)

                # calculate resampling
                k = self._stratified_resampling(w)
                x_filt = x_pred[:, k]

                # add regularization
                if self.regularization:
                    x_filt += self.eta[0](*self.eta[1], size = self.n_particle).T
            
            # substitute initial smooth value
            x_smooth[t + 1] = x_filt
            
            # calculate mean of filtering results
            self.x_filt_mean[t + 1] = np.mean(x_filt, axis = 1)

            # calculate fixed lag smoothing
            if (t > lag - 1) :
                x_smooth[t - lag:t + 1] = x_smooth[t - lag:t + 1, :, k]
            else :
                x_smooth[:t + 1] = x_smooth[:t + 1, :, k]

        # calculate mean of smoothing results
        self.x_smooth_mean = np.mean(x_smooth, axis = 2)


    def get_smoothed_value(self, dim = None) :
        """Get RTS smoothed value

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


    # last likelihood function and parameters (尤度関数とパラメータの決定)
    def _last_likelihood(self, X, t):
        """Extract the final dimensions of `X`
        Extract the final `ndim` dimensions at index `t` if `X` has >= `ndim` + 1
        dimensions, otherwise return `X`.
        
        Args:
            X : array with at least dimension `ndims`
            t : int
                index to use for the `ndims` + 1th dimension

        Returns:
            Y : array with dimension `ndims`
                the final `ndims` dimensions indexed by `t`
        """
        if self.observation_parameters_time_invariant:
            return X
        else:
            try:
                return X[t]
            except:
                raise ValueError("you must check which likelihood " + 
                    "parameters are time-invariant.")
