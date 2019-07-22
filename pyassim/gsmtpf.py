"""
================================================================
Inference with Gaussian Soft Markov Transition Particle Filter
================================================================
This module implements the Gaussian Soft Markov Particle Filter,
for Nonlinear Non-Gaussian state space models
"""

import numpy as np
try:
    import cupy
    print("successfully import cupy at smtpf.")
    xp = cupy
    from cupy import linalg
    import cupy.random as rd
except:
    xp = np
    from scipy import linalg
    import numpy.random as rd

from .utils import array1d, array2d, check_random_state, get_params, \
    preprocess_arguments, check_random_state
from .util_functions import _parse_observations, _last_dims, \
    _determine_dimensionality

class GaussianSoftMarkovTransitionParticleFilter(object):
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
        dimensionality_reduction_map {function}
            : map from observation variables to low dimeionality.
            this maps from [n_samples, n_dim_obs] to [n_samples, n_dim_dr]
        cluster_mean [n_dim_sys, n_dim_dr] {numpy-array, float}
            : mean of observation after DR for each cluster
        cluster_inverse_covariance [n_dim_sys, n_dim_dr, n_dim_dr] {numpy-array, float}
            : inverse covariance of observation after DR for each cluster
        initial_distribution {str} 
            initial distribution for discrete state x
            初期状態分布
        initial_parameter [n_dim_sys] {numpy-array, float} 
            parameter for initial distribution
            初期状態分布のパラメーター[状態変数軸]
        F, transition_kernel [n_dim_sys, n_dim_sys] {numpy-array, float}
            also known as :math:`f`. transition function from x_{t-1} to x_{t}
            システムモデルの遷移関数 [時間軸]
        eta, regularization_noise [n_time - 1] {(method, parameters)}
            : noise distribution for regularization. noise distribution
            must be parametric and need ixput variable `size`,
            which mean number of ensemble
            正則化のためのノイズ分布
        n_particle {int}
            : number of particles (ensembles)
            粒子数
        n_dim_sys {int}
            : dimension of system variable. the number of clusters of discrete states.
            システム変数の次元，クラスター数
        n_dim_obs {int}
            : dimension of observation variable
            観測変数の次元
        n_dim_dr {int}
            : dimension of observation variable after dimensionality reduction
        dtype {xp.dtype}
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
                dimensionality_reduction_map = None,
                cluster_mean = None,
                cluster_inverse_covariance = None,
                initial_distribution = "poisson",
                initial_parameter = None,
                transition_kernel = None,
                regularization_noise = None,
                n_particle = 100, n_dim_sys = None, n_dim_obs = None,
                n_dim_dr = None,
                dtype = xp.float32, seed = 10) :
        # check order of tensor and mask missing values
        # self.y = _parse_observations(observation)
        self.y = xp.asarray(observation, dtype = dtype)

        # determine dimensionality
        self.n_dim_sys = _determine_dimensionality(
            [(transition_kernel, array2d, -1)],
            n_dim_sys
        )

        self.n_dim_obs = _determine_dimensionality(
            [(observation, array1d, -1)],
            n_dim_obs
        )

        # dimensionality reduction map
        if dimensionality_reduction_map is None:
            self.dimensionality_reduction_map = lambda x: x
            self.n_dim_dr = self.n_dim_obs
        else:
            self.dimensionality_reduction_map = dimensionality_reduction_map
            ylen_dr = self.dimensionality_reduction_map(self.y[0].reshape(1,-1)).shape[1]
            if n_dim_dr is None:
                self.n_dim_dr = ylen_dr
            elif ylen_dr != self.n_dim_dr:
                raise ValueError("Dimension of observation value after DR must be {}.\n".format(self.n_dim_dr)
                    + "However, your DR reduction is mapping from {}d to {}d.".format(self.n_dim_obs,
                        ylen_dr))

        # transition_kernel
        # None -> uniform transition kernel
        if transition_kernel is None:
            self.F = xp.ones((self.n_dim_sys, self.n_dim_sys), dtype = dtype) / self.n_dim_sys
        else:
            self.F = transition_kernel

        # initial_mean None -> xp.zeros
        if initial_distribution in ["poisson", "gamma"]:
            self.initial_distribution = initial_distribution
        else:
            raise ValueError("Initial distribution \"" + initial_distribution + "\" can't use in "
                + "this program. \n You can only use \"poisson\" or \"gamma\".")

        # initial_covariance None -> xp.eye
        if initial_parameter is None:
            if initial_distribution == "poisson":
                self.initial_parameter = 5 * xp.eye(self.n_dim_sys, dtype = dtype)
            elif initial_distribution == "gamma":
                self.initial_parameter = xp.asarray([2.0 * xp.ones(self.n_dim_sys, dtype=dtype),
                                        2.0 * xp.ones(self.n_dim_sys, dtype=dtype)])
        else:
            self.initial_parameter = initial_parameter

        # cluster_mean None -> ValueError
        if cluster_mean is None:
            raise ValueError("Mean of each cluster \"cluster_mean\" must be input.")
        else:
            if cluster_mean.shape == (self.n_dim_sys, self.n_dim_dr):
                self.cluster_mean = cluster_mean
            else:
                raise ValueError("Shape of \"cluster_mean\" must be ({},{}).\n".format(self.n_dim_sys,
                                                                                    self.n_dim_dr)
                    + "However, your shape of \"cluster_mean\" is " + cluster_mean.shape + ".")

        # cluster_inverse_covariance None -> ValueError
        if cluster_inverse_covariance is None:
            raise ValueError("Mean of each cluster \"cluster_covariance\" must be input.")
        else:
            if cluster_inverse_covariance.shape == (self.n_dim_sys, self.n_dim_dr, self.n_dim_dr):
                self.cluster_inverse_covariance = cluster_inverse_covariance
            else:
                raise ValueError("Shape of \"cluster_covariance\" must be ({},{},{}).\n".format(
                                                                                    self.n_dim_sys,
                                                                                    self.n_dim_dr,
                                                                                    self.n_dim_dr)
                    + "However, your shape of \"cluster_covariance\" is " + cluster_inverse_covariance.shape + ".")

        # regularization noise
        if regularization_noise is None:
            self.regularization = False
        else:
            self.eta = regularization_noise
            self.regularization = True

        self.n_particle = n_particle
        xp.random.seed(seed)
        self.dtype = dtype
        self.log_likelihood = - xp.inf
    

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
        Y = xp.zeros((self.n_dim_obs, self.n_particle), dtype = self.dtype)
        Y.T[:] = y
        return xp.exp((- 0.5 * (Y - mean).T @ linalg.pinv(covariance) \
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
        Y = xp.zeros((self.n_dim_dr, self.n_particle), dtype = self.dtype)
        Y.T[:] = y
        return (
            - 0.5 * (Y - mean).T @ linalg.pinv(covariance) @ (Y - mean)
            ).diagonal()


    def _log_norm_likelihood2(self, y, mean, inverse_covariance) :
        """calculate log likelihood for Gauss distribution whose parameters
        are `mean` and `covariance`

        Args:
            y [n_dim_obs] {numpy-array, float} 
                observation point which measure likehoodness
                観測 [観測変数軸]
            mean [n_particle, n_dim_obs] {numpy-array, float}
                : mean of Gauss distribution
                各粒子に対する正規分布の平均 [粒子軸，観測変数軸]
            inverse_covariance [n_dim_obs, n_dim_obs] {numpy-array, float}
                : inverse covariance of Gauss distribution
                正規分布の逆共分散 [観測変数軸]
        """
        # Y = xp.zeros((self.n_dim_dr, self.n_particle), dtype = self.dtype)
        # Y.T[:] = y
        return - 0.5 * (y - mean).T @ inverse_covariance @ (y - mean)


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
        if xp.any(w_cumsum < u) == False:
            return 0
        k = xp.max(idx[w_cumsum < u])
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
        w_cumsum = xp.cumsum(weights)

        # generate basic labels
        idx = xp.asanyarray(range(self.n_particle))

        # storage for k
        k_list = xp.zeros(self.n_particle, dtype = xp.int32)
        
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
        idx = xp.asanyarray(range(self.n_particle))
        u0 = rd.uniform(0, 1 / self.n_particle)
        u = [1 / self.n_particle*i + u0 for i in range(self.n_particle)]
        w_cumsum = xp.cumsum(weights)
        k = xp.asanyarray([
            self._emperical_cummulative_inv(w_cumsum, idx, val) for val in u
            ])
        return k
    

    def forward(self):
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
        self.x_pred_mean = xp.zeros((T + 1, self.n_dim_sys), dtype = self.dtype)
        self.x_filt_mean = xp.zeros((T + 1, self.n_dim_sys), dtype = self.dtype)
        x_pred = xp.zeros((self.n_dim_sys, self.n_particle), dtype = self.dtype)
        x_filt = xp.zeros((self.n_dim_sys, self.n_particle), dtype = self.dtype)

        # initial distribution
        # ToDo : this calculation cost is big
        # sys x particle
        if self.initial_distribution == "poisson":
            for i in range(self.n_dim_sys):
                x_filt[i] = rd.poisson(self.initial_parameter[i], size = self.n_particle)
        elif self.initial_distribution == "gamma":
            for i in range(self.n_dim_sys):
                x_filt[i] = rd.gamma(self.initial_parameter[0,i], self.initial_parameter[1,i],
                                    size = self.n_particle)

        # standardization
        x_filt = x_filt / xp.sum(x_filt, axis=0)
        
        # initial setting
        self.x_pred_mean[0] = xp.mean(x_filt, axis=1)
        self.x_filt_mean[0] = self.x_pred_mean[0]

        for t in range(T):
            # visualize calculating times
            print("\r filter calculating... t={}/{}\n".format(t, T), end="")
            
            ## filter update
            # calculate ensemble prediction
            # s x p = s x s   s x p
            x_pred = self.F @ x_filt

            # calculate mean of ensemble prediction
            self.x_pred_mean[t + 1] = xp.mean(x_pred, axis = 1)
            
            # treat missing values
            if xp.any(xp.isnan(self.y[t])):
                x_filt = x_pred
            else:
                # log likelihood for each cluster
                w_cluster = xp.zeros(self.n_dim_sys, dtype=self.dtype)
                y_dr = self.dimensionality_reduction_map(self.y[t])
                for i in range(self.n_dim_sys):
                    w_cluster[i] = self._log_norm_likelihood2(y_dr, self.cluster_mean[i],
                        self.cluster_inverse_covariance[i])

                # avoid overflow
                w_cluster = xp.exp(w_cluster - xp.max(w_cluster))

                # normalize weithts
                w_cluster = w_cluster / xp.sum(w_cluster)

                # log likelihood for each particle for y[t]
                w = x_filt.T @ w_cluster

                # normalize weights
                w = w / xp.sum(w)

                # calculate resampling
                k = self._stratified_resampling(w)
                print(len(xp.unique(k)))
                x_filt = x_pred[:, k]

                # add regularization
                if self.regularization:
                    x_filt += self.eta[0](*self.eta[1], size = self.n_particle).T
            
            # calculate mean of filtering results
            self.x_filt_mean[t + 1] = xp.mean(x_filt, axis = 1)

        
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


    def update_transition_matrices(self):
        """Update Markov transition kernel by minimum norm solution.
        """
        self.F = self.x_filt_mean[1:].T @ linalg.pinv(self.x_filt_mean[:-1])


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
        self.x_pred_mean = xp.zeros((T + 1, self.n_dim_sys), dtype = self.dtype)
        self.x_filt_mean = xp.zeros((T + 1, self.n_dim_sys), dtype = self.dtype)
        self.x_smooth_mean = xp.zeros((T + 1, self.n_dim_sys), dtype = self.dtype)
        x_pred = xp.zeros((self.n_dim_sys, self.n_particle), dtype = self.dtype)
        x_filt = xp.zeros((self.n_dim_sys, self.n_particle), dtype = self.dtype)
        x_smooth = xp.zeros((T + 1, self.n_dim_sys, self.n_particle),
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
            self.x_pred_mean[t + 1] = xp.mean(x_pred, axis = 1)
            
            # treat missing values
            if xp.any(xp.ma.getmask(self.y[t])):
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
                    w = xp.exp(w - xp.max(w))
                else:
                    w = w / xp.max(w)

                # calculate resampling
                k = self._stratified_resampling(w)
                x_filt = x_pred[:, k]

                # add regularization
                if self.regularization:
                    x_filt += self.eta[0](*self.eta[1], size = self.n_particle).T
            
            # substitute initial smooth value
            x_smooth[t + 1] = x_filt
            
            # calculate mean of filtering results
            self.x_filt_mean[t + 1] = xp.mean(x_filt, axis = 1)

            # calculate fixed lag smoothing
            if (t > lag - 1) :
                x_smooth[t - lag:t + 1] = x_smooth[t - lag:t + 1, :, k]
            else :
                x_smooth[:t + 1] = x_smooth[:t + 1, :, k]

        # calculate mean of smoothing results
        self.x_smooth_mean = xp.mean(x_smooth, axis = 2)


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
