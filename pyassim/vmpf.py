"""
===============================
Inference with Particle Filter
===============================
This module implements the Particle Filter and Particle Smoother,
for Nonlinear Non-Gaussian state space models
"""

import math
import itertools
from multiprocessing import Pool
import multiprocessing as multi

import numpy as np

from utils import array1d, array2d, check_random_state, get_params, \
    preprocess_arguments, check_random_state
from util_functions import _parse_observations, _last_dims, \
    _determine_dimensionality, _log_sum_exp



def _calculate_psi(xf, xp, Q_inv):
    return ((xf - xp).reshape(1,-1) @ Q_inv @ (xf - xp))[0]


def _calculate_obs_grad(H, R_inv, innovation):
    return H.T @ R_inv @ innovation



class GaussianVariationalMappingParticleFilter(object):
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
        y, observation [n_timestep, n_dim_obs] {xp-array, float}
            also known as :math:`y`. observation value
            観測値 [時間軸,観測変数軸]
        initial_mean [n_dim_sys] {float} 
            also known as :math:`\mu_0`. initial state mean
            初期状態分布の期待値 [状態変数軸]
        initial_covariance [n_dim_sys, n_dim_sys] {xp-array, float} 
            also known as :math:`\Sigma_0`. initial state covariance
            初期状態分布の共分散行列[状態変数軸，状態変数軸]
        f, transition_functions [n_timestep] {function}
            also known as :math:`f`. transition function from x_{t-1} to x_{t}
            システムモデルの遷移関数 [時間軸] or []
        q, transition_noise [n_timestep - 1] {(method, parameters)}
            also known as :math:`p(v)`. method and parameters of transition
            noise. noise distribution must be parametric and need input variable
            `size`, which mean number of ensemble
            システムノイズの発生方法とパラメータ [時間軸]
            サイズは指定できる形式
        h, observation_functions [n_timestep] {function}
            also known as :math:`h`. observation function from x_{t} to y_{t}
            観測演算子 [時間軸] or []
        eta, regularization_noise [n_timestep - 1] {(method, parameters)}
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
        dtype {self.xp.dtype}
            : dtype of xp-array
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
                transition_functions = None, observation_functions = None,
                observation_grad_functions = None,
                transition_covariance = None, observation_covariance = None,
                epsilon = 1e-2,
                max_iteration = 50, min_grad = 1e-4,
                kernel = "rbf", kernel_parameters = [0.1],
                save_particles = False,
                n_particle = 100, n_dim_sys = None, n_dim_obs = None,
                use_gpu = False, cpu_number = "all",
                dtype = "float64", seed = 10):
        self.use_gpu = use_gpu
        if use_gpu:
            import cupy
            self.xp = cupy
            self.xp_type = "cupy"
        else:
            self.xp = np
            self.xp_type = "numpy"

        # check order of tensor and mask missing values
        self.y = self.xp.array(observation, dtype=dtype)
        self.n_timestep = len(self.y)

        # determine dimensionality
        self.n_dim_sys = _determine_dimensionality(
            [(initial_mean, array1d, -1),
            (initial_covariance, array2d, -2),
            (transition_covariance, array2d, -2)],
            n_dim_sys
        )

        self.n_dim_obs = _determine_dimensionality(
            [(observation, array1d, -1),
            (observation_covariance, array2d, -2)],
            n_dim_obs
        )

        # transition_functions
        # None -> system + noise
        if transition_functions is None:
            self.f = [lambda x: x]
        else:
            self.f = transition_functions

        if observation_functions is None:
            self.h = [lambda x: x]
        else:
            self.h = observation_functions

        if observation_grad_functions is None:
            self.hg = [lambda x: self.xp.eye(len(x))]
        else:
            self.hg = observation_grad_functions

        if transition_covariance is None:
            self.Q = self.xp.eye(self.n_dim_sys, dtype = dtype)
        else:
            self.Q = transition_covariance.astype(dtype)
        if self.Q.ndim==2:
            self.Q_inv = self.xp.linalg.pinv(self.Q)

        if observation_covariance is None:
            self.R = self.xp.eye(self.n_dim_obs, dtype = dtype)
        else:
            self.R = observation_covariance.astype(dtype)
        if self.R.ndim==2:
            self.R_inv = self.xp.linalg.pinv(self.R)

        # initial_mean None -> self.xp.zeros
        if initial_mean is None:
            self.initial_mean = self.xp.zeros(self.n_dim_sys, dtype = dtype)
        else:
            self.initial_mean = initial_mean.astype(dtype)

        # initial_covariance None -> self.xp.eye
        if initial_covariance is None:
            self.initial_covariance = self.xp.eye(self.n_dim_sys, dtype = dtype)
        else:
            self.initial_covariance = initial_covariance.astype(dtype)

        if kernel in ["rbf"]:
            self.kernel = kernel
            self.kernel_parameters = kernel_parameters
        else:
            raise ValueError("kernel {} does not exist.".format(kernel))

        self.epsilon = epsilon
        self.max_iteration = max_iteration
        self.min_grad = min_grad

        self.save_particles = save_particles
        self.n_particle = n_particle
        self.xp.random.seed(seed)
        self.dtype = dtype

        if cpu_number == "all":
            self.cpu_number = multi.cpu_count()
        else:
            self.cpu_number = cpu_number

    

    def forward(self):
        """Calculate prediction and filter for observation times.

        Attributes (self):
            x_pred_mean [n_timestep+1, n_dim_sys] {xp-array, float}
                : mean of `x_pred` regarding to particles at time t
                時刻 t における x_pred の粒子平均 [時間軸，状態変数軸]
            x_filt_mean [n_timestep+1, n_dim_sys] {xp-array, float}
                : mean of `x_filt` regarding to particles
                時刻 t における状態変数のフィルタ平均 [時間軸，状態変数軸]

        Attributes (local):
            T {int}
                : length of time-series
                時系列の長さ
            x_pred [n_dim_sys, n_particle]
                : hidden state at time t given observations for each particle
                状態変数の予測アンサンブル [状態変数軸，粒子軸]
            x_filt [n_dim_sys, n_particle] {xp-array, float}
                : hidden state at time t given observations for each particle
                状態変数のフィルタアンサンブル [状態変数軸，粒子軸]
            w [n_particle] {xp-array, float}
                : weight (likelihoodness) lambda of each particle
                各粒子の尤度 [粒子軸]
            v [n_dim_sys, n_particle] {xp-array, float}
                : ensemble menbers of system noise
                各時刻の状態ノイズ [状態変数軸，粒子軸]
            k [n_particle] {xp-array, float}
                : index numbers for resampling
                各時刻のリサンプリングインデックス [粒子軸]
        """    
        # initial filter, prediction
        self.x_pred_mean = self.xp.zeros((self.n_timestep+1, self.n_dim_sys), dtype = self.dtype)
        self.x_filt_mean = self.xp.zeros((self.n_timestep+1, self.n_dim_sys), dtype = self.dtype)
        # x_pred = self.xp.zeros((self.n_dim_sys, self.n_particle), dtype = self.dtype)

        # initial setting
        self.x_pred_mean[0] = self.initial_mean
        self.x_filt_mean[0] = self.initial_mean

        # initial distribution
        if self.save_particles:
            self.x_pred = self.xp.zeros((self.n_timestep+1, self.n_particle, self.n_dim_sys))
            self.x_filt = self.xp.zeros((self.n_timestep+1, self.n_particle, self.n_dim_sys))
            self.x_pred[0] = self.xp.random.multivariate_normal(self.initial_mean, self.initial_covariance, 
                size = self.n_particle) # (np,Dx)

            for t in range(self.n_timestep):
                # visualize calculating times
                print("\r filter calculating... t={}/{}".format(t, self.n_timestep), end="")
                if t!=0:
                    self.x_pred[t] = self._predict_update(t, self.x_filt[t-1])
                self.x_filt[t] = self._filter_update(t, self.x_pred[t])
        else:
            x_pred = self.xp.random.multivariate_normal(self.initial_mean, self.initial_covariance, 
                size = self.n_particle).T

            for t in range(self.n_timestep):
                # visualize calculating times
                print("\r filter calculating... t={}/{}".format(t, self.n_timestep), end="")
                if t!=0:
                    x_pred = self._predict_update(t, x_filt)
                x_filt = self._filter_update(t, x_pred)
            


    def _predict_update(self, t, x_filt):
        """Calculate predict update
        """
        f = _last_dims(self.f, t-1, 1)[0]
        Q = _last_dims(self.Q, t-1, 2)
        # calculate ensemble prediction
        x_pred = f(x_filt) + self.xp.random.multivariate_normal(self.xp.zeros(self.n_dim_sys), Q)
        # calculate mean of ensemble prediction
        self.x_pred_mean[t+1] = self.xp.mean(x_pred, axis=0)
        return x_pred


    def _filter_update(self, t, x_pred):
        """Calculate filter update
        """
        # treat missing values
        x_filt = x_pred.copy()
        iteration = 0
        while(iteration < self.max_iteration):
            log_prob_grad = self._log_prob_grad(t, x_filt, x_pred)# (np2,Dx)
            gram_matrix, gram_matrix_grad = self._gram_matrix_and_grad(x_filt) # (np1,np2),(np1,np2,Dx)
            gram_matrix = self.xp.repeat(self.xp.expand_dims(gram_matrix,2), self.n_dim_sys, 2) # (np1,np2,Dx)
            # gram_matrix_grad = self._gram_matrix_grad(x_filt) # (np1,np2,Dx)
            KL_grad = - ((gram_matrix * log_prob_grad) + gram_matrix_grad).sum(axis=1) / self.n_particle # (np1,Dx)
            x_filt = x_filt - self.epsilon * KL_grad # (np1,Dx)
            iteration += 1
        
        # calculate mean of filtering results
        self.x_filt_mean[t+1] = self.xp.mean(x_filt, axis=0)
        return x_filt


    def _log_prob_grad(self, t, x_filt, x_pred):
        hg = _last_dims(self.hg, t, 1)[0]
        h = _last_dims(self.h, t, 1)[0]
        H = hg(x_filt) # (npf,Dy,Dx)
        # f = _last_dims(self.f, t, 1)[0]
        # x_pred_expand = self.xp.tile(self.xp.expand_dims(x_pred, 1), (1, self.n_particle, 1)) # (npp,npf,Dx)
        # psi = self.xp.exp(-0.5*(x_filt - x_pred_expand) @ self.Q_inv @ self.xp.outer(x_filt, x_pred_expand))
        if self.Q.ndim==2:
            Q_inv = self.Q_inv
        elif self.Q.ndim==3:
            Q = _last_dims(self.Q, t, 2)
            Q_inv = self.xp.linalg.pinv(Q)
        if self.R.ndim==2:
            R_inv = self.R_inv
        elif self.R.ndim==3:
            R = _last_dims(self.R, t, 2)
            R_inv = self.xp.linalg.pinv(R)


        if self.use_gpu:
            Q_inv = Q_inv.get()
            x_filt = x_filt.get()
            x_pred = x_pred.get()

        p = Pool(self.cpu_number)
        psi = self.xp.array(p.starmap(_calculate_psi, zip(np.repeat(x_filt, self.n_particle, 0),
                                            np.tile(x_pred, (self.n_particle,1)),
                                            itertools.repeat(Q_inv)))).reshape(self.n_particle, self.n_particle) #(npf,npp)
        p.close()
        # print("b:",psi)
        psi = self.xp.exp(-0.5 * psi) # concerns: overflow or underflow
        # psi = -0.5 * psi
        # print("a:",psi)

        # log_prob_grad = H.transpose(0,2,1) @ R_inv @ (self.y[t] - self.h(x_filt).T) # (npf,Dx,npf)
        p = Pool(self.cpu_number)
        obs_grad = self.xp.array(p.starmap(_calculate_obs_grad, zip(H,
                                            itertools.repeat(R_inv),
                                            self.y[t] - h(x_filt)))).reshape(self.n_particle, self.n_dim_sys) #(npf,Dx)
        p.close()
        # Q_inv:(Dx,Dx), xf:(np,Dx), psi-xp:(np,np,Dx)
        transition_grad = - (Q_inv @ (x_filt - ((self.xp.tile(self.xp.expand_dims(psi,2),(1,1,self.n_dim_sys)) * x_pred).sum(axis=1).T
                                                / psi.sum(axis=1)).T).T).T # (npf,Dx)
        # transition_grad = - (Q_inv @ (x_filt - self.xp.exp(_log_sum_exp(self.xp.tile(self.xp.expand_dims(psi,2),(1,1,self.n_dim_sys))
        #                                                                 + self.xp.log(x_pred), axis=1, xp_type=self.xp_type).T
        #                                                 - _log_sum_exp(psi, axis=1)
        #                                                 ).T).T).T # (npf, Dx)

        return obs_grad + transition_grad


    def _gram_matrix_and_grad(self, x):
        # x:(np,Dx)
        if self.kernel=="rbf":
            if self.Q.ndim==2:
                Q_inv = self.Q_inv
            elif self.Q.ndim==3:
                Q = _last_dims(self.Q, t, 2)
                Q_inv = self.xp.linalg.pinv(Q)
            if self.use_gpu:
                Q_inv = Q_inv.get()
                x = x.get()
            A_inv = Q_inv / self.kernel_parameters[0]

            p = Pool(self.cpu_number)
            gram_matrix = self.xp.array(p.starmap(_calculate_psi, zip(np.repeat(x, self.n_particle, 0),
                                                np.tile(x, (self.n_particle,1)),
                                                itertools.repeat(A_inv)))).reshape(self.n_particle, self.n_particle) #(np1,np2)
            p.close()
            gram_matrix = self.xp.exp(-0.5 * gram_matrix) #(np1,np2)

            # print(A_inv.shape)
            # print(np.repeat(x, self.n_particle, 0).shape)
            # print(np.tile(x, (self.n_particle, 1)).shape)
            grad = np.tensordot(A_inv, (np.repeat(x, self.n_particle, 0).reshape(self.n_particle, self.n_particle, self.n_dim_sys)
                    - np.tile(x, (self.n_particle, 1)).reshape(self.n_particle, self.n_particle, self.n_dim_sys)).transpose(2,0,1),
                    axes=(1,0))
            grad = - (gram_matrix * grad).transpose(1,2,0) # (np1,np2,Dx)

        return gram_matrix, grad

            
        
    def get_predicted_value(self, dim=None, get_particles=False):
        """Get predicted value

        Args:
            dim {int} : dimensionality for extract from predicted result

        Returns (xp-array, float)
            : mean of hidden state at time t given observations
            from times [0...t]
        """
        # if not implement `filter`, implement `filter`
        try :
            self.x_pred_mean[0]
        except :
            self.forward()

        if get_particles and self.save_particles:
            if dim is None:
                return self.x_pred[1:]
            elif dim <= self.x_pred.shape[1]:
                return self.x_pred[1:, int(dim)]
            else:
                raise ValueError("The dim must be less than "
                 + self.x_pred.shape[1] + ".")
        else:
            if dim is None:
                return self.x_pred_mean[1:]
            elif dim <= self.x_pred_mean.shape[1]:
                return self.x_pred_mean[1:, int(dim)]
            else:
                raise ValueError("The dim must be less than "
                 + self.x_pred_mean.shape[1] + ".")


    def get_filtered_value(self, dim=None, get_particles=False):
        """Get filtered value

        Args:
            dim {int} : dimensionality for extract from filtered result

        Returns (xp-array, float)
            : mean of hidden state at time t given observations
            from times [0...t]
        """
        # if not implement `filter`, implement `filter`
        try :
            self.x_filt_mean[0]
        except :
            self.forward()

        if get_particles and self.save_particles:
            if dim is None:
                return self.x_filt[1:]
            elif dim <= self.x_filt.shape[1]:
                return self.x_filt[1:,:,int(dim)]
            else:
                raise ValueError("The dim must be less than "
                 + self.x_filt.shape[1] + ".")
        else:
            if dim is None:
                return self.x_filt_mean[1:]
            elif dim <= self.x_filt_mean.shape[1]:
                return self.x_filt_mean[1:, int(dim)]
            else:
                raise ValueError("The dim must be less than "
                 + self.x_filt_mean.shape[1] + ".")


    def _smooth_update(self, t, x_pred, lag):
        # treat missing values
        if self.xp.any(self.xp.isnan(self.y[t])):
            x_filt = x_pred
        else:
            # (log) likelihood for each particle for y[t]
            w = _log_likelihood(self.y[t], x_pred, *lfp) 

            # avoid evaporation
            w = self.xp.exp(w - self.xp.max(w))

            # calculate resampling
            k = _stratified_resampling(w)
            x_filt = x_pred[:, k]

            # add regularization
            if self.regularization:
                x_filt += self.eta[0](*self.eta[1], size = self.n_particle).T
        
        
        return x_filt


    def smooth(self, lag = 10):
        """calculate fixed lag smooth. Because of memory saving,
        also describe filtering step

        Args:
            lag {int}
                : lag of smoothing
                平滑化のためのラグ
        
        Attributes (self):
            x_pred_mean [n_timestep+1, n_dim_sys] {xp-array, float}
                : mean of `x_pred` regarding to particles at time t
                時刻 t における x_pred の粒子平均 [時間軸，状態変数軸]
            x_filt_mean [n_timestep+1, n_dim_sys] {xp-array, float}
                : mean of `x_filt` regarding to particles
                時刻 t における状態変数のフィルタ平均 [時間軸，状態変数軸]
            x_smooth_mean [n_timestep, n_dim_sys] {xp-array, float}
                : mean of `x_smooth` regarding to particles at time t
                時刻 t における状態変数の平滑化平均 [時間軸，状態変数軸]

        Attributes (local):
            x_pred [n_dim_sys, n_particle]
                : hidden state at time t given observations for each particle
                状態変数の予測アンサンブル [状態変数軸，粒子軸]
            x_filt [n_dim_sys, n_particle] {xp-array, float}
                : hidden state at time t given observations for each particle
                状態変数のフィルタアンサンブル [状態変数軸，粒子軸]
            x_smooth [n_timestep, n_dim_sys, n_particle] {xp-array, float}
                : hidden state at time t given observations[:t+lag] for each particle
                状態変数の平滑化アンサンブル [時間軸，状態変数軸，粒子軸]
            w [n_particle] {xp-array, float}
                : weight (likelihoodness) lambda of each particle
                各粒子の尤度 [粒子軸]
            v [n_dim_sys, n_particle] {xp-array, float}
                : ensemble menbers of system noise
                各時刻の状態ノイズ [状態変数軸，粒子軸]
            k [n_particle] {xp-array, float}
                : index numbers for resampling
                各時刻のリサンプリングインデックス [粒子軸]
        """        
        # initial filter, prediction
        self.x_pred_mean = self.xp.zeros((self.n_timestep + 1, self.n_dim_sys), dtype = self.dtype)
        self.x_filt_mean = self.xp.zeros((self.n_timestep + 1, self.n_dim_sys), dtype = self.dtype)
        self.x_smooth_mean = self.xp.zeros((self.n_timestep + 1, self.n_dim_sys), dtype = self.dtype)

        # initial setting
        self.x_pred_mean[0] = self.initial_mean
        self.x_filt_mean[0] = self.initial_mean
        self.x_smooth_mean[0] = self.initial_mean

        if self.save_particles:
            self.x_pred = self.xp.zeros((self.n_timestep + 1, self.n_dim_sys, self.n_particle), dtype = self.dtype)
            self.x_filt = self.xp.zeros((self.n_timestep + 1, self.n_dim_sys, self.n_particle), dtype = self.dtype)
            self.x_smooth = self.xp.zeros((self.n_timestep + 1, self.n_dim_sys, self.n_particle),
                 dtype = self.dtype)

            # initial distribution
            self.x_filt[0] = self.xp.random.multivariate_normal(self.initial_mean, self.initial_covariance, 
                size = self.n_particle).T
            self.x_pred[0] = self.x_filt[0]
            self.x_smooth[0] = self.x_filt[0]

            for t in range(self.n_timestep):
                print("\r filter and smooth calculating... t={}/{}".format(t, self.n_timestep), end="")
                
                ## filter update
                self.x_pred[t+1] = self._predict_update(t, self.x_filt[t])
                self.x_filt[t+1], k = self._filter_update(t, self.x_pred[t+1])
                # x_filt = self._smooth_update(t, x_pred, lag)

                # substitute initial smooth value
                self.x_smooth[t+1] = self.x_filt[t+1]

                # calculate fixed lag smoothing
                if t > lag - 1:
                    self.x_smooth[t-lag:t+1] = self.x_smooth[t-lag:t+1, :, k]
                else :
                    self.x_smooth[:t+1] = self.x_smooth[:t+1, :, k]

            # calculate mean of smoothing results
            self.x_smooth_mean = self.xp.mean(self.x_smooth, axis = 2)
        else:
            x_smooth = self.xp.zeros((self.n_timestep + 1, self.n_dim_sys, self.n_particle),
                 dtype = self.dtype)

            # initial distribution
            x_filt = self.xp.random.multivariate_normal(self.initial_mean, self.initial_covariance, 
                size = self.n_particle).T

            for t in range(self.n_timestep):
                print("\r filter and smooth calculating... t={}/{}".format(t, self.n_timestep), end="")
                
                ## filter update
                x_pred = self._predict_update(t, x_filt)
                x_filt, k = self._filter_update(t, x_pred)
                # x_filt = self._smooth_update(t, x_pred, lag)

                # substitute initial smooth value
                x_smooth[t+1] = x_filt

                # calculate fixed lag smoothing
                if t > lag - 1:
                    x_smooth[t-lag:t+1] = x_smooth[t-lag:t+1, :, k]
                else :
                    x_smooth[:t+1] = x_smooth[:t+1, :, k]

            # calculate mean of smoothing results
            self.x_smooth_mean = self.xp.mean(x_smooth, axis = 2)


    def get_smoothed_value(self, dim=None, get_particles=False) :
        """Get RTS smoothed value

        Args:
            dim {int} : dimensionality for extract from RTS smoothed result

        Returns (xp-array, float)
            : mean of hidden state at time t given observations
            from times [0...T]
        """
        # if not implement `smooth`, implement `smooth`
        try :
            self.x_smooth_mean[0]
        except :
            self.smooth()

        if get_particles and self.save_particles:
            if dim is None:
                return self.x_smooth[1:]
            elif dim <= self.x_smooth.shape[1]:
                return self.x_smooth[1:, int(dim)]
            else:
                raise ValueError("The dim must be less than "
                 + self.x_smooth.shape[1] + ".")
        else:
            if dim is None:
                return self.x_smooth_mean[1:]
            elif dim <= self.x_smooth_mean.shape[1]:
                return self.x_smooth_mean[1:, int(dim)]
            else:
                raise ValueError("The dim must be less than "
                 + self.x_smooth_mean.shape[1] + ".")






