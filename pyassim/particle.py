"""
===============================
Inference with Particle Filter
===============================
This module implements the Particle Filter and Particle Smoother,
for Nonlinear Non-Gaussian state space models
"""

import numpy as np

from .utils import array1d, array2d, check_random_state, get_params, \
    preprocess_arguments, check_random_state
from .util_functions import _parse_observations, _last_dims, \
    _determine_dimensionality
from .stats import _log_gaussian_distribution_kernel_Sigma_multi, \
    _stratified_resampling


class ParticleFilter(object):
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
        initial_mean [n_dim_sys] {float} 
            also known as :math:`\mu_0`. initial state mean
        initial_covariance [n_dim_sys, n_dim_sys] {xp-array, float} 
            also known as :math:`\Sigma_0`. initial state covariance
        f, transition_functions [n_timestep] {function}
            also known as :math:`f`. transition function from x_{t-1} to x_{t}
        q, transition_noise [n_timestep - 1] {(method, parameters)}
            also known as :math:`p(v)`. method and parameters of transition
            noise. noise distribution must be parametric and need input variable
            `size`, which mean number of ensemble
        h, observation_functions [n_timestep] {function}
            also known as :math:`h`. observation function from x_{t} to y_{t}
        eta, regularization_noise [n_timestep - 1] {(method, parameters)}
            : noise distribution for regularization. noise distribution
            must be parametric and need input variable `size`,
            which mean number of ensemble
        n_particles {int}
            : number of particles (ensembles)
        n_dim_sys {int}
            : dimension of system variable
        n_dim_obs {int}
            : dimension of observation variable
        dtype {self.xp.dtype}
            : dtype of xp-array
        seed {int}
            : random seed

    Attributes:
        regularization {boolean}
            : which particle filter has regularization. If true,
            after filtering step, add state variables to regularization noise
            because of protecting from degeneration of particle.
            If false, doesn't add regularization noise.
    """
    def __init__(self, observation = None, 
                transition_functions = None, observation_functions = None,
                initial_mean = None, initial_covariance = None,
                transition_noise = None,
                save_particles = False,
                regularization_noise = None,
                n_particles = 100, n_dim_sys = None, n_dim_obs = None,
                use_gpu = False,
                dtype = "float64", seed = 10):
        self.use_gpu = use_gpu
        if use_gpu:
            import cupy
            self.xp = cupy
        else:
            self.xp = np

        # check order of tensor and mask missing values
        self.y = self.xp.array(observation, dtype=dtype)
        self.n_timestep = len(self.y)

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
            self.q = (self.xp.random.multivariate_normal,
                [self.xp.zeros(self.n_dim_sys, dtype = dtype),
                self.xp.eye(self.n_dim_sys, dtype = dtype)])
        else:
            self.q = transition_noise


        if observation_functions is None:
            self.h = [lambda x: x]
        else:
            self.h = observation_functions

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

        # regularization noise
        if regularization_noise is None:
            self.regularization = False
        else:
            self.eta = regularization_noise
            self.regularization = True

        self.save_particles = save_particles
        self.n_particles = n_particles
        self.xp.random.seed(seed)
        self.dtype = dtype


    def _predict_update(self, t, x_filt):
        """Calculate predict update
        """
        f = _last_dims(self.f, t, 1)[0]
        # raise parametric system noise
        v = self.q[0](*self.q[1], size=self.n_particles).T
        # calculate ensemble prediction
        x_pred = f(*[x_filt, v])
        # calculate mean of ensemble prediction
        self.x_pred_mean[t] = self.xp.mean(x_pred, axis=1)
        return x_pred


    def _filter_update(self, t, x_pred):
        """Calculate filter update
        """
        # treat missing values
        if self.xp.any(self.xp.isnan(self.y[t])):
            x_filt = x_pred
        else:
            w = _log_likelihood(self.y[t], x_pred)
            
            # avoid evaporation
            w = self.xp.exp(w - self.xp.max(w))

            # normalize weights
            w = w / self.xp.sum(w)

            # calculate resampling
            k = self._stratified_resampling(w)
            x_filt = x_pred[:, k]

            # add regularization
            if self.regularization:
                x_filt += self.eta[0](*self.eta[1], size=self.n_particles).T
        
        # calculate mean of filtering results
        self.x_filt_mean[t] = self.xp.mean(x_filt, axis=1)
        return x_filt, k

    

    def forward(self):
        """Calculate prediction and filter for observation times.

        Attributes (self):
            x_pred_mean [n_timestep+1, n_dim_sys] {xp-array, float}
                : mean of `x_pred` regarding to particles at time t
            x_filt_mean [n_timestep+1, n_dim_sys] {xp-array, float}
                : mean of `x_filt` regarding to particles

        Attributes (local):
            T {int}
                : length of time-series
            x_pred [n_dim_sys, n_particles]
                : hidden state at time t given observations for each particle
            x_filt [n_dim_sys, n_particles] {xp-array, float}
                : hidden state at time t given observations for each particle
            w [n_particles] {xp-array, float}
                : weight (likelihoodness) lambda of each particle
            v [n_dim_sys, n_particles] {xp-array, float}
                : ensemble menbers of system noise
            k [n_particles] {xp-array, float}
                : index numbers for resampling
        """    
        # initial filter, prediction
        self.x_pred_mean = self.xp.zeros((self.n_timestep, self.n_dim_sys), dtype = self.dtype)
        self.x_filt_mean = self.xp.zeros((self.n_timestep, self.n_dim_sys), dtype = self.dtype)
        # x_pred = self.xp.zeros((self.n_dim_sys, self.n_particles), dtype = self.dtype)

        # initial setting
        self.x_pred_mean[0] = self.initial_mean
        self.x_filt_mean[0] = self.initial_mean

        # initial distribution
        if self.save_particles:
            self.x_pred = self.xp.zeros((self.n_timestep, self.n_dim_sys, self.n_particles))
            self.x_filt = self.xp.zeros((self.n_timestep, self.n_dim_sys, self.n_particles))
            self.x_pred[0] = self.xp.random.multivariate_normal(self.initial_mean, self.initial_covariance, 
                size=self.n_particles).T
            self.x_filt[0], _ = self._filter_update(0, self.x_pred[0])

            for t in range(1,self.n_timestep):
                # visualize calculating times
                print("\r filter calculating... t={}/{}".format(t+1, self.n_timestep), end="")
                self.x_pred[t] = self._predict_update(t, self.x_filt[t-1])
                self.x_filt[t], _ = self._filter_update(t, self.x_pred[t])
        else:
            x_pred = self.xp.random.multivariate_normal(self.initial_mean, self.initial_covariance, 
                size=self.n_particles).T
            x_filt, _ = self._filter_update(0, x_pred)

            for t in range(1,self.n_timestep):
                # visualize calculating times
                print("\r filter calculating... t={}/{}".format(t+1, self.n_timestep), end="")
                x_pred = self._predict_update(t, x_filt)
                x_filt, _ = self._filter_update(t, x_pred)
            
            
        
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
                return self.x_pred
            elif dim <= self.x_pred.shape[1]:
                return self.x_pred[:, int(dim)]
            else:
                raise ValueError("The dim must be less than "
                 + self.x_pred.shape[1] + ".")
        else:
            if dim is None:
                return self.x_pred_mean
            elif dim <= self.x_pred_mean.shape[1]:
                return self.x_pred_mean[:, int(dim)]
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
                return self.x_filt
            elif dim <= self.x_filt.shape[1]:
                return self.x_filt[:, int(dim)]
            else:
                raise ValueError("The dim must be less than "
                 + self.x_filt.shape[1] + ".")
        else:
            if dim is None:
                return self.x_filt_mean
            elif dim <= self.x_filt_mean.shape[1]:
                return self.x_filt_mean[:, int(dim)]
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
                x_filt += self.eta[0](*self.eta[1], size = self.n_particles).T
        
        
        return x_filt


    def smooth(self, lag=10):
        """calculate fixed lag smooth. Because of memory saving,
        also describe filtering step

        Args:
            lag {int}
                : lag of smoothing
        
        Attributes (self):
            x_pred_mean [n_timestep+1, n_dim_sys] {xp-array, float}
                : mean of `x_pred` regarding to particles at time t
            x_filt_mean [n_timestep+1, n_dim_sys] {xp-array, float}
                : mean of `x_filt` regarding to particles
            x_smooth_mean [n_timestep, n_dim_sys] {xp-array, float}
                : mean of `x_smooth` regarding to particles at time t

        Attributes (local):
            x_pred [n_dim_sys, n_particles]
                : hidden state at time t given observations for each particle
            x_filt [n_dim_sys, n_particles] {xp-array, float}
                : hidden state at time t given observations for each particle
            x_smooth [n_timestep, n_dim_sys, n_particles] {xp-array, float}
                : hidden state at time t given observations[:t+lag] for each particle
            w [n_particles] {xp-array, float}
                : weight (likelihoodness) lambda of each particle
            v [n_dim_sys, n_particles] {xp-array, float}
                : ensemble menbers of system noise
            k [n_particles] {xp-array, float}
                : index numbers for resampling
        """        
        # initial filter, prediction
        self.x_pred_mean = self.xp.zeros((self.n_timestep, self.n_dim_sys), dtype = self.dtype)
        self.x_filt_mean = self.xp.zeros((self.n_timestep, self.n_dim_sys), dtype = self.dtype)
        self.x_smooth_mean = self.xp.zeros((self.n_timestep, self.n_dim_sys), dtype = self.dtype)

        # initial setting
        self.x_pred_mean[0] = self.initial_mean
        self.x_filt_mean[0] = self.initial_mean
        self.x_smooth_mean[0] = self.initial_mean

        if self.save_particles:
            self.x_pred = self.xp.zeros((self.n_timestep, self.n_dim_sys, self.n_particles), dtype = self.dtype)
            self.x_filt = self.xp.zeros((self.n_timestep, self.n_dim_sys, self.n_particles), dtype = self.dtype)
            self.x_smooth = self.xp.zeros((self.n_timestep, self.n_dim_sys, self.n_particles),
                 dtype = self.dtype)

            # initial distribution
            self.x_pred[0] = self.xp.random.multivariate_normal(self.initial_mean, self.initial_covariance, 
                size=self.n_particles).T
            self.x_filt[0], _ = self._filter_update(0, self.x_pred[0])
            self.x_smooth[0] = self.x_filt[0]

            for t in range(1,self.n_timestep):
                print("\r filter and smooth calculating... t={}/{}".format(t, self.n_timestep), end="")
                
                ## filter update
                self.x_pred[t] = self._predict_update(t, self.x_filt[t-1])
                self.x_filt[t], k = self._filter_update(t, self.x_pred[t])
                # x_filt = self._smooth_update(t, x_pred, lag)

                # substitute initial smooth value
                self.x_smooth[t] = self.x_filt[t]

                # calculate fixed lag smoothing
                if t > lag - 1:
                    self.x_smooth[t-lag:t] = self.x_smooth[t-lag:t, :, k]
                else :
                    self.x_smooth[:t] = self.x_smooth[:t, :, k]

            # calculate mean of smoothing results
            self.x_smooth_mean = self.xp.mean(self.x_smooth, axis=2)
        else:
            x_smooth = self.xp.zeros((lag+1, self.n_dim_sys, self.n_particles),
                 dtype = self.dtype)

            # initial distribution
            x_pred = self.xp.random.multivariate_normal(self.initial_mean, self.initial_covariance, 
                size = self.n_particles).T
            x_filt = self._filter_update(0, self.x_pred[0])
            x_smooth[-1] = x_filt

            for t in range(1,self.n_timestep):
                print("\r filter and smooth calculating... t={}/{}".format(t, self.n_timestep), end="")
                
                ## filter update
                x_pred = self._predict_update(t, x_filt)
                x_filt, k = self._filter_update(t, x_pred)
                # x_filt = self._smooth_update(t, x_pred, lag)

                # substitute initial smooth value
                x_smooth[:-1] = x_smooth[1:]
                x_smooth[-1] = x_filt

                # calculate fixed lag smoothing
                x_smooth[:-1] = x_smooth[:-1,:,k]

                # calculate mean of smoothing results
                self.x_smooth_mean[max(t-lag,0)] = self.xp.mean(x_smooth[0], axis=1)
            self.x_smooth_mean[-lag-1:] = self.xp.mean(x_smooth, axis=2)


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
                return self.x_smooth
            elif dim <= self.x_smooth.shape[1]:
                return self.x_smooth[:, int(dim)]
            else:
                raise ValueError("The dim must be less than "
                 + self.x_smooth.shape[1] + ".")
        else:
            if dim is None:
                return self.x_smooth_mean
            elif dim <= self.x_smooth_mean.shape[1]:
                return self.x_smooth_mean[:, int(dim)]
            else:
                raise ValueError("The dim must be less than "
                 + self.x_smooth_mean.shape[1] + ".")




class ParticleFilterGaussian(ParticleFilter):
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
        R, observation_covariance [n_timestep, n_dim_obs, n_dim_obs] {xp-array, float}
            : covariance of Gaussian likelihood function
    """
    def __init__(self, observation = None, 
                transition_functions = None, 
                observation_functions = None,
                initial_mean = None, initial_covariance = None,
                transition_noise = None,
                observation_covariance = None,
                save_particles = None,
                regularization_noise = None,
                n_particles = 100, n_dim_sys = None, n_dim_obs = None,
                use_gpu = False,
                dtype = "float64", seed = 10):
        ParticleFilter.__init__(self, observation, 
            transition_functions, observation_functions,
            initial_mean, initial_covariance, transition_noise,
            save_particles, regularization_noise,
            n_particles, n_dim_sys, n_dim_obs, use_gpu, dtype, seed)

        if observation_covariance is None:
            self.observation_covariance = self.xp.eye(self.n_dim_obs, dtype=self.dtype)
        elif observation_covariance.ndim == 2:
            if observation_covariance.shape == (self.n_dim_obs, self.n_dim_obs):
                self.R = self.xp.array(observation_covariance, dtype=self.dtype)
            else:
                raise ValueError("Shape of observation covariance must be ({},{}), ".format(self.n_dim_obs,
                                                                                        self.n_dim_obs)
                                + "however, shape of correspondence is ({}).".format(observation_covariance.shape))
            self._log_likelihood = _log_gaussian_distribution_kernel_Sigma_multi(self.R)
            self._observation_covariance_time_invariant = True
        elif observation_covariance.ndim == 3:
            if observation_covariance.shape == (self.n_timestep, self.n_dim_obs, self.n_dim_obs):
                self.R = self.xp.array(observation_covariance, dtype=self.dtype)
            else:
                raise ValueError("Shape of observation covariance must be ({},{},{}), ".format(self.n_timestep,
                                                                                        self.n_dim_obs,
                                                                                        self.n_dim_obs)
                                + "however, shape of correspondence is ({}).".format(observation_covariance.shape))
            self._observation_covariance_time_invariant = False
                


    def _filter_update(self, t, x_pred):
        """Calculate filter update
        """
        # treat missing values
        if self.xp.any(self.xp.isnan(self.y[t])):
            x_filt = x_pred
        else:
            h = _last_dims(self.h, t, 1)[0]
            if self._observation_covariance_time_invariant:
                _log_likelihood = self._log_likelihood
            else:
                _log_likelihood = _log_gaussian_distribution_kernel_Sigma_multi(self.R[t])

            w = _log_likelihood(h(x_pred.T), self.y[t])
            
            # avoid evaporation
            w = self.xp.exp(w - self.xp.max(w))

            # normalize weights
            w = w / self.xp.sum(w)

            # calculate resampling
            k = _stratified_resampling(w, self.xp)
            x_filt = x_pred[:, k]

            # add regularization
            if self.regularization:
                x_filt += self.eta[0](*self.eta[1], size=self.n_particles).T
        
        # calculate mean of filtering results
        self.x_filt_mean[t] = self.xp.mean(x_filt, axis=1)
        return x_filt, k



class ParticleFilterPoisson(ParticleFilter):
    """Poisson
    """
    def __init__(self, observation = None, 
                transition_functions = None, 
                observation_functions = None,
                initial_mean = None, initial_covariance = None,
                transition_noise = None,                
                save_particles = None,
                regularization_noise = None,
                n_particles = 100, n_dim_sys = None, n_dim_obs = None,
                use_gpu = False,
                dtype = "float64", seed = 10):
        ParticleFilter.__init__(self, observation, 
            transition_functions, observation_functions,
            initial_mean, initial_covariance, transition_noise,
            save_particles, regularization_noise,
            n_particles, n_dim_sys, n_dim_obs, use_gpu, dtype, seed)
                


    def _filter_update(self, t, x_pred):
        """Calculate filter update
        """
        # treat missing values
        if self.xp.any(self.xp.isnan(self.y[t])):
            x_filt = x_pred
        else:
            h = _last_dims(self.h, t, 1)[0]
            w = _log_poisson_kernel(self.y[t], h(x_pred.T))
            
            # avoid evaporation
            w = self.xp.exp(w - self.xp.max(w))

            # normalize weights
            w = w / self.xp.sum(w)

            # calculate resampling
            k = _stratified_resampling(w, self.xp)
            x_filt = x_pred[:, k]

            # add regularization
            if self.regularization:
                x_filt += self.eta[0](*self.eta[1], size = self.n_particles).T
        
        # calculate mean of filtering results
        self.x_filt_mean[t + 1] = self.xp.mean(x_filt, axis = 1)
        return x_filt, k



