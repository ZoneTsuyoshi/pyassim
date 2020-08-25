'''
gaussian particle filter
'''

import numpy as np
import numpy.random as rd
from scipy import linalg

from .util_functions import _last_dims
from .stats import _log_gaussian_distribution_kernel_Sigma_multi
from .particle import ParticleFilter

class GaussianParticleFilterGauss(ParticleFilter):
    '''
    Gaussian Particle Filter

    <Input Variables>
    y, observation [n_time, n_dim_obs] {numpy-array, float}
        : observation y
    initial_mean [n_dim_sys] {float} 
        : initial state mean
    initial_covariance [n_dim_sys, n_dim_sys] {numpy-array, float} 
        : initial state covariance
    f, transition_functions [n_time] {function}
        : transition function from x_{t-1} to x_t
    q, transition_noise [n_time - 1] {(method, parameters)}
        : transition noise for v_t
    lf, likelihood_functions [n_time] or [] {function}
        : likelihood function between x_t and y_t
    lfp, likelihood_function_parameters [n_time, n_param] or [n_param]
     {numpy-array, float}
        : parameters for likelihood function
    likelihood_function_is_log_form {boolean}
        : likelihood functions are log form?
    observation_parameters_time_invariant {boolean}
        : time-invariantness of observation parameters
    n_particles {int} : number of particles
    n_dim_sys {int} : dimension of system variable
    n_dim_obs {int} : dimension of observation variable
    dtype {np.dtype} : numpy dtype
    seed {int} : random seed

    <Variables>
    regularization {boolean}
        :if True, add regularization term
    '''
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

            # calculate filter update
            x_filt_mean = w @ x_pred.T #(Dx)
            x_centered = (x_pred.T - x_filt_mean).T #(Dx,np)
            V_filt = self.xp.sum(w
                    * self.xp.repeat(x_centered.reshape(1,self.n_dim_sys,self.n_particles),self.n_dim_sys,0)
                    * self.xp.repeat(x_centered.reshape(self.n_dim_sys,1,self.n_particles),self.n_dim_sys,1),
                    axis=2) #(Dx,Dx)
            x_filt = self.xp.random.multivariate_normal(x_filt_mean, V_filt, size=self.n_particles).T #(Dx,np)

            # add regularization
            if self.regularization:
                x_filt += self.eta[0](*self.eta[1], size=self.n_particles).T
        
        # calculate mean of filtering results
        self.x_filt_mean[t] = self.xp.mean(x_filt, axis=1)
        return x_filt, 0
    


