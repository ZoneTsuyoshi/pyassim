# local ensemble transform kalman filter


# install packages
import math
import itertools

import numpy as np
import numpy.random as rd
# import pandas as pd

from numpy import linalg

from multiprocessing import Pool
import multiprocessing as multi

from .utils import array1d, array2d, check_random_state, get_params, \
    preprocess_arguments, check_random_state
from .util_functions import _parse_observations, _last_dims, \
    _determine_dimensionality



# map global to local
def global_to_local(adjacency_matrix, global_variable):
    '''
    adjacency_matrix [n_dim] {numpy-array, float}
        : adjacency matrix from global to local
    global_variable [n_dim] {numpy-array, float}
        : global variable
    '''
    return global_variable[adjacency_matrix]


# parallel processing for local calculation
def local_multi_processing(x_pred_mean_local, x_pred_center_local,
        y_background_mean_local, y_background_center_local, y_local,
        R_local, i, A_sys, n_particle, rho):
    ## Step4 : calculate matrix C
    # particle x local_obs
    C = y_background_center_local.T @ linalg.pinv(R_local)


    ## Step5 : calculate analysis error covariance in ensemble space
    # particle x particle
    analysis_error_covariance = linalg.pinv(
        (n_particle - 1) / rho * np.eye(n_particle) \
        + C @ y_background_center_local
        )


    ## Step6 : calculate analysis weight matrix in ensemble space
    # particle x particle
    analysis_weight_matrix = linalg.sqrtm(
        (n_particle - 1) * analysis_error_covariance
        )


    # Step7 : calculate analysis weight ensemble
    # particle
    analysis_weight_mean = analysis_error_covariance @ C @ (
        (y_local - y_background_center_local.T).T
        )

    # particle x particle
    analysis_weight_ensemble = (analysis_weight_matrix.T + analysis_weight_mean).T


    ## Step8 : calculate analysis system variable in model space
    # local_sys x particle
    analysis_system = (x_pred_mean_local + (
        x_pred_center_local @ analysis_weight_ensemble
        ).T).T


    ## Step9 : move analysis result to global analysis
    # sys x particle
    return analysis_system[len(np.where(A_sys[:i])[0])]


class LocalEnsembleTransformKalmanFilter(object):
    '''
    Local Ensemble Transform Kalman Filter

    <Input Variables>
    y, observation [n_time, n_dim_obs] {numpy-array, float}
        : observation y
    initial_mean [n_dim_sys] {float} 
        : initial state mean
    f, transition_functions [n_time] {function}
        : transition function from x_{t-1} to x_t
    h, observation_functions [n_time] {function}
        : observation function from x_t to y_t
    q, transition_noise [n_time - 1] {(method, parameters)}
        : transition noise for v_t
    R, observation_covariance [n_time, n_dim_obs, n_dim_obs] {numpy-array, float} 
        : covariance of observation normal noise
    A_sys, system_adjacency_matrix [n_dim_sys, n_dim_sys] {numpy-array, int}
        : adjacency matrix of system variables
    A_obs, observation_adjacency_matrix [n_dim_obs, n_dim_obs] {numpy-array, int}
        : adjacency matrix of system variables
    rho {float} : multipliative covariance inflating factor
    n_particle {int} : number of particles
    n_dim_sys {int} : dimension of system variable
    n_dim_obs {int} : dimension of observation variable
    dtype {np.dtype} : numpy dtype
    seed {int} : random seed
    cpu_number {int} : cpu number for parallel processing
    '''

    def __init__(self, observation = None, transition_functions = None,
                observation_functions = None, initial_mean = None,
                transition_noise = None, observation_covariance = None,
                system_adjacency_matrix = None, observation_adjacency_matrix = None,
                rho = 1,
                n_particle = 100, n_dim_sys = None, n_dim_obs = None,
                dtype = np.float32, seed = 10, cpu_number = 'all') :

        self.y = _parse_observations(observation)

        self.n_dim_sys = _determine_dimensionality(
            [(initial_mean, array1d, -1)],
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

        # observation_matrices
        # None -> np.eye
        if observation_functions is None:
            self.h = [lambda x : x]
        else:
            self.h = observation_functions

        # transition_noise
        # None -> standard normal distribution
        if transition_noise is None:
            self.q = (rd.multivariate_normal,
                [np.zeros(self.n_dim_sys, dtype = dtype),
                np.eye(self.n_dim_sys, dtype = dtype)])
        else:
            self.q = transition_noise

        # observation_covariance
        # None -> np.eye
        if observation_covariance is None:
            self.R = np.eye(self.n_dim_obs, dtype = dtype)
        else:
            self.R = observation_covariance.astype(dtype)

        # initial_mean None -> np.zeros
        if initial_mean is None:
            self.initial_mean = np.zeros(self.n_dim_sys, dtype = dtype)
        else:
            self.initial_mean = initial_mean.astype(dtype)

        # system_adjacency_matrix None -> np.eye
        if system_adjacency_matrix is None:
            self.A_sys = np.eye(self.n_dim_sys).astype(bool)
        else:
            self.A_sys = system_adjacency_matrix.astype(bool)

        # observation_adjacency_matrix is None -> np.eye
        if observation_adjacency_matrix is None:
            self.A_obs = np.eye(self.n_dim_obs).astype(bool)
        else:
            self.A_obs = observation_adjacency_matrix.astype(bool)

        self.rho = rho
        self.n_particle = n_particle
        np.random.seed(seed)
        self.seed = seed
        self.dtype = dtype
        if cpu_number == 'all':
            self.cpu_number = multi.cpu_count()
        else:
            self.cpu_number = cpu_number


    # filtering step
    def filter(self):
        '''
        T {int} : length of data y
        <class variables>
        x_pred_mean [n_time+1, n_dim_sys] {numpy-array, float}
            : mean of x_pred regarding to particles at time t
        x_filt [n_time+1, n_dim_sys, n_particle] {numpy-array, float}
            : hidden state at time t given observations for each particle
        x_filt_mean [n_time+1, n_dim_sys] {numpy-array, float}
            : mean of x_filt regarding to particles

        <global variables>
        x_pred [n_dim_sys, n_particle] {numpy-array, float}
            : hidden state at time t given observations for each particle
        x_pred_center [n_dim_sys, n_particle] {numpy-array, float}
            : centering of x_pred
        v [n_dim_sys, n_particle] {numpy-array, float}
            : system noise for transition
        y_background [n_dim_obs, n_particle] {numpy-array, float}
            : background value of observation space
        y_background_mean [n_dim_obs] {numpy-array, float}
            : mean of y_background for particles
        y_background_center [n_dim_obs, n_particle] {numpy-array, float}
            : centerized value of y_background for particles

        <local variables>
        x_pred_mean_local [n_dim_local] {numpy-array, float}
            : localization from x_pred_mean
        x_pred_center_local [n_dim_local, n_particle] {numpy-array, float}
            : localization from x_pred_center
        y_background_mean_local [n_dim_local] {numpy-array, float}
            : localization from y_background_mean
        y_background_center_local [n_dim_local, n_particle] {numpy-array, float}
            : localization from y_background_center
        y_local [n_dim_local] {numpy-array, float}
            : localization from observation y
        R_local [n_dim_local, n_dim_local] {numpy-array, float}
            : localization from observation covariance R
        '''

        # lenght of time-series
        T = self.y.shape[0]

        ## definition of array
        self.x_pred_mean = np.zeros((T + 1, self.n_dim_sys), dtype = self.dtype)
        self.x_filt = np.zeros((T + 1, self.n_dim_sys, self.n_particle),
             dtype = self.dtype)
        self.x_filt[0].T[:] = self.initial_mean
        self.x_filt_mean = np.zeros((T + 1, self.n_dim_sys), dtype = self.dtype)

        self.x_pred_mean[0] = self.initial_mean
        self.x_filt_mean[0] = self.initial_mean

        # prediction and filtering
        for t in range(T):
            # visualization for calculation
            print('\r filter calculating... t={}'.format(t+1) + '/' + str(T), end='')

            ## filter update
            # prediction
            f = _last_dims(self.f, t, 1)[0]

            # raise parametric system noise
            # sys x particle
            v = self.q[0](*self.q[1], size = self.n_particle).T

            # ensemble prediction
            # sys x particle
            x_pred = f(*[self.x_filt[t], v])

            # time x sys
            self.x_pred_mean[t + 1] = np.mean(x_pred, axis = 1)

            # treat missing values
            if np.any(np.ma.getmask(self.y[t])):
                # time x sys x particle
                self.x_filt[t + 1] = x_pred
            else:
                ## Step1 : model space -> observation space
                h = _last_dims(self.h, t, 1)[0]
                R = _last_dims(self.R, t)

                # y_background : obs x particle
                y_background = h(x_pred)

                # y_background mean : obs
                y_background_mean = np.mean(y_background, axis = 1)

                # y_background center : obs x particle
                y_background_center = (y_background.T - y_background_mean).T


                ## Step2 : calculate for model space
                # x_pred_center : sys x particle
                x_pred_center = (x_pred.T - self.x_pred_mean[t + 1]).T


                ## Step3 : select data for grid point
                # n_dim_sys x local_sys
                x_pred_mean_local = self._parallel_global_to_local(
                    self.x_pred_mean[t], self.n_dim_sys, self.A_sys
                    )

                # n_dim_sys x local_sys x particle
                x_pred_center_local = self._parallel_global_to_local(x_pred_center,
                    self.n_dim_sys, self.A_sys)

                # n_dim_sys x local_obs
                y_background_mean_local = self._parallel_global_to_local(
                    y_background_mean, self.n_dim_obs, self.A_obs
                    )

                # n_dim_sys x local_obs x particle
                y_background_center_local = self._parallel_global_to_local(
                    y_background_center,
                    self.n_dim_obs, self.A_obs
                    )

                # n_dim_sys x local_obs
                y_local = self._parallel_global_to_local(
                    self.y[t], self.n_dim_obs, self.A_obs
                    )

                # n_dim_sys x local_obs x local_obs
                R_local = np.zeros(self.n_dim_sys, dtype = object)
                for i in range(self.n_dim_sys):
                    R_local[i] = R[self.A_obs[i]][:, self.A_obs[i]]

                ## Step4-9 : local processing
                p = Pool(self.cpu_number)
                self.x_filt[t + 1] = p.starmap(local_multi_processing, zip(
                    x_pred_mean_local, x_pred_center_local, y_background_mean_local,
                    y_background_center_local, y_local, R_local,
                    range(self.n_dim_sys),
                    self.A_sys, itertools.repeat(self.n_particle),
                    itertools.repeat(self.rho)
                    ))
                p.close()

            self.x_filt_mean[t + 1] = np.mean(self.x_filt[t + 1], axis = 1)


    # get predicted value     
    def get_predicted_value(self, dim = None) :
        try :
            self.x_pred_mean[0]
        except :
            self.filter()

        if dim is None:
            return self.x_pred_mean[1:]
        elif dim <= self.x_pred_mean.shape[1]:
            return self.x_pred_mean[1:, int(dim)]
        else:
            raise ValueError('The dim must be less than '
             + self.x_pred_mean.shape[1] + '.')


    def get_filtered_value(self, dim = None) :
        try :
            self.x_filt_mean[0]
        except :
            self.filter()

        if dim is None:
            return self.x_filt_mean[1:]
        elif dim <= self.x_filt_mean.shape[1]:
            return self.x_filt_mean[1:, int(dim)]
        else:
            raise ValueError('The dim must be less than '
             + self.x_filt_mean.shape[1] + '.')


    # parallel calculation for global to local
    def _parallel_global_to_local(self, variable, n_dim, adjacency_matrix):
        '''
        variable : input variable which transited from global to local
        n_dim : dimension of variables
        adjacency_matrix : adjacency matrix for transition
        '''
        p = Pool(self.cpu_number)
        new_variable = p.starmap(global_to_local,
            zip(adjacency_matrix, itertools.repeat(variable, n_dim)))
        p.close()
        return new_variable

