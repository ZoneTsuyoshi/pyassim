"""
=====================================================
Inference with Locally and Spatially Uniform LOCK
=====================================================
This module implements the locally and spatially uniform LOCK
for Linear-Gaussian state space models
"""
from logging import getLogger, StreamHandler, DEBUG
logger = getLogger("llock")
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

import os
import math
import time
import multiprocessing as mp
import itertools

import numpy as np

from utils import array1d, array2d
from util_functions import _parse_observations, _last_dims, \
    _determine_dimensionality



def _predict_state(F,x):
    return F @ x


def _predict_covariance(M3_i,V_i,Fi,F_i,Q_i):
    f_i = np.zeros(len(F_i)+1)
    f_i[M3_i] = Fi
    fVF_i = np.zeros(len(F_i)+1)
    fVF_i[:-1] = (f_i[:-1].reshape(1,-1) @ V_i @ F_i.T)[0]
    return Q_i + fVF_i[M3_i]
    # return Q_i + (f_i.reshape(1,-1) @ V_i @ F_i)[0,M3_i]


def _predict_calculation(M3_i,V_i,Fi,F_i,Q_i,x):
    f_i = np.zeros(len(F_i)+1)
    f_i[M3_i] = Fi
    fVF_i = np.zeros(len(F_i)+1)
    fVF_i[:-1] = (f_i[:-1].reshape(1,-1) @ V_i @ F_i.T)[0]
    result = np.zeros(len(Fi)+1)
    result[0] = Fi @ x
    result[1:] = Q_i + fVF_i[M3_i]
    return result


def _predict_lag_calculation(M3_i,V_i,V_iL,Fi,F_i,Q_i,x):
    f_i = np.zeros(len(F_i)+1)
    f_i[M3_i] = Fi
    fVF_i = np.zeros(len(F_i)+1)
    fVF_i[:-1] = (f_i[:-1].reshape(1,-1) @ V_i @ F_i.T)[0]
    v_iL = np.zeros((len(V_iL), len(F_i)+1))
    v_iL[:,M3_i] = V_iL
    F_iex = np.zeros((F_i.shape[0]+1, F_i.shape[1]))
    F_iex[:-1] = F_i
    result = np.zeros((len(V_iL)+1, len(Fi)+1))
    result[0,0] = Fi @ x
    result[-1,1:] = Q_i + fVF_i[M3_i]
    result[:-1,1:] = v_iL[:,:-1] @ F_iex[M3_i].T
    return result


def _filter_Kalman_gain_and_covariance(M_i,M3_i,Vi,V_i,R_i):
    K_i = np.zeros(len(V_i)+1)
    # print(V_i, R_i)
    # K_i[:-1] = (V_i[M_i].reshape(1,-1) @ np.linalg.pinv(V_i + R_i))[0]
    K_i[:-1] = np.linalg.solve(V_i + R_i, V_i[M_i])
    KV_i = np.zeros(len(V_i)+1)
    KV_i[:-1] = (K_i[:-1].reshape(1,-1) @ V_i)[0]
    return K_i[M3_i], Vi - (KV_i)[M3_i]
    # VmKV = np.zeros(len(V_i)+1)
    # VmKV[:-1] = V_i[M_i] - KV_i[:-1]
    # return K_i[M3_i], (VmKV)[M3_i]
    # return K_i[0, M3_i], Vi - (K_i @ V_i)[0, M3_i]


def _filter_state(xi,Ki,yLi,xLi):
    return xi + Ki @ (yLi - xLi)


def _filter_calculation(M_i,M3_i,Vi,V_i,R_i,xi,yLi,xLi):
    K_i = np.zeros(len(V_i)+1)
    K_i[:-1] = np.linalg.solve(V_i + R_i, V_Li[M_i]) # (N3+1)
    KV_i = np.zeros(len(V_i)+1) # (N3+1)
    KV_i[:,:-1] = (K_i[:-1].reshape(1,-1) @ V_i)[0] # (N3+1)
    result = np.zeros(len(Vi)+1)
    result[0] = xi + K_i[M3_i] @ (yLi - xLi)
    result[1:] = VLi - (KV_i)[M3_i]
    return result


def _filter_lag_calculation(M_i,M3_i,VLi,V_i,V_Li,R_i,xi,yLi,xLi):
    K_i = np.zeros((len(VLi),len(V_i)+1))
    K_i[:,:-1] = np.linalg.solve(V_i + R_i, V_Li.transpose()[M_i]).transpose() # (L+1,N3+1)
    # for j in range(len(VLi)):
    #     K_i[j,:-1] = np.linalg.solve(V_i + R_i, V_Li[j,M_i]) # (N3+1)
    KV_i = np.zeros((len(VLi), len(V_i)+1)) # (L+1,N3+1)
    KV_i[:,:-1] = (K_i[:,:-1] @ V_i) # (L+1,N3+1)
    result = np.zeros((len(VLi+1),VLi.shape[1]+1))
    result[0,0] = xi + K_i[-1][M3_i] @ (yLi - xLi)
    result[:,1:] = VLi - (KV_i)[:,M3_i]
    return result


def _local_calculation(Pmax,Xii,yi,Pi):
    theta = np.zeros(Pmax+1)
    theta[1:] = np.linalg.pinv(Xii) @ yi
    return theta[Pi]


class LSLOCKME(object) :
    """Implements the Locally and Spatially Uniform LOCK.
    This class implements the LSLOCK,
    for a Linear Gaussian model specified by,
    .. math::
        x_{t+1}   &= F_{t} x_{t} + b_{t} + v_{t} \\
        y_{t}     &= H_{t} x_{t} + d_{t} + w_{t} \\
        [v_{t}, w_{t}]^T &\sim N(0, [[Q_{t}, O], [O, R_{t}]])
    The LSLOCK is an algorithm designed to estimate
    :math:`P(x_t | y_{0:t})` and :math:`F` in real-time. 
    As all state transitions and observations are
    linear with Gaussian distributed noise, these distributions can be
    represented exactly as Gaussian distributions with mean
    `x_filt[t]` and covariances `V_filt`.

    Args:
        observation [n_time, n_dim_obs] {numpy-array, float}
            also known as :math:`y`. observation value
        initial_mean [n_dim_sys] {float} 
            also known as :math:`\mu_0`. initial state mean
        initial_covariance [n_dim_sys, n_dim_sys] {numpy-array, float} 
            also known as :math:`\Sigma_0`. initial state covariance
        transition_matrix [n_dim_sys, n_dim_sys] 
            or [n_dim_sys, n_dim_sys]{numpy-array, float}
            also known as :math:`F`. transition matrix from x_{t-1} to x_{t}
        observation_matrix [n_dim_sys, n_dim_obs] {numpy-array, float}
            also known as :math:`H`. observation matrix from x_{t} to y_{t}
        transition_covariance [n_time-1, n_dim_sys, n_dim_sys]
             or [n_dim_sys, n_dim_sys]
            {numpy-array, float}
            also known as :math:`Q`. system transition covariance
        observation_covariance [n_time, n_dim_obs, n_dim_obs] {numpy-array, float} 
            also known as :math:`R`. observation covariance
        adjacency_matrix [n_dim_sys, n_dim_sys] {numpy-array, float}
            also known as :math:`A`. adjacency matrix, 
            if there is a link between i and j, A[i,j]=1, else A[i,j]=0.
            Besides, you should A[i,i]=1 forall i.
        method {string}
            : method for localized calculation
            "elementwise": calculation for each element of transition matrix
            "local-average": average calculation for specific two observation dimenstions
            "all-average": average calculation for each observation dimenstions
        update_interval {int}
            interval of update transition matrix F
        eta (in (0,1])
            update rate for update transition matrix F
        cutoff
            cutoff distance for update transition matrix F
        save_dir {str, directory-like}
            directory for saving transition matrices and filtered states.
            if this variable is `None`, cannot save them.
        advance_mode {bool}
            if True, calculate transition matrix before filtering.
            if False, calculate the matrix after filtering.
        n_dim_sys {int}
            dimension of system transition variable
        n_dim_obs {int}
            dimension of observation variable
        dtype {type}
            data type of numpy-array
        use_gpu {bool}
            wheather use gpu and cupy.
            if True, you need install package `cupy`.
            if False, set `numpy` for calculation.
        num_cpu {int} or `all`
            number of cpus duaring calculating transition matrix.
            you can set `all` or positive integer.


    Attributes:
        y : `observation`
        F : `transition_matrix`
        Q : `transition_covariance`
        H : `observation_matrix`
        R : `observation_covariance`
    """

    def __init__(self, observation = None,
                initial_mean = None, initial_covariance = None,
                transition_matrix = None, observation_matrix = None,
                transition_covariance = None, observation_covariance = None,
                localization_matrix_1st = None,
                localization_matrix_2nd = None,
                localization_matrix_3rd = None,
                address_vector = None,
                address_matrix = None,
                parameter_matrix = None, common_matrix = None,
                correspoding_local_dimension = None,
                transition_covariance_type = "diag",
                observation_covariance_type = "diag",
                estimation_length = 10, estimation_interval = 1,
                eta = 1., cutoff = 10., 
                estimation_mode = "backward",
                save_dir = None, save_vars = ["state", "transition_matrix"],
                n_dim_sys = None, n_dim_obs = None, dtype = "float32",
                use_gpu = False, num_cpu = "all"):
        """Setup initial parameters.
        """
        if use_gpu:
            try:
                import cupy
                self.xp = cupy
                self.use_gpu = True
            except:
                self.xp = np
                self.use_gpu = False
        else:
            self.xp = np
            self.use_gpu = False

        # determine dimensionality
        self.n_dim_sys = n_dim_sys
        self.n_dim_obs = n_dim_obs

        self.y = self.xp.zeros((observation.shape[0], self.n_dim_obs+1), dtype = dtype)
        self.y[:,:-1] = self.xp.asarray(observation).copy()

        if localization_matrix_1st is None:
            self.L1 = self.xp.ones((self.n_dim_obs+1, 1))
        else:
            self.L1 = self.xp.asarray(localization_matrix_1st, dtype = int)
        self.n_local_1st = self.L1.shape[1]

        if localization_matrix_2nd is None:
            self.L2 = self.xp.ones((self.n_dim_obs+1, 1))
        else:
            self.L2 = self.xp.asarray(localization_matrix_2nd, dtype = int)
        self.n_local_2nd = self.L2.shape[1]
        #self.n_local_2nd2 = 2 * self.n_local_2nd - self.n_local_1st

        if localization_matrix_3rd is None:
            self.L3 = self.xp.ones((self.n_dim_obs+1, 1))
        else:
            # obs x 2nd x 2 x (2nd2*2nd2)
            self.L3 = self.xp.asarray(localization_matrix_3rd, dtype = int)
        self.n_local_3rd = int(math.sqrt(self.L3.shape[2]))

        if address_vector is None:
            self.M = self.xp.zeros(self.n_dim_obs+1, dtype = int)
            for i in range(self.n_dim_obs):
                self.M[i] = self.xp.where(self.L[i]==i)[0][0]
        else:
            self.M = self.xp.asarray(address_vector, dtype = int)

        if address_matrix is None:
            self.M3 = self.xp.zeros(self.n_dim_obs+1, dtype = int)
            for i in range(self.n_dim_obs):
                self.M3[i] = self.xp.where(self.L[i]==i)[0][0]
        else:
            # obs x 2nd x 2
            self.M3 = self.xp.asarray(address_matrix, dtype = int)

        if common_matrix is None:
            self.C = self.xp.zeros((self.n_local_2nd, 2, 2), dtype=int)
        else:
            self.C = self.xp.asarray(common_matrix, dtype = int)

        if parameter_matrix is None:
            self.P = self.xp.zeros((self.L1.shape[0]-1, self.L1.shape[1]), dtype=int)
            self.P[:] = self.xp.arange(self.L1.shape[1]) + 1
        else:
            self.P = self.xp.asarray(parameter_matrix, dtype=int)
        self.Pmax = int(self.P.max())

        if correspoding_local_dimension is None:
            self.B = self.xp.tile(self.xp.arange(self.n_local_1st, dtype=int), (self.n_dim_sys+1, 1))
        else:
            self.B = self.xp.asarray(correspoding_local_dimension, dtype=int)

        if initial_mean is None:
            self.initial_mean = self.xp.zeros(self.n_dim_sys, dtype = dtype)
        else:
            self.initial_mean = self.xp.asarray(initial_mean, dtype = dtype)
        
        if initial_covariance is None:
            self.initial_covariance = self.xp.zeros((self.n_dim_sys, self.n_local_2nd), dtype = dtype)
            self.initial_covariance[self.xp.arange(self.n_dim_sys), self.M] = 1
        else:
            self.initial_covariance = self.xp.asarray(initial_covariance, dtype = dtype)

        if transition_matrix is None:
            self.F = self.xp.ones((self.n_dim_sys+1, self.n_local_2nd+1), dtype = dtype)
        else:
            self.F = self.xp.zeros((self.n_dim_sys+1, self.n_local_2nd+1), dtype=dtype)
            self.F[self.xp.repeat(self.xp.arange(self.n_dim_sys, dtype=int), self.n_local_1st), self.B.flatten()] = \
                self.xp.asarray(transition_matrix[:-1]).flatten()
            
        if transition_covariance is None:
            self.tcov_type = "diag"
            # self.Q = self.xp.ones(self.n_dim_sys, dtype = dtype)
            self.Q = self.xp.zeros((self.n_dim_sys, self.n_local_2nd), dtype = dtype)
            self.Q[self.xp.arange(self.n_dim_sys), self.M] = 1
        else:
            if transition_covariance_type == "diag":
                assert transition_covariance.ndim == 1
            elif transition_covariance_type == "local":
                assert transition_covariance.shape == (self.n_dim_sys, self.n_local_2nd)
            else:
                raise ValueError("Type of covariance must be \"diag\" or \"local\".")
            self.tcov_type = transition_covariance_type
            self.Q = self.xp.asarray(transition_covariance, dtype = dtype)
        
        if observation_covariance is None:
            self.ocov_type = "diag"
            self.R = self.xp.zeros(self.n_dim_obs+1, dtype = dtype)
        else:
            if observation_covariance_type == "diag":
                assert observation_covariance.ndim == 1
                self.R = self.xp.zeros(self.n_dim_obs+1, dtype = dtype)
                self.R[:-1] = self.xp.asarray(observation_covariance)
            elif observation_covariance_type == "local":
                assert observation_covariance.shape == (self.n_dim_obs, self.n_local_2nd)
                self.R = self.xp.zeros((self.n_dim_obs+1, self.n_local_2nd+1))
                self.R[-1,0] = 1
                self.R[:-1,:-1] = self.xp.asarray(observation_covariance)
            self.ocov_type = observation_covariance_type

        if estimation_mode in ["forward", "middle", "backward"]:
            self.estimation_mode = estimation_mode
        else:
            raise ValueError("\"estimation_mode\" must be choosen from \"forward\","
                            + " \"middle\", or \"backward\".")

        if self.estimation_mode in ["forward", "backward"]:
            self.tau = int(estimation_length)
            self.tau2 = int((estimation_length - 1) / 2)
        else:
            self.tau2 = int((estimation_length - 1) / 2)
            self.tau = 2 * self.tau2 + 1

        self.I = estimation_interval
        self.tm_count = 1

        self.save_vars = []
        if save_dir is not None:
            self.save_dir = save_dir
            if "state" in save_vars or "x" in save_vars:
                self.save_vars.append("x")
            if "transition_matrix" in save_vars or "F" in save_vars:
                self.save_vars.append("F")
                self.fillnum = len(str(int(self.y.shape[0] / self.I)))
                self.xp.save(os.path.join(self.save_dir, "transition_matrix_" + str(0).zfill(self.fillnum) + ".npy"), self.F[:-1,:-1])
            if "covariance" in save_vars or "V" in save_vars:
                self.save_vars.append("V")
            if "forecasted_covariance" in save_vars or "Vf" in save_vars:
                self.save_vars.append("Vf")
            if "gain" in save_vars or "K" in save_vars:
                self.save_vars.append("K")
            self.Tfill = len(str(len(self.y)))

        if num_cpu == "all":
            self.num_cpu = mp.cpu_count()
        else:
            self.num_cpu = num_cpu

        self.eta = eta
        self.cutoff = cutoff
        self.dtype = dtype
        self.times = self.xp.zeros(7)


    def forward(self):
        """Calculate prediction and filter for observation times.

        Attributes:
            T {int}
                : length of data y
            x_pred [n_time, n_dim_sys] {numpy-array, float}
                : mean of hidden state at time t given observations
                 from times [0...t-1]
            V_pred [n_dim_sys, n_dim_sys] {numpy-array, float}
                : covariance of hidden state at time t given observations
                 from times [0...t-1]
            x_filt [n_time, n_dim_sys] {numpy-array, float}
                : mean of hidden state at time t given observations from times [0...t]
            V_filt [n_dim_sys, n_dim_sys] {numpy-array, float}
                : covariance of hidden state at time t given observations
                 from times [0...t]
        """

        T = self.y.shape[0]
        self.x_pred = self.xp.zeros((T, self.n_dim_sys+1), dtype = self.dtype)
        self.x_filt = self.xp.zeros((T, self.n_dim_sys+1), dtype = self.dtype)
        self.V_pred = self.xp.zeros((self.n_dim_sys+1, self.n_local_2nd+1),
             dtype = self.dtype)
        self.V_filt = self.xp.zeros((self.n_dim_sys+1, self.n_local_2nd+1),
             dtype = self.dtype)

        # calculate prediction and filter for every time
        for t in range(T):
            # visualize calculating time
            print("\r filter calculating... t={}".format(t) + "/" + str(T), end="")

            if t == 0:
                # initial setting
                self.x_pred[0,:-1] = self.initial_mean
                self.V_pred[:-1,:-1] = self.initial_covariance.copy()
                self._update_transition_matrix(self.tau)
            else:
                if t >= 2 and t < T-self.tau+1 and (t-1)%self.I==0 and self.estimation_mode=="forward":
                    self._update_transition_matrix(t+self.tau-1)
                elif t >= self.tau+1 and (t-self.tau)%self.I==0 and self.estimation_mode=="backward":
                    self._update_transition_matrix(t)
                elif t >= self.tau2+2 and t < T-self.tau2 and (t-self.tau2-1)%self.I==0 and self.estimation_mode=="middle":
                    self._update_transition_matrix(t+self.tau2)
                # start_time = time.time()
                self._predict_update(t)
                # self.times[0] += time.time() - start_time
            
            if self.xp.any(self.xp.isnan(self.y[t])):
                self.x_filt[t] = self.x_pred[t]
                self.V_filt = self.V_pred
            else:
                # start_time = time.time()
                self._filter_update(t)
                # self.times[1] += time.time() - start_time

            if "Vf" in self.save_vars:
                self.xp.save(os.path.join(self.save_dir, "covariancef_{}.npy".format(str(t).zfill(self.Tfill))), self.V_pred[:-1,:-1])
            if "V" in self.save_vars:
                self.xp.save(os.path.join(self.save_dir, "covariance_{}.npy".format(str(t).zfill(self.Tfill))), self.V_filt[:-1,:-1])

        if "x" in self.save_vars:
            self.xp.save(os.path.join(self.save_dir, "states.npy"), self.x_filt[:,:-1])


    def _predict_update(self, t):
        """Calculate fileter update without noise

        Args:
            t {int} : observation time
        """
        # calculate predicted distribution for time t
        # start_time = time.time()
        # p = mp.Pool(self.num_cpu)
        # self.x_pred[t,:-1] = p.starmap(_predict_state, zip(self.F[:-1,:-1], self.x_filt[t-1][self.L2[:-1]]))
        # p.close()
        # self.times[0] += time.time() - start_time

        L_flatten = self.L2[:-1].flatten()

        start_time = time.time()
        p = mp.Pool(self.num_cpu)
        # M3_i,V_i,Fi,F_i,Q_i
        # self.V_pred[:-1,:-1] = self.xp.array(p.starmap(_predict_covariance, zip(
        #                                             self.M3,
        #                                             self.V_filt[self.L3[0], self.L3[1]].reshape(self.n_dim_sys, self.n_local_3rd, self.n_local_3rd),
        #                                             self.F[:-1,:-1],
        #                                             self.F[self.L3[0], self.L3[1]].reshape(self.n_dim_sys, self.n_local_3rd, self.n_local_3rd),
        #                                             self.Q))).reshape(self.n_dim_sys, self.n_local_2nd)

        # 200617: all calculation
        # M3_i,V_i,Fi,F_i,Q_i,x_i
        results = self.xp.array(p.starmap(_predict_calculation, zip(
                                                    self.M3,
                                                    self.V_filt[self.L3[0], self.L3[1]].reshape(self.n_dim_sys, self.n_local_3rd, self.n_local_3rd),
                                                    self.F[:-1,:-1],
                                                    self.F[self.L3[0], self.L3[1]].reshape(self.n_dim_sys, self.n_local_3rd, self.n_local_3rd),
                                                    self.Q,
                                                    self.x_filt[t-1][self.L2[:-1]])))

        p.close()

        self.x_pred[t,:-1] = results[:,0]
        self.V_pred[:-1,:-1] = results[:,1:]
        self.times[1] += time.time() - start_time


        # exchange of information
        self.V_pred[self.C[:,0,:].flatten(), self.C[:,1,:].flatten()] = \
            self.V_pred[self.xp.tile(self.C[:,0,:], (1,2)).flatten(), self.xp.tile(self.C[:,1,:], (1,2)).flatten()].reshape(-1,2).mean(axis=1)
        # for c in self.C:
        #     self.V_pred[c[0], c[1]] = self.V_pred[c[0], c[1]].mean()


    def _filter_update(self, t):
        """Calculate fileter update without noise

        Args:
            t {int} : observation time

        Attributes:
            K [n_dim_sys, n_dim_obs] {numpy-array, float}
                : Kalman gain matrix for time t
        """
        # calculate filter step
        K = self.xp.zeros((self.n_dim_sys+1, self.n_local_2nd), dtype=self.dtype)
        start_time = time.time()

        L_flatten = self.L2[:-1].flatten()
        p = mp.Pool(self.num_cpu)
        # if self.ocov_type == "diag":
        #     # j,n_dim,Li,Lj,Mi,Mj,Vi,Vj,Vd,Rd
        #     # K[:-1] = self.xp.array(p.starmap(_filter_Kalman_gain_diag, zip(L_flatten,
        #     #                                         itertools.repeat(self.n_dim_obs),
        #     #                                         self.xp.repeat(self.L2[:-1], self.n_local_2nd, axis=0),
        #     #                                         self.L2[L_flatten],
        #     #                                         self.xp.repeat(self.M[:-1], self.n_local_2nd),
        #     #                                         self.M[L_flatten],
        #     #                                         self.xp.repeat(self.V_pred[:-1], self.n_local_2nd, axis=0),
        #     #                                         self.V_pred[L_flatten],
        #     #                                         itertools.repeat(self.V_pred[self.xp.arange(self.n_dim_sys+1), self.M]),
        #     #                                         itertools.repeat(self.R)))).reshape(self.n_dim_sys, self.n_local_2nd)
        #     pass
        # elif self.ocov_type == "local":
        #     # M_i,M3_i,Vi,V_i,R_i
        #     K[:-1], self.V_filt[:-1,:-1] = self.xp.array(p.starmap(_filter_Kalman_gain_and_covariance, zip(
        #                                             self.M,
        #                                             self.M3,
        #                                             self.V_pred[:-1,:-1],
        #                                             self.V_pred[self.L3[0], self.L3[1]].reshape(self.n_dim_sys, self.n_local_3rd, self.n_local_3rd),
        #                                             self.R[self.L3[0], self.L3[1]].reshape(self.n_dim_sys, self.n_local_3rd, self.n_local_3rd)
        #                                             ))).reshape(self.n_dim_sys, 2, self.n_local_2nd).swapaxes(0,1)

        #     # for M_i, M3_i, Vi, V_i, R_i in zip(
        #     #                             self.M,
        #     #                             self.M3,
        #     #                             self.V_pred[:-1,:-1],
        #     #                             self.V_pred[self.L3[0], self.L3[1]].reshape(self.n_dim_sys, self.n_local_3rd, self.n_local_3rd),
        #     #                             self.R[self.L3[0], self.L3[1]].reshape(self.n_dim_sys, self.n_local_3rd, self.n_local_3rd)
        #     #                             ):
        #     #     K[i], V_filt[i] = _filter_Kalman_gain_and_covariance(M_i,M3_i,Vi,V_i,R_i)

        # 200617: all calculation
        # M_i,M3_i,Vi,V_i,R_i,xi,yLi,xLi
        results = self.xp.array(p.starmap(_filter_calculation, zip(
                                                    self.M,
                                                    self.M3,
                                                    self.V_pred[:-1,:-1],
                                                    self.V_pred[self.L3[0], self.L3[1]].reshape(self.n_dim_sys, self.n_local_3rd, self.n_local_3rd),
                                                    self.R[self.L3[0], self.L3[1]].reshape(self.n_dim_sys, self.n_local_3rd, self.n_local_3rd),
                                                    self.x_pred[t,:-1],
                                                    self.y[t][self.L2[:-1]],
                                                    self.x_pred[t][self.L2[:-1]]
                                                    )))
        p.close()
        self.x_filt[t,:-1] = results[:,0]
        self.V_filt[:-1,:-1] = results[:,1:]

        self.times[2] += time.time() - start_time
        if "K" in self.save_vars:
            self.xp.save(os.path.join(self.save_dir, "Kalman_gain_{}.npy".format(str(t).zfill(self.Tfill))), K[:-1])
        
        # start_time = time.time()
        # p = mp.Pool(self.num_cpu)
        # # xi,Ki,yLi,xLi
        # self.x_filt[t,:-1] = p.starmap(_filter_state, zip(self.x_pred[t,:-1],
        #                                             K[:-1],
        #                                             self.y[t][self.L2[:-1]],
        #                                             self.x_pred[t][self.L2[:-1]]))
        # p.close()
        # self.times[3] += time.time() - start_time

        # exchange of information
        self.V_filt[self.C[:,0,:].flatten(), self.C[:,1,:].flatten()] = \
            self.V_filt[self.xp.tile(self.C[:,0,:], (1,2)).flatten(), self.xp.tile(self.C[:,1,:], (1,2)).flatten()].reshape(-1,2).mean(axis=1)
        # for c in self.C:
        #     self.V_filt[c[0], c[1]] = self.V_filt[c[0], c[1]].mean()



    def _update_transition_matrix(self, t):
        """Update transition matrix

        Args:
            t {int} : observation time
        """
        Fh = self.xp.zeros((self.n_dim_sys+1, self.n_local_2nd+1), dtype=self.dtype)

        start_time = time.time()
        Xi = self.xp.zeros(((self.n_dim_obs+1)*self.tau, self.Pmax+1), dtype=self.dtype)
        Xi[(self.n_dim_obs+1)*self.xp.repeat(self.xp.arange(self.tau, dtype=int), self.n_dim_obs*self.n_local_1st)
            + self.xp.tile(self.xp.repeat(self.xp.arange(self.n_dim_obs, dtype=int), self.n_local_1st), self.tau),
            self.xp.tile(self.P.flatten(), self.tau)] \
            = self.y[t-self.tau:t].flatten()[(self.n_dim_obs+1)*self.xp.repeat(self.xp.arange(self.tau, dtype=int), 
                                                                                self.n_dim_obs*self.n_local_1st)
                + self.xp.tile(self.L1[:-1].flatten(), self.tau)]

        tile_repeat_arange = (self.n_dim_obs+1)*self.xp.tile(self.xp.repeat(self.xp.arange(self.tau, dtype=int), 
                                                                            self.n_local_1st), self.n_dim_obs)
        L_flatten_tile = self.xp.tile(self.L1[:-1], self.tau).flatten()
        slicing_array = tile_repeat_arange + L_flatten_tile
        p = mp.Pool(self.num_cpu)
        # Pmax,Xii,yi,Pi
        Fh[self.xp.repeat(self.xp.arange(self.n_dim_sys, dtype=int), self.n_local_1st), self.B.flatten()] \
                = self.xp.array(p.starmap(_local_calculation, zip(itertools.repeat(self.Pmax),
                                                    Xi[slicing_array, 1:].reshape(self.n_dim_obs, self.n_local_1st*self.tau, self.Pmax),
                                                    self.y[t-self.tau+1:t+1].flatten()[slicing_array]\
                                                        .reshape(self.n_dim_obs, self.n_local_1st*self.tau),
                                                    self.P))).reshape(-1)
        p.close()

        self.times[5] += time.time() - start_time

        self.times[6] += 1
        if self.tm_count==1:
            self.F = Fh.copy()
        else:
            self.F = self.F - self.eta * self.xp.minimum(self.xp.maximum(-self.cutoff, self.F - Fh), self.cutoff)

        if "F" in self.save_vars:
            self.xp.save(os.path.join(self.save_dir, "transition_matrix_" + str(self.tm_count).zfill(self.fillnum) + ".npy"), self.F[:-1,:-1])
        self.tm_count += 1


    def get_predicted_value(self, dim = None):
        """Get predicted value

        Args:
            dim {int} : dimensionality for extract from predicted result

        Returns (numpy-array, float)
            : mean of hidden state at time t given observations
            from times [0...t-1]
        """
        # if not implement `forward`, implement `forward`
        try :
            self.x_pred[0,:-1]
        except :
            self.forward()

        if dim is None:
            return self.x_pred[:,:-1]
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
        # if not implement `forward`, implement `forward`
        try :
            self.x_filt[0,:-1]
        except :
            self.forward()

        if dim is None:
            return self.x_filt[:,:-1]
        elif dim <= self.x_filt.shape[1]:
            return self.x_filt[:, int(dim)]
        else:
            raise ValueError('The dim must be less than '
                 + self.x_filt.shape[1] + '.')



    def fixed_lag_smooth(self, L=5):
        """Calculate prediction and filter for observation times.

        Attributes:
            T {int}
                : length of data y
            x_pred [n_time, n_dim_sys] {numpy-array, float}
                : mean of hidden state at time t given observations
                 from times [0...t-1]
            V_pred [n_dim_sys, n_dim_sys] {numpy-array, float}
                : covariance of hidden state at time t given observations
                 from times [0...t-1]
            x_filt [n_time, n_dim_sys] {numpy-array, float}
                : mean of hidden state at time t given observations from times [0...t]
            V_filt [n_dim_sys, n_dim_sys] {numpy-array, float}
                : covariance of hidden state at time t given observations
                 from times [0...t]
        """

        T = self.y.shape[0]
        self.x_pred = self.xp.zeros((T, self.n_dim_sys+1), dtype = self.dtype)
        self.x_filt = self.xp.zeros((T, self.n_dim_sys+1), dtype = self.dtype)
        self.V_pred = self.xp.zeros((L+1, self.n_dim_sys+1, self.n_local_2nd+1),
             dtype = self.dtype)
        self.V_filt = self.xp.zeros((L+1, self.n_dim_sys+1, self.n_local_2nd+1),
             dtype = self.dtype)

        # calculate prediction and filter for every time
        for t in range(T):
            # visualize calculating time
            print("\r filter calculating... t={}".format(t) + "/" + str(T), end="")

            if t == 0:
                # initial setting
                self.x_pred[0,:-1] = self.initial_mean
                self.V_pred[:,:-1,:-1] = self.initial_covariance.copy()
                self._update_transition_matrix(self.tau)
            else:
                if t >= 2 and t < T-self.tau+1 and (t-1)%self.I==0 and self.estimation_mode=="forward":
                    self._update_transition_matrix(t+self.tau-1)
                elif t >= self.tau+1 and (t-self.tau)%self.I==0 and self.estimation_mode=="backward":
                    self._update_transition_matrix(t)
                elif t >= self.tau2+2 and t < T-self.tau2 and (t-self.tau2-1)%self.I==0 and self.estimation_mode=="middle":
                    self._update_transition_matrix(t+self.tau2)
                # start_time = time.time()
                self._predict_update_lag(t, L)
                # self.times[0] += time.time() - start_time
            
            if self.xp.any(self.xp.isnan(self.y[t])):
                self.x_filt[t] = self.x_pred[t].copy()
                self.V_filt[-1] = self.V_pred[-1].copy()
            else:
                # start_time = time.time()
                self._filter_update_lag(t, L)
                # self.times[1] += time.time() - start_time

            low = max(0,t-L)
            if t==T-1:
                for i,s in enumerate(range(t-L, T)):
                    if "Vf" in self.save_vars:
                        self.xp.save(os.path.join(self.save_dir, "covariancef_{}.npy".format(str(s).zfill(self.Tfill))), self.V_pred[i,:-1,:-1])
                    if "V" in self.save_vars:
                        self.xp.save(os.path.join(self.save_dir, "covariance_{}.npy".format(str(s).zfill(self.Tfill))), self.V_filt[i,:-1,:-1])
            else:
                if "Vf" in self.save_vars:
                    self.xp.save(os.path.join(self.save_dir, "covariancef_{}.npy".format(str(low).zfill(self.Tfill))), self.V_pred[0,:-1,:-1])
                if "V" in self.save_vars:
                    self.xp.save(os.path.join(self.save_dir, "covariance_{}.npy".format(str(low).zfill(self.Tfill))), self.V_filt[0,:-1,:-1])


        if "x" in self.save_vars:
            self.xp.save(os.path.join(self.save_dir, "states.npy"), self.x_filt[:,:-1])


    def _predict_update_lag(self, t, L):
        """Calculate fileter update without noise

        Args:
            t {int} : observation time
        """
        L_flatten = self.L2[:-1].flatten()

        # time backward
        start_time = time.time()
        p = mp.Pool(self.num_cpu)
        # M3_i,V_i,V_iL,Fi,F_i,Q_i,x_i
        results = self.xp.array(p.starmap(_predict_lag_calculation, zip(
                                                    self.M3,
                                                    self.V_filt[-1][self.L3[0], self.L3[1]].reshape(self.n_dim_sys, self.n_local_3rd, self.n_local_3rd),
                                                    self.V_filt[1:,:-1,:-1].transpose(1,0,2),
                                                    self.F[:-1,:-1],
                                                    self.F[self.L3[0], self.L3[1]].reshape(self.n_dim_sys, self.n_local_3rd, self.n_local_3rd),
                                                    self.Q,
                                                    self.x_filt[t-1][self.L2[:-1]])))
        p.close()

        # for M3_i,V_i,V_iL,Fi,F_i,Q_i,x_i in zip(self.M3,
        #             self.V_filt[-1][self.L3[0], self.L3[1]].reshape(self.n_dim_sys, self.n_local_3rd, self.n_local_3rd),
        #             self.V_filt[1:,:-1,:-1].transpose(1,0,2),
        #             self.F[:-1,:-1],
        #             self.F[self.L3[0], self.L3[1]].reshape(self.n_dim_sys, self.n_local_3rd, self.n_local_3rd),
        #             self.Q,
        #             self.x_filt[t-1][self.L2[:-1]]):
        #     _predict_lag_calculation(M3_i,V_i,V_iL,Fi,F_i,Q_i,x_i)

        self.x_pred[t,:-1] = results[:,0,0]
        self.V_pred[:,:-1,:-1] = results[:,:,1:].transpose(1,0,2)
        self.times[1] += time.time() - start_time


        # exchange of information
        for i in range(L+1):
            self.V_pred[i][self.C[:,0,:].flatten(), self.C[:,1,:].flatten()] = \
                self.V_pred[i][self.xp.tile(self.C[:,0,:], (1,2)).flatten(), self.xp.tile(self.C[:,1,:], (1,2)).flatten()].reshape(-1,2).mean(axis=1)


    def _filter_update_lag(self, t, L):
        """Calculate fileter update without noise

        Args:
            t {int} : observation time

        Attributes:
            K [n_dim_sys, n_dim_obs] {numpy-array, float}
                : Kalman gain matrix for time t
        """
        # calculate filter step
        K = self.xp.zeros((self.n_dim_sys+1, self.n_local_2nd), dtype=self.dtype)
        start_time = time.time()

        L_flatten = self.L2[:-1].flatten()
        p = mp.Pool(self.num_cpu)
        # M_i,M3_i,VLi,V_i,V_Li,R_i,xi,yLi,xLi
        results = self.xp.array(p.starmap(_filter_lag_calculation, zip(
                                                    self.M,
                                                    self.M3,
                                                    self.V_pred[:,:-1,:-1].transpose(1,0,2),
                                                    self.V_pred[-1][self.L3[0], self.L3[1]].reshape(self.n_dim_sys, self.n_local_3rd, self.n_local_3rd),
                                                    self.V_pred.transpose(1,2,0)[self.L3[0], self.L3[1]].\
                                                        transpose(0,2,1).reshape(self.n_dim_sys, L+1, self.n_local_3rd, self.n_local_3rd),
                                                    self.R[self.L3[0], self.L3[1]].reshape(self.n_dim_sys, self.n_local_3rd, self.n_local_3rd),
                                                    self.x_pred[t,:-1],
                                                    self.y[t][self.L2[:-1]],
                                                    self.x_pred[t][self.L2[:-1]]
                                                    )))
        p.close()

        # for M_i,M3_i,VLi,V_i,V_Li,R_i,xi,yLi,xLi in zip(self.M,
        #                 self.M3,
        #                 self.V_pred[:,:-1,:-1].transpose(1,0,2),
        #                 self.V_pred[-1][self.L3[0], self.L3[1]].reshape(self.n_dim_sys, self.n_local_3rd, self.n_local_3rd),
        #                 self.V_pred.transpose(1,2,0)[self.L3[0], self.L3[1]].\
        #                     transpose(0,2,1).reshape(self.n_dim_sys, L+1, self.n_local_3rd, self.n_local_3rd),
        #                 self.R[self.L3[0], self.L3[1]].reshape(self.n_dim_sys, self.n_local_3rd, self.n_local_3rd),
        #                 self.x_pred[t,:-1],
        #                 self.y[t][self.L2[:-1]],
        #                 self.x_pred[t][self.L2[:-1]]):
        #     _filter_lag_calculation(M_i,M3_i,VLi,V_i,V_Li,R_i,xi,yLi,xLi)


        self.x_filt[t,:-1] = results[:,0,0]
        self.V_filt[:,:-1,:-1] = results[:,:,1:].transpose(1,0,2)

        self.times[2] += time.time() - start_time
        if "K" in self.save_vars:
            self.xp.save(os.path.join(self.save_dir, "Kalman_gain_{}.npy".format(str(t).zfill(self.Tfill))), K[:-1])
        

        # exchange of information
        for i in range(L+1):
            self.V_filt[i][self.C[:,0,:].flatten(), self.C[:,1,:].flatten()] = \
                self.V_filt[i][self.xp.tile(self.C[:,0,:], (1,2)).flatten(), self.xp.tile(self.C[:,1,:], (1,2)).flatten()].reshape(-1,2).mean(axis=1)
