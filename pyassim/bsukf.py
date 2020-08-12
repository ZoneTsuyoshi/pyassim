"""
=============================
Inference with Sequential Update Kalman Filter
=============================
This module implements the Sequential Update Kalman Filter
and Kalman Smoother for Linear-Gaussian state space models
"""
from logging import getLogger, StreamHandler, DEBUG
logger = getLogger("bsukf")
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

# write dot when desktop
from .utils import array1d, array2d
from .util_functions import _parse_observations, _last_dims, \
    _determine_dimensionality


def _local_calculation(i, j, A, y, S_inverse, G_mean, G_precision, xp=np):
    # preparation for localization
    where_are_local = A[i] | A[j]
    local_node_number_i = len(xp.where(where_are_local[:i])[0])
    local_node_number_j = len(xp.where(where_are_local[:j])[0])
    global_to_local = xp.where(where_are_local)[0] # use slice data
    local_length = len(global_to_local)

    # global to local
    y_local = y[:, global_to_local]
    S_inverse_local = S_inverse[xp.ix_(global_to_local, global_to_local)]
    Zbb_local = y_local[:-1].T @ y_local[:-1]
    Znb_local = y_local[1:].T @ y_local[:-1]
    G_mean_local = G_mean[xp.ix_(global_to_local, global_to_local)]
    G_precision_local = G_precision[xp.ix_(global_to_local, global_to_local)]

    # calculation
    # C_local = xp.zeros((local_length**2, local_length**2))
    # beta_local = xp.zeros(local_length**2)

    # for m in range(local_length**2):
    #     mq = m//local_length
    #     mr = m%local_length
    #     for n in range(local_length**2):
    #         nq = n//local_length
    #         nr = n%local_length
    #         C_local[m,n] = S_inverse_local[mq,nq]*Zbb_local[mr,nr]
    #         if m==n:
    #             C_local[m,n] += G_precision_local[mq,mr]
    #     beta_local[m] = S_inverse_local[mq] @ Znb_local[:,mr] + G_mean_local[mq,mr] * G_precision_local[mq,mr]

    beta_local = (S_inverse_local @ Znb_local + G_mean_local * G_precision_local).reshape(-1)
    C_local = xp.kron(S_inverse_local, Zbb_local) + xp.diag(G_precision_local.reshape(-1))

    C_beta_local = C_local.copy()
    C_beta_local[:,local_node_number_i*local_length+local_node_number_j] = beta_local
    sign1, logdet1 = xp.linalg.slogdet(C_beta_local)
    sign2, logdet2 = xp.linalg.slogdet(C_local)
    # G_mean_local = (xp.linalg.pinv(C_local) @ beta_local).reshape(local_length, local_length)
    # G_mean_local = xp.linalg.lstsq(C_local, beta_local)[0].reshape(local_length, local_length)
    # return G_mean_local[local_node_number_i, local_node_number_j]#, \
            # Zbb_local[local_node_number_i, local_node_number_j]
    # return xp.linalg.det(C_beta_local) / xp.linalg.det(C_local)
    return sign1 * sign2 * xp.exp(logdet1 - logdet2)


class BayesianSequentialUpdateKalmanFilter(object) :
    """Implements the Kalman Filter, Kalman Smoother, and EM algorithm.
    This class implements the Kalman Filter, Kalman Smoother, and EM Algorithm
    for a Linear Gaussian model specified by,
    .. math::
        x_{t+1}   &= F_{t} x_{t} + b_{t} + v_{t} \\
        y_{t}     &= H_{t} x_{t} + d_{t} + w_{t} \\
        [v_{t}, w_{t}]^T &\sim N(0, [[Q_{t}, O], [O, R_{t}]])
    The Kalman Filter is an algorithm designed to estimate
    :math:`P(x_t | y_{0:t})`.  As all state transitions and observations are
    linear with Gaussian distributed noise, these distributions can be
    represented exactly as Gaussian distributions with mean
    `x_filt[t]` and covariances `V_filt[t]`.
    Similarly, the Kalman Smoother is an algorithm designed to estimate
    :math:`P(x_t | y_{0:T-1})`.

    Args:
        observation [n_time, n_dim_obs] {numpy-array, float}
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
        observation_matrices [n_time, n_dim_sys, n_dim_obs] or [n_dim_sys, n_dim_obs]
             {numpy-array, float}
            also known as :math:`H`. observation matrix from x_{t} to y_{t}
            観測行列[時間軸，状態変数軸，観測変数軸] or [状態変数軸，観測変数軸]
        transition_covariance [n_time - 1, n_dim_noise, n_dim_noise]
             or [n_dim_sys, n_dim_noise]
            {numpy-array, float}
            also known as :math:`Q`. system transition covariance for times
            システムノイズの共分散行列[時間軸，ノイズ変数軸，ノイズ変数軸]
        observation_covariance [n_time, n_dim_obs, n_dim_obs] {numpy-array, float} 
            also known as :math:`R`. observation covariance for times.
            観測ノイズの共分散行列[時間軸，観測変数軸，観測変数軸]
        method {string}
            : method for localized calculation
            "elementwise": calculation for each element of transition matrix
            "local-average": average calculation for specific 2 observation dimenstions
            "all-average": average calculation for each observation dimenstions
        update_interval {int}
            : interval of update transition matrix F
        eta (in (0.1))
            : update rate for transition matrix F
        n_dim_sys {int}
            : dimension of system transition variable
            システム変数の次元
        n_dim_obs {int}
            : dimension of observation variable
            観測変数の次元
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
                transition_matrices = None, observation_matrices = None,
                transition_covariance = None, observation_covariance = None,
                observation_transition_matrix_mean = None,
                observation_transition_matrix_precision = None,
                adjacency_matrix = None, method = "elementwise",
                update_interval = 1,
                save_directory = None, save_state_interval = 1,
                accumulation = [], precision_weight = 0.5,
                advance_mode = True, adjust_precision_on = False,
                n_dim_sys = None, n_dim_obs = None, dtype = "float32",
                use_gpu = True, num_cpu = "all"):
        """Setup initial parameters.
        """
        self.use_gpu = use_gpu
        if use_gpu:
            import cupy
            self.xp = cupy
        else:
            self.xp = np

        # determine dimensionality
        self.n_dim_sys = _determine_dimensionality(
            [(transition_matrices, array2d, -2),
             (initial_mean, array1d, -1),
             (initial_covariance, array2d, -2),
             (observation_matrices, array2d, -1)],
            n_dim_sys
        )

        self.n_dim_obs = _determine_dimensionality(
            [(observation_matrices, array2d, -2),
             (observation_covariance, array2d, -2),
             (observation_transition_matrix_mean, array2d, -2),
             (observation_transition_matrix_precision, array2d, -2)],
            n_dim_obs
        )

        # self.y = _parse_observations(observation)
        self.y = self.xp.asarray(observation).copy()

        if initial_mean is None:
            self.initial_mean = self.xp.zeros(self.n_dim_sys, dtype = dtype)
        else:
            self.initial_mean = self.xp.asarray(initial_mean, dtype = dtype)
        
        if initial_covariance is None:
            self.initial_covariance = self.xp.eye(self.n_dim_sys, dtype = dtype)
        else:
            self.initial_covariance = self.xp.asarray(initial_covariance, dtype = dtype)

        if transition_matrices is None:
            self.F = self.xp.eye(self.n_dim_sys, dtype = dtype)
        else:
            self.F = self.xp.asarray(transition_matrices, dtype = dtype)

        if transition_covariance is not None:
            self.Q = self.xp.asarray(transition_covariance, dtype = dtype)
        else:
            self.Q = self.xp.eye(self.n_dim_sys, dtype = dtype)

        if observation_matrices is None:
            self.H = self.xp.eye(self.n_dim_obs, self.n_dim_sys, dtype = dtype)
        else:
            self.H = self.xp.asarray(observation_matrices, dtype = dtype)
        
        if observation_covariance is None:
            self.R = self.xp.eye(self.n_dim_obs, dtype = dtype)
        else:
            self.R = self.xp.asarray(observation_covariance, dtype = dtype)

        if observation_transition_matrix_mean is None:
            self.G_mean = self.xp.eye(self.n_dim_obs, dtype=dtype)
        else:
            self.G_mean = self.xp.asarray(observation_transition_matrix_mean, dtype=dtype)

        if observation_transition_matrix_precision is None:
            self.G_precision = self.xp.ones((self.n_dim_obs, self.n_dim_obs), dtype=dtype)
        else:
            self.G_precision = self.xp.asarray(observation_transition_matrix_precision, dtype=dtype)
        self.F_precision = self.xp.linalg.pinv(self.H) @ self.G_precision @ self.H

        if adjacency_matrix is None:
            # if adjacency matrix is None, then not localization mode.
            self.lmode_on = False
        else:
            self.A = self.xp.asarray(adjacency_matrix, dtype = bool)
            self.lmode_on = True

        if method in ["elementwise", "local-average", "all-average"]:
            self.method = method
        else:
            raise ValueError("Variable \"method\" only allows \"elementwise\", \"local-average\" "
                + "or \"all-average\". So, your setting \"{}\" need to be changed.".format(method))

        self.update_interval = int(update_interval)

        if save_directory is None:
            self.save_change = False
        else:
            self.save_change = True
            self.save_directory = save_directory
            self.fillnum = len(str(int(self.y.shape[0] / self.update_interval)))
            self.xp.save(os.path.join(self.save_directory, 
                "transition_matrix_" + str(0).zfill(self.fillnum) + ".npy"), self.F)
            self.xp.save(os.path.join(self.save_directory, 
                "transition_matrix_pre_" + str(0).zfill(self.fillnum) + ".npy"), self.F_precision)
        self.tm_count = 1

        self.accumulation_dic = {}
        for char in ["F", "G", "F_precision", "G_precision", "gamma"]:
            if char in accumulation:
                self.accumulation_dic[char] = []

        if "F" in self.accumulation_dic:
            self.accumulation_dic["F"].append(self.F)
        if "G" in self.accumulation_dic:
            self.accumulation_dic["G"].append(self.G_mean)
        if "F_precision" in self.accumulation_dic:
            self.accumulation_dic["F_precision"].append(self.F_precision)
        if "G_precision" in self.accumulation_dic:
            self.accumulation_dic["G_precision"].append(self.G_precision)

        if num_cpu == "all":
            self.num_cpu = mp.cpu_count()
        else:
            self.num_cpu = num_cpu

        self.save_state_interval = save_state_interval
        self.advance_mode = advance_mode
        self.precision_weight = precision_weight
        self.adjust_precision_on = adjust_precision_on
        self.dtype = dtype
        self.times = self.xp.zeros(5)


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
        # self.V_pred = self.xp.zeros((T, self.n_dim_sys, self.n_dim_sys),
        #      dtype = self.dtype)
        self.x_filt = self.xp.zeros((T, self.n_dim_sys), dtype = self.dtype)
        # self.V_filt = self.xp.zeros((T, self.n_dim_sys, self.n_dim_sys),
        #      dtype = self.dtype)

        # calculate prediction and filter for every time
        for t in range(T):
            # visualize calculating time
            print("\r filter calculating... t={}".format(t) + "/" + str(T), end="")

            if t == 0:
                # initial setting
                self.x_pred[0] = self.initial_mean
                self.V_pred = self.initial_covariance.copy()
            else:
                # change 191016
                if self.advance_mode and t-1<T-self.update_interval and (t-1)%self.update_interval==0:
                # if self.advance_mode and t<T-self.update_interval and t%self.update_interval==0:
                    self._update_transition_matrix(t+self.update_interval-1)
                start_time = time.time()
                self._predict_update(t)
                self.times[0] = time.time() - start_time
            
            # If y[t] has any mask, skip filter calculation
            # if self.xp.any(self.xp.ma.getmask(self.y[t])) :
            if self.xp.any(self.xp.isnan(self.y[t])):
                self.x_filt[t] = self.x_pred[t]
                self.V_filt = self.V_pred
            else :
                start_time = time.time()
                self._filter_update(t)
                self.times[1] = time.time() - start_time
                if (not self.advance_mode) and t-1>self.update_interval and (t-1)%self.update_interval==0:
                # if (not self.advance_mode) and t>self.update_interval and t%self.update_interval==0:
                    self._update_transition_matrix(t)


        if self.save_change:
            self.xp.save(os.path.join(self.save_directory, "states.npy"), self.x_filt[::self.save_state_interval])


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
        self.V_pred = F @ self.V_filt @ F.T + Q


    def _filter_update(self, t):
        """Calculate fileter update without noise

        Args:
            t {int} : observation time

        Attributes:
            K [n_dim_sys, n_dim_obs] {numpy-array, float}
                : Kalman gain matrix for time t [状態変数軸，観測変数軸]
                カルマンゲイン
        """
        # extract parameters for time t
        H = _last_dims(self.H, t, 2)
        R = _last_dims(self.R, t, 2)

        # calculate filter step
        K = self.V_pred @ (
            H.T @ self.xp.linalg.inv(H @ (self.V_pred @ H.T) + R)
            )
        # target = self.xp.isnan(self.y[t])
        # self.y[t][target] = (H @ self.x_pred[t])[target]
        self.x_filt[t] = self.x_pred[t] + K @ (
            self.y[t] - (H @ self.x_pred[t])
            )
        # self.y[t][target] = (H @ self.x_filt[t])[target]
        self.V_filt = self.V_pred - K @ (H @ self.V_pred)


    def _update_transition_matrix(self, t):
        """Update transition matrix

        Args:
            t {int} : observation time
        """
        H = _last_dims(self.H, t, 2)
        R = _last_dims(self.R, t, 2)
        Q = _last_dims(self.Q, t, 2)

        # common calculation
        H_pseudo = self.xp.linalg.pinv(H)
        # S_inverse = self.xp.linalg.pinv(R + H @ Q @ H.T - H_pseudo @ self.F @ H @ R @ H.T @ self.F.T @ H_pseudo.T)
        S_inverse = self.xp.linalg.pinv(R + H @ (Q + self.F @ H_pseudo @ R @ H_pseudo.T @ self.F.T) @ H.T)

        start_time = time.time()
        if self.lmode_on:
            if self.method=="elementwise": # elementwise
                Zbb = self.xp.zeros((self.n_dim_obs, self.n_dim_obs), dtype=self.dtype)

                if True:
                    if self.use_gpu:
                        A = self.A.get()
                        y = self.y[t-self.update_interval:t+1].get()
                        S_inverse = S_inverse.get()
                        G_mean = self.G_mean.get()
                        G_precision = self.G_precision.get()
                    else:
                        A = self.A
                        y = self.y[t-self.update_interval:t+1]
                        G_mean = self.G_mean
                        G_precision = self.G_precision
                    where_is_A = np.where(A)

                    p = mp.Pool(self.num_cpu)
                    #G_mean_seq, Zbb_seq
                    G_mean_seq = p.starmap(_local_calculation, zip(where_is_A[0],
                                                                where_is_A[1],
                                                                itertools.repeat(A),
                                                                itertools.repeat(y),
                                                                itertools.repeat(S_inverse),
                                                                itertools.repeat(G_mean),
                                                                itertools.repeat(G_precision)))
                    p.close()
                    self.G_mean[A] = G_mean_seq
                    Zbb = self.y[t-self.update_interval:t].T @ self.y[t-self.update_interval:t]
                    # Zbb[A] = Zbb_seq
                else:
                    where_is_A = self.xp.where(self.A)
                    for i,j in zip(where_is_A[0], where_is_A[1]):
                        self.G_mean[i,j], Zbb[i,j] = _local_calculation(i, j, self.A, 
                                                            self.y[t-self.update_interval:t+1], S_inverse, 
                                                            self.G_mean, self.G_precision, xp=self.xp)

            elif self.method=="local-average": # local-average
                for i in range(self.n_dim_obs):
                    local_node_number = len(self.xp.where(self.A[i][:i])[0]) #LA
                    global_node_number = self.xp.where(self.A[i])[0]
                    Gh = self.y[t-self.update_interval+1:t+1, global_node_number].T \
                            @ self.xp.linalg.pinv(self.y[t-self.update_interval:t, global_node_number].T)
                    G[i, global_node_number] += Gh[local_node_number] #LA
                    G[global_node_number, i] += Gh[:, local_node_number] #LA
                G /= 2.0 #LA
            elif self.method=="all-average": #all-average
                C = self.xp.zeros((self.n_dim_obs, self.n_dim_obs), dtype=self.dtype) #AA
                for i in range(self.n_dim_obs):
                    global_node_number = self.xp.where(self.A[i])[0]
                    Gh = self.y[t-self.update_interval+1:t+1, global_node_number].T \
                            @ self.xp.linalg.pinv(self.y[t-self.update_interval:t, global_node_number].T)
                    G[self.xp.ix_(global_node_number, global_node_number)] += Gh #AA
                    C[self.xp.ix_(global_node_number, global_node_number)] += 1 #AA
                C[C==0] = 1 #AA
                G /= C #AA

            self.times[2] += time.time() - start_time
        else:
            # Fh = self.xp.linalg.pinv(H) @ self.y[t-self.update_interval+1:t+1].T \
            #         @ self.xp.linalg.pinv(self.y[t-self.update_interval:t].T) @ Hb
            Znb = self.y[t-self.update_interval+1:t+1].T @ self.y[t-self.update_interval:t]
            Zbb = self.y[t-self.update_interval:t].T @ self.y[t-self.update_interval:t]
            # C = self.xp.zeros((self.n_dim_obs**2, self.n_dim_obs**2), dtype=self.dtype)
            # beta = self.xp.zeros(self.n_dim_obs**2, dtype=self.dtype)

            # for i in range(self.n_dim_obs**2):
            #     iq = i//self.n_dim_obs
            #     ir = i%self.n_dim_obs
            #     for j in range(self.n_dim_obs**2):
            #         jq = j//self.n_dim_obs
            #         jr = j%self.n_dim_obs
            #         C[i,j] = S_inverse[iq,jq]*Zbb[ir,jr]
            #         if i==j:
            #             C[i,j] += self.G_precision[iq,ir]
            #     beta[i] = S_inverse[iq] @ Znb[:,ir] + self.G_mean[iq,ir] * self.G_precision[iq,ir]

            C = self.xp.kron(S_inverse, Zbb) + self.xp.diag(self.G_precision.reshape(-1))
            beta = (S_inverse @ Znb + self.G_mean * self.G_precision).reshape(-1)
            self.G_mean = (self.xp.linalg.pinv(C) @ beta).reshape(self.n_dim_obs, self.n_dim_obs) #ok
        

        ## common calculation
        if self.adjust_precision_on:
            if self.tm_count==1:
                self.G_precision = self.G_precision \
                        + self.xp.outer(self.xp.diag(S_inverse), self.xp.diag(Zbb))
            else:
                add_term = self.xp.outer(self.xp.diag(S_inverse), self.xp.diag(Zbb))
                adjust_coefficient = self.xp.minimum(self.xp.absolute(add_term).mean() \
                                    / self.xp.absolute(self.G_precision).mean(), 1)
                self.G_precision *= self.precision_weight * adjust_coefficient
                self.G_precision = self.precision_weight * adjust_coefficient * self.G_precision \
                                    + add_term 
        else:
            self.G_precision += self.xp.outer(self.xp.diag(S_inverse), self.xp.diag(Zbb))
            # self.G_precision = self.precision_weight * self.G_precision \
            #             + self.xp.outer(self.xp.diag(S_inverse), self.xp.diag(Zbb))
            

        self.F = H_pseudo @ self.G_mean @ H
        self.F_precision = H_pseudo @ self.G_precision @ H

        self.times[3] += time.time() - start_time
        self.times[4] += 1


        if self.save_change:
            self.xp.save(os.path.join(self.save_directory, 
                                    "transition_matrix_" + str(self.tm_count).zfill(self.fillnum) + ".npy"), self.F)
            self.xp.save(os.path.join(self.save_directory, 
                                    "transition_matrix_pre_" + str(self.tm_count).zfill(self.fillnum) + ".npy"), self.F_precision)

        if "F" in self.accumulation_dic:
            self.accumulation_dic["F"].append(self.F)
        if "G" in self.accumulation_dic:
            self.accumulation_dic["G"].append(self.G_mean)
        if "F_precision" in self.accumulation_dic:
            self.accumulation_dic["F_precision"].append(self.F_precision)
        if "G_precision" in self.accumulation_dic:
            self.accumulation_dic["G_precision"].append(self.G_precision)
        if "gamma" in self.accumulation_dic and self.adjust_precision_on and self.tm_count!=1:
            self.accumulation_dic["gamma"].append(adjust_coefficient)

        self.tm_count += 1

        if not self.adjust_precision_on:
            self.G_precision *= self.precision_weight



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
        if "F" in self.accumulation_dic:
            if ids is None:
                return self.accumulation_dic["F"]
            else:
                return self.accumulation_dic["F"][ids]
        else:
            return self.F


    def smooth(self):
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

        # if not implement `filter`, implement `filter`
        try :
            self.x_pred[0]
        except :
            self.filter()

        T = self.y.shape[0]
        self.x_smooth = self.xp.zeros((T, self.n_dim_sys), dtype = self.dtype)
        self.V_smooth = self.xp.zeros((T, self.n_dim_sys, self.n_dim_sys),
             dtype = self.dtype)
        A = self.xp.zeros((self.n_dim_sys, self.n_dim_sys), dtype = self.dtype)

        self.x_smooth[-1] = self.x_filt[-1]
        self.V_smooth[-1] = self.V_filt[-1]

        # t in [0, T-2] (notice t range is reversed from 1~T)
        for t in reversed(range(T - 1)) :
            # visualize calculating times
            print("\r smooth calculating... t={}".format(T - t)
                 + "/" + str(T), end="")

            # extract parameters for time t
            F = _last_dims(self.F, t, 2)

            # calculate fixed interval smoothing gain
            A = self.xp.dot(self.V_filt[t], self.xp.dot(F.T, self.xp.linalg.pinv(self.V_pred[t + 1])))
            
            # fixed interval smoothing
            self.x_smooth[t] = self.x_filt[t] \
                + self.xp.dot(A, self.x_smooth[t + 1] - self.x_pred[t + 1])
            self.V_smooth[t] = self.V_filt[t] \
                + self.xp.dot(A, self.xp.dot(self.V_smooth[t + 1] - self.V_pred[t + 1], A.T))

            
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
