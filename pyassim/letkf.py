# local ensemble transform kalman filter
'''
18.04.02
- ローカルへの変換は隣接行列として実装
18.04.06
- multiprocessing による並列化処理
　心なしかあまり早くなった気がしない
'''


# install packages
import math
import itertools

import numpy as np
import numpy.random as rd
import pandas as pd

from scipy import linalg

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
        大域変数を局所変数に写像するための隣接行列
    global_variable [n_dim] {numpy-array, float}
        : global variable
        大域変数
    '''
    return global_variable[adjacency_matrix]


# parallel processing for local calculation
def local_multi_processing(x_pred_mean_local, x_pred_center_local,
        y_background_mean_local, y_background_center_local, y_local,
        R_local, i, A_sys, n_particle, rho):
    ## Step4 : calculate matrix C
    # R はここでしか使わないので，線型方程式 R C^T=Y を解く方が速いかもしれない
    # R が時不変なら毎回逆行列計算するコスト抑制をしても良い
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

    # analysis_weight_matrix が対称なら転置とる必要がなくなる
    # particle x particle
    analysis_weight_ensemble = (analysis_weight_matrix.T + analysis_weight_mean).T


    ## Step8 : calculate analysis system variable in model space
    # 転置が多くて少し気持ち悪い
    # local_sys x particle
    analysis_system = (x_pred_mean_local + (
        x_pred_center_local @ analysis_weight_ensemble
        ).T).T


    ## Step9 : move analysis result to global analysis
    # sys x particle
    return analysis_system[len(np.where(A_sys[:i])[0])]


class LocalEnsembleTransformKalmanFilter(object):
    '''
    Local Ensemble Transform Kalman Filter のクラス

    <Input Variables>
    y, observation [n_time, n_dim_obs] {numpy-array, float}
        : observation y
        観測値 [時間軸,観測変数軸]
    initial_mean [n_dim_sys] {float} 
        : initial state mean
        初期状態分布の期待値 [状態変数軸]
    f, transition_functions [n_time] {function}
        : transition function from x_{t-1} to x_t
        システムモデルの遷移関数 [時間軸] or []
    h, observation_functions [n_time] {function}
        : observation function from x_t to y_t
        観測関数 [時間軸] or []
    q, transition_noise [n_time - 1] {(method, parameters)}
        : transition noise for v_t
        システムノイズの発生方法とパラメータ [時間軸]
        サイズは指定できる形式
    R, observation_covariance [n_time, n_dim_obs, n_dim_obs] {numpy-array, float} 
        : covariance of observation normal noise
        観測正規ノイズの共分散行列 [時間軸，観測変数軸，観測変数軸]
    A_sys, system_adjacency_matrix [n_dim_sys, n_dim_sys] {numpy-array, int}
        : adjacency matrix of system variables
        システム変数の隣接行列 [状態変数軸，状態変数軸]
    A_obs, observation_adjacency_matrix [n_dim_obs, n_dim_obs] {numpy-array, int}
        : adjacency matrix of system variables
        観測変数の隣接行列 [観測変数軸，観測変数軸]
    rho {float} : multipliative covariance inflating factor
    n_particle {int} : number of particles (粒子数)
    n_dim_sys {int} : dimension of system variable （システム変数の次元）
    n_dim_obs {int} : dimension of observation variable （観測変数の次元）
    dtype {np.dtype} : numpy dtype (numpy のデータ型)
    seed {int} : random seed (ランダムシード)
    cpu_number {int} : cpu number for parallel processing (CPU数)
    '''

    def __init__(self, observation = None, transition_functions = None,
                observation_functions = None, initial_mean = None,
                transition_noise = None, observation_covariance = None,
                system_adjacency_matrix = None, observation_adjacency_matrix = None,
                rho = 1,
                n_particle = 100, n_dim_sys = None, n_dim_obs = None,
                dtype = np.float32, seed = 10, cpu_number = 'all') :

        # 次元数をチェック，欠測値のマスク処理
        self.y = _parse_observations(observation)

        # 次元決定
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
        T {int} : length of data y （時系列の長さ）
        <class variables>
        x_pred_mean [n_time+1, n_dim_sys] {numpy-array, float}
            : mean of x_pred regarding to particles at time t
            時刻 t における x_pred の粒子平均 [時間軸，状態変数軸]
        x_filt [n_time+1, n_dim_sys, n_particle] {numpy-array, float}
            : hidden state at time t given observations for each particle
            状態変数のフィルタアンサンブル [時間軸，状態変数軸，粒子軸]
        x_filt_mean [n_time+1, n_dim_sys] {numpy-array, float}
            : mean of x_filt regarding to particles
            時刻 t における状態変数のフィルタ平均 [時間軸，状態変数軸]

        <global variables>
        x_pred [n_dim_sys, n_particle] {numpy-array, float}
            : hidden state at time t given observations for each particle
            状態変数の予測アンサンブル [状態変数軸，粒子軸]
        x_pred_center [n_dim_sys, n_particle] {numpy-array, float}
            : centering of x_pred
            x_pred の中心化 [状態変数軸，粒子軸]
        v [n_dim_sys, n_particle] {numpy-array, float}
            : system noise for transition
            システムノイズ [状態変数軸，粒子軸]
        y_background [n_dim_obs, n_particle] {numpy-array, float}
            : background value of observation space
            観測空間における背景値(=一気先予測値) [観測変数軸，粒子軸]
        y_background_mean [n_dim_obs] {numpy-array, float}
            : mean of y_background for particles
            y_background の粒子平均 [観測変数軸]
        y_background_center [n_dim_obs, n_particle] {numpy-array, float}
            : centerized value of y_background for particles
            y_background の粒子中心化 [観測変数軸，粒子軸]

        <local variables>
        x_pred_mean_local [n_dim_local] {numpy-array, float}
            : localization from x_pred_mean
            x_pred_mean の局所値 [局所変数軸]
        x_pred_center_local [n_dim_local, n_particle] {numpy-array, float}
            : localization from x_pred_center
            x_pred_center の局所値 [局所変数軸，粒子軸]
        y_background_mean_local [n_dim_local] {numpy-array, float}
            : localization from y_background_mean
            y_background_mean の局所値 [局所変数軸]
        y_background_center_local [n_dim_local, n_particle] {numpy-array, float}
            : localization from y_background_center
            y_background_center の局所値 [局所変数軸，粒子軸]
        y_local [n_dim_local] {numpy-array, float}
            : localization from observation y
            観測 y の局所値 [局所変数軸]
        R_local [n_dim_local, n_dim_local] {numpy-array, float}
            : localization from observation covariance R
            観測共分散 R の局所値 [局所変数軸，局所変数軸]
        '''

        # 時系列の長さ, lenght of time-series
        T = self.y.shape[0]

        ## 配列定義, definition of array
        # 時刻0における予測・フィルタリングは初期値, initial setting
        self.x_pred_mean = np.zeros((T + 1, self.n_dim_sys), dtype = self.dtype)
        self.x_filt = np.zeros((T + 1, self.n_dim_sys, self.n_particle),
             dtype = self.dtype)
        self.x_filt[0].T[:] = self.initial_mean
        self.x_filt_mean = np.zeros((T + 1, self.n_dim_sys), dtype = self.dtype)

        # 初期値のセッティング, initial setting
        self.x_pred_mean[0] = self.initial_mean
        self.x_filt_mean[0] = self.initial_mean

        # 各時刻で予測・フィルタ計算, prediction and filtering
        for t in range(T):
            # 計算している時間を可視化, visualization for calculation
            print('\r filter calculating... t={}'.format(t+1) + '/' + str(T), end='')

            ## filter update
            # 一期先予測, prediction
            f = _last_dims(self.f, t, 1)[0]

            # システムノイズをパラメトリックに発生, raise parametric system noise
            # sys x particle
            v = self.q[0](*self.q[1], size = self.n_particle).T

            # アンサンブル予測, ensemble prediction
            # sys x particle
            x_pred = f(*[self.x_filt[t], v])

            # x_pred_mean を計算, calculate x_pred_mean
            # time x sys
            self.x_pred_mean[t + 1] = np.mean(x_pred, axis = 1)

            # 欠測値の対処, treat missing values
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
                # global を local に移す写像（並列処理）
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

            # フィルタ分布のアンサンブル平均の計算
            self.x_filt_mean[t + 1] = np.mean(self.x_filt[t + 1], axis = 1)


    # get predicted value (一期先予測値を返す関数, Filter 関数後に値を得たい時)
    def get_predicted_value(self, dim = None) :
        # filter されてなければ実行
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


    # get filtered value (フィルタ値を返す関数，Filter 関数後に値を得たい時)
    def get_filtered_value(self, dim = None) :
        # filter されてなければ実行
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


    # parallel calculation for global to local (大域変数を局所変数に移すための計算)
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

