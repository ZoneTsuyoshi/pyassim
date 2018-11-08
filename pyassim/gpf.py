'''
gaussian particle filter のクラス
18.05.15
- フィルター作成
'''

import numpy as np
import numpy.random as rd
from scipy import linalg

from .utils import array1d, array2d, check_random_state, get_params, \
    preprocess_arguments, check_random_state
from .util_functions import _parse_observations, _last_dims, \
    _determine_dimensionality

class GaussianParticleFilter(object):
    '''
    Gaussian Particle Filter のクラス

    <Input Variables>
    y, observation [n_time, n_dim_obs] {numpy-array, float}
        : observation y
        観測値 [時間軸,観測変数軸]
    initial_mean [n_dim_sys] {float} 
        : initial state mean
        初期状態分布の期待値 [状態変数軸]
    initial_covariance [n_dim_sys, n_dim_sys] {numpy-array, float} 
        : initial state covariance
        初期状態分布の共分散行列[状態変数軸，状態変数軸]
    f, transition_functions [n_time] {function}
        : transition function from x_{t-1} to x_t
        システムモデルの遷移関数 [時間軸] or []
    q, transition_noise [n_time - 1] {(method, parameters)}
        : transition noise for v_t
        システムノイズの発生方法とパラメータ [時間軸]
        サイズは指定できる形式
    lf, likelihood_functions [n_time] or [] {function}
        : likelihood function between x_t and y_t
        観測モデルの尤度関数 [時間軸] or []
    lfp, likelihood_function_parameters [n_time, n_param] or [n_param]
     {numpy-array, float}
        : parameters for likelihood function
        尤度関数のパラメータ群 [時間軸，パラメータ軸] or [パラメータ軸]
    likelihood_function_is_log_form {boolean}
        : likelihood functions are log form?
        尤度関数の対数形式の有無
    observation_parameters_time_invariant {boolean}
        : time-invariantness of observation parameters
        観測パラメータの時不変性の有無
    n_particle {int} : number of particles (粒子数)
    n_dim_sys {int} : dimension of system variable （システム変数の次元）
    n_dim_obs {int} : dimension of observation variable （観測変数の次元）
    dtype {np.dtype} : numpy dtype (numpy のデータ型)
    seed {int} : random seed (ランダムシード)

    <Variables>
    regularization {boolean}
        : which regularization is True
        正則化項の導入の有無
    '''
    def __init__(self, observation = None, 
                initial_mean = None, initial_covariance = None,
                transition_functions = None, transition_noise = None,
                likelihood_functions = None, likelihood_function_parameters = None,
                likelihood_function_is_log_form = True,
                observation_parameters_time_invariant = True,
                n_particle = 100, n_dim_sys = None, n_dim_obs = None,
                dtype = np.float32, seed = 10) :

        # 次元数をチェック，欠測値のマスク処理
        self.y = _parse_observations(observation)

        # 次元決定
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

            # 正規尤度は用いるが，パラメータRだけ変えたい場合
            if likelihood_functions is None:
                self.lfp = [np.eye(self.n_dim_obs, dtype = dtype)]
            else:
                self.lfp = likelihood_function_parameters

        self.n_particle = n_particle
        np.random.seed(seed)
        self.dtype = dtype
        self.log_likelihood = - np.inf
    

    # likelihood for normal distribution
    # 正規分布の尤度 (カーネルのみ)
    def _norm_likelihood(self, y, mean, covariance):
        '''
        y [n_dim_obs] {numpy-array, float} 
            : observation
            観測 [観測変数軸]
        mean [n_particle, n_dim_obs] {numpy-array, float}
            : mean of normal distribution
            各粒子に対する正規分布の平均 [粒子軸，観測変数軸]
        covariance [n_dim_obs, n_dim_obs] {numpy-array, float}
            : covariance of normal distribution
            正規分布の共分散 [観測変数軸]
        '''
        Y = np.zeros((self.n_dim_obs, self.n_particle), dtype = self.dtype)
        Y.T[:] = y
        return np.exp((- 0.5 * (Y - mean).T @ linalg.pinv(covariance) \
                @ (Y - mean))[:, 0])
    

    # log likelihood for normal distribution
    # 正規分布の対数尤度 (カーネルのみ)
    def _log_norm_likelihood(self, y, mean, covariance) :
        '''
        y [n_dim_obs] {numpy-array, float} 
            : observation
            観測 [観測変数軸]
        mean [n_particle, n_dim_obs] {numpy-array, float}
            : mean of normal distribution
            各粒子に対する正規分布の平均 [粒子軸，観測変数軸]
        covariance [n_dim_obs, n_dim_obs] {numpy-array, float}
            : covariance of normal distribution
            正規分布の共分散 [観測変数軸]
        '''
        Y = np.zeros((self.n_dim_obs, self.n_particle), dtype = self.dtype)
        Y.T[:] = y
        return (
            - 0.5 * (Y - mean).T @ linalg.pinv(covariance) @ (Y - mean)
            ).diagonal()
    

    # filtering
    def filter(self):
        '''
        T {int} : 時系列の長さ，length of y
        x_pred_mean [n_time+1, n_dim_sys] {numpy-array, float}
            : mean of x_pred regarding to particles at time t
            時刻 t における x_pred の粒子平均 [時間軸，状態変数軸]
        x_filt_mean [n_time+1, n_dim_sys] {numpy-array, float}
            : mean of x_filt regarding to particles
            時刻 t における状態変数のフィルタ平均 [時間軸，状態変数軸]

        x_pred [n_dim_sys, n_particle]
            : hidden state at time t given observations for each particle
            状態変数の予測アンサンブル [状態変数軸，粒子軸]
        x_filt [n_dim_sys, n_particle] {numpy-array, float}
            : hidden state at time t given observations for each particle
            状態変数のフィルタアンサンブル [状態変数軸，粒子軸]

        w [n_particle] {numpy-array, float}
            : weight lambda of each particle
            各粒子の尤度 [粒子軸]
        v [n_dim_sys, n_particle] {numpy-array, float}
            : system noise particles
            各時刻の状態ノイズ [状態変数軸，粒子軸]
        '''

        # 時系列の長さ, number of time-series data
        T = len(self.y)
        
        # initial filter, prediction
        self.x_pred_mean = np.zeros((T + 1, self.n_dim_sys), dtype = self.dtype)
        self.x_filt_mean = np.zeros((T + 1, self.n_dim_sys), dtype = self.dtype)
        x_pred = np.zeros((self.n_dim_sys, self.n_particle), dtype = self.dtype)

        # initial distribution
        # sys x particle
        x_filt = rd.multivariate_normal(self.initial_mean, self.initial_covariance, 
            size = self.n_particle).T
        
        # initial setting
        self.x_pred_mean[0] = self.initial_mean
        self.x_filt_mean[0] = self.initial_mean

        for t in range(T):
            print("\r filter calculating... t={}".format(t), end="")
            
            ## filter update
            # 一期先予測, prediction
            f = _last_dims(self.f, t, 1)[0]

            # システムノイズをパラメトリックに発生, raise parametric system noise
            v = self.q[0](*self.q[1], size = self.n_particle).T

            # アンサンブル予測, ensemble prediction
            # sys x particle
            x_pred = f(*[x_filt, v])

            # mean
            # sys
            self.x_pred_mean[t + 1] = np.mean(x_pred, axis = 1)
            
            # 欠測値の対処, treat missing values
            if np.any(np.ma.getmask(self.y[t])):
                x_filt = x_pred
                self.x_filt_mean[t + 1] = self.x_pred_mean[t + 1]
            else:
                # log likelihood for each particle for y[t]
                # 対数尤度の計算
                lf = self._last_likelihood(self.lf, t)
                lfp = self._last_likelihood(self.lfp, t)
                try:
                    # particle
                    w = lf(self.y[t], x_pred, *lfp)
                except:
                    raise ValueError('you must check likelihood_functions' 
                        + 'and parameters.')
                
                # 発散防止, avoid evaporation
                if self.likelihood_function_is_log_form:
                    w = np.exp(w - np.max(w))
                else:
                    w = w / np.max(w)

                # 重みの正規化, normalization
                # particle
                w = w / np.sum(w)

                # 重み付き平均の計算
                # sys
                self.x_filt_mean[t + 1] = w @ x_pred.T

                # 重み付き共分散の計算
                # sys x particle
                x_pred.T[:] -= self.x_filt_mean[t + 1]

                # sys x sys
                V_filt = np.dot(np.multiply(x_pred, w), x_pred.T)

                # sys x particle
                x_filt = rd.multivariate_normal(self.x_filt_mean[t + 1],
                    V_filt, size = self.n_particle).T

        
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


    # # メモリ節約のため，filter のオーバーラップ
    # def smooth(self, lag = 10):
    #     '''
    #     lag {int} : ラグ，lag of smoothing
    #     T {int} : 時系列の長さ，length of y

    #     x_pred_mean [n_time+1, n_dim_sys] {numpy-array, float}
    #         : mean of x_pred regarding to particles at time t
    #         時刻 t における x_pred の粒子平均 [時間軸，状態変数軸]
    #     x_filt_mean [n_time+1, n_dim_sys] {numpy-array, float}
    #         : mean of x_filt regarding to particles at time t
    #         時刻 t における状態変数のフィルタ平均 [時間軸，状態変数軸]
    #     x_smooth_mean [n_time, n_dim_sys] {numpy-array, float}
    #         : mean of x_smooth regarding to particles at time t
    #         時刻 t における状態変数の平滑化平均 [時間軸，状態変数軸]

    #     x_pred [n_dim_sys, n_particle]
    #         : hidden state at time t given observations[:t-1] for each particle
    #         状態変数の予測アンサンブル [状態変数軸，粒子軸]
    #     x_filt [n_particle, n_dim_sys] {numpy-array, float}
    #         : hidden state at time t given observations[:t] for each particle
    #         状態変数のフィルタアンサンブル [状態変数軸，粒子軸]
    #     x_smooth [n_time, n_dim_sys, n_particle] {numpy-array, float}
    #         : hidden state at time t given observations[:t+lag] for each particle
    #         状態変数の平滑化アンサンブル [時間軸，状態変数軸，粒子軸]

    #     w [n_particle] {numpy-array, float}
    #         : weight lambda of each particle
    #         各粒子の尤度 [粒子軸]
    #     v [n_dim_sys, n_particle] {numpy-array, float}
    #         : system noise particles
    #         各時刻の状態ノイズ [状態変数軸，粒子軸]
    #     '''

    #     # 時系列の長さ, number of time-series data
    #     T = len(self.y)
        
    #     # initial filter, prediction
    #     self.x_pred_mean = np.zeros((T + 1, self.n_dim_sys), dtype = self.dtype)
    #     self.x_filt_mean = np.zeros((T + 1, self.n_dim_sys), dtype = self.dtype)
    #     self.x_smooth_mean = np.zeros((T + 1, self.n_dim_sys), dtype = self.dtype)
    #     x_pred = np.zeros((self.n_dim_sys, self.n_particle), dtype = self.dtype)
    #     x_filt = np.zeros((self.n_dim_sys, self.n_particle), dtype = self.dtype)
    #     x_smooth = np.zeros((T + 1, self.n_dim_sys, self.n_particle),
    #          dtype = self.dtype)

    #     # initial distribution
    #     x_filt = rd.multivariate_normal(self.initial_mean, self.initial_covariance, 
    #         size = self.n_particle).T
        
    #     # initial setting
    #     self.x_pred_mean[0] = self.initial_mean
    #     self.x_filt_mean[0] = self.initial_mean
    #     self.x_smooth_mean[0] = self.initial_mean

    #     for t in range(T):
    #         print("\r filter and smooth calculating... t={}".format(t), end="")
            
    #         ## filter update
    #         # 一期先予測, prediction
    #         f = _last_dims(self.f, t, 1)[0]

    #         # システムノイズをパラメトリックに発生, raise parametric system noise
    #         v = self.q[0](*self.q[1], size = self.n_particle).T

    #         # アンサンブル予測, ensemble prediction
    #         x_pred = f(*[x_filt, v])

    #         # mean
    #         self.x_pred_mean[t + 1] = np.mean(x_pred, axis = 1)
            
    #         # 欠測値の対処, treat missing values
    #         if np.any(np.ma.getmask(self.y[t])):
    #             x_filt = x_pred
    #         else:
    #             # log likelihood for each particle for y[t]
    #             # 対数尤度の計算
    #             lf = self._last_likelihood(self.lf, t)
    #             lfp = self._last_likelihood(self.lfp, t)
    #             try:
    #                 w = lf(self.y[t], x_pred, *lfp)
    #             except:
    #                 raise ValueError('you must check likelihood_functions'
    #                     + ' and parameters.')
                
    #             # 発散防止, avoid evaporation
    #             if self.likelihood_function_is_log_form:
    #                 w = np.exp(w - np.max(w))
    #             else:
    #                 w = w / np.max(w)
            
    #         # initial smooth value
    #         x_smooth[t + 1] = x_filt
            
    #         # mean
    #         self.x_filt_mean[t + 1] = np.mean(x_filt, axis = 1)

    #         # smoothing
    #         if (t > lag - 1) :
    #             x_smooth[t - lag:t + 1] = x_smooth[t - lag:t + 1, :, k]
    #         else :
    #             x_smooth[:t + 1] = x_smooth[:t + 1, :, k]

    #     # x_smooth_mean
    #     self.x_smooth_mean = np.mean(x_smooth, axis = 2)


    # get smoothed value
    def get_smoothed_value(self, dim = None) :
        # smooth されてなければ実行
        try :
            self.x_smooth_mean[0]
        except :
            self.smooth()

        if dim is None:
            return self.x_smooth_mean[1:]
        elif dim <= self.x_smooth_mean.shape[1]:
            return self.x_smooth_mean[1:, int(dim)]
        else:
            raise ValueError('The dim must be less than '
             + self.x_smooth_mean.shape[1] + '.')


    # last likelihood function and parameters (尤度関数とパラメータの決定)
    def _last_likelihood(self, X, t):
        """Extract the final dimensions of `X`
        Extract the final `ndim` dimensions at index `t` if `X` has >= `ndim` + 1
        dimensions, otherwise return `X`.
        Parameters
        ----------
        X : array with at least dimension `ndims`
        t : int
            index to use for the `ndims` + 1th dimension

        Returns
        -------
        Y : array with dimension `ndims`
            the final `ndims` dimensions indexed by `t`
        """
        if self.observation_parameters_time_invariant:
            return X
        else:
            try:
                return X[t]
            except:
                raise ValueError('you must check which likelihood ' + 
                    'parameters are time-invariant.')
