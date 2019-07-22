# Copyright (c) The pyakalman developers.
# All rights reserved.
"""
utility functions
"""

import numpy as np
try:
    import cupy
    xp = cupy
except:
    xp = np


def _determine_dimensionality(variables, default = None):
    """Derive the dimensionality of the state space
    Parameters
    ----------
    variables : list of ({None, array}, conversion function, index)
        variables, functions to convert them to arrays, and indices in those
        arrays to derive dimensionality from.

    default : {None, int}
        default dimensionality to return if variables is empty
    
    Returns
    -------
    dim : int
        dimensionality of state space as derived from variables or default.
    """

    # gather possible values based on the variables
    candidates = []
    for (v, converter, idx) in variables:
        if v is not None:
            v = converter(v)
            candidates.append(v.shape[idx])

    # also use the manually specified default
    if default is not None:
        candidates.append(default)

    # ensure consistency of all derived values
    # If dimensionality of candidates doesn't have consistency,
    # raise ValueError
    if len(candidates) == 0:
        return 1
    else:
        if not xp.all(xp.array(candidates) == candidates[0]):
            print(candidates)
            raise ValueError(
                "The shape of all " +
                "parameters is not consistent.  " +
                "Please re-check their values."
            )
        return candidates[0]


def _parse_observations(obs):
    """Safely convert observations to their expected format"""
    obs = xp.ma.atleast_2d(obs)

    # 2軸目の方が大きい場合は，第1軸と第2軸を交換
    if obs.shape[0] == 1 and obs.shape[1] > 1:
        obs = obs.T

    # 欠測値をマスク処理
    obs = xp.ma.array(obs, mask = xp.isnan(obs))
    return obs


def _last_dims(X, t, ndims = 2):
    """Extract the final dimensions of `X`
    Extract the final `ndim` dimensions at index `t` if `X` has >= `ndim` + 1
    dimensions, otherwise return `X`.
    Parameters
    ----------
    X : array with at least dimension `ndims`
    t : int
        index to use for the `ndims` + 1th dimension
    ndims : int, optional
        number of dimensions in the array desired

    Returns
    -------
    Y : array with dimension `ndims`
        the final `ndims` dimensions indexed by `t`
    """
    X = xp.asarray(X)
    if len(X.shape) == ndims + 1:
        return X[t]
    elif len(X.shape) == ndims:
        return X
    else:
        raise ValueError(("X only has %d dimensions when %d" +
                " or more are required") % (len(X.shape), ndims))


# calculate transition covariance
def _calc_transition_covariance(self, G, Q):
    """Calculate transition covariance

    Args:
        G [n_time - 1, n_dim_sys, n_dim_noise] or [n_dim_sys, n_dim_noise]
            {numpy-array, float}
            transition noise matrix
            ノイズ変換行列[時間軸，状態変数軸，ノイズ変数軸] or [状態変数軸，ノイズ変数軸]
        Q [n_time - 1, n_dim_noise, n_dim_noise] or [n_dim_sys, n_dim_noise]
            {numpy-array, float}
            system transition covariance for times
            システムノイズの共分散行列[時間軸，ノイズ変数軸，ノイズ変数軸]
    """
    if G.ndim == 2:
        GT = G.T
    elif G.ndim == 3:
        GT = G.transpose(0,2,1)
    else:
        raise ValueError('The ndim of transition_noise_matrices'
            + ' should be 2 or 3,' + ' but your input is ' + str(G.ndim) + '.')
    if Q.ndim == 2 or Q.ndim == 3:
        return xp.matmul(G, xp.matmul(Q, GT))
    else:
        raise ValueError('The ndim of transition_covariance should be 2 or 3,'
            + ' but your input is ' + str(Q.ndim) + '.')


# calculate MSE
def mean_squared_error(x, y):
    assert x.shape == y.shape
    return xp.sqrt(xp.sum(xp.square(x - y))) / x.size


# calculate MAE
def mean_absolute_error(x, y):
    assert x.shape == y.shape
    return xp.mean(xp.absolute(x - y))
