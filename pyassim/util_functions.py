# Copyright (c) The pyakalman developers.
# All rights reserved.
"""
utility functions
"""

import numpy as np
try:
    import cupy
except:
    pass


def judge_xp_type(xp_type = "numpy"):
    if xp_type in ["numpy", False]:
        return np
    elif xp_type in ["cupy", True]:
        return cupy


def _determine_dimensionality(variables, default = None, xp_type = "numpy"):
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
    xp = judge_xp_type(xp_type)

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


def _parse_observations(obs, xp_type="numpy"):
    """Safely convert observations to their expected format"""
    xp = judge_xp_type(xp_type)
    obs = xp.ma.atleast_2d(obs)

    # 2軸目の方が大きい場合は，第1軸と第2軸を交換
    if obs.shape[0] == 1 and obs.shape[1] > 1:
        obs = obs.T

    # 欠測値をマスク処理
    obs = xp.ma.array(obs, mask = xp.isnan(obs))
    return obs


def _last_dims(X, t, ndims = 2, xp_type="numpy"):
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
    xp = judge_xp_type(xp_type)
    X = xp.asarray(X)
    if len(X.shape) == ndims + 1:
        return X[t]
    elif len(X.shape) == ndims:
        return X
    else:
        raise ValueError(("X only has %d dimensions when %d" +
                " or more are required") % (len(X.shape), ndims))


def _log_sum_exp(a, axis=None, keepdims=False, xp_type="numpy"):
    """Calculate logsumexp like as scipy.special.logsumexp
    """
    xp = judge_xp_type(xp_type)
    a_max = a.max(axis=axis, keepdims=True)

    if a_max.ndim > 0:
        a_max[~xp.isfinite(a_max)] = 0
    elif not xp.isfinite(a_max):
        a_max = 0

    adj_exp = xp.exp(a - a_max)
    sum_exp = adj_exp.sum(axis=axis, keepdims=keepdims)
    out = xp.log(sum_exp)

    if not keepdims:
        a_max = xp.squeeze(a_max, axis=axis)
    out += a_max

    return out


# calculate transition covariance
def _calc_transition_covariance(G, Q):
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


# log prob gauss
def _log_prob_gauss(mean=None, cov=None, pre=None, xp_type="numpy"):
    xp = judge_xp_type(xp_type)

    if mean is not None:
        if cov is not None:
            def func(x):
                if x.ndim==1:
                    x = x.reshape(-1,1)
                -0.5 * (len(mean)*math.log(2*math.pi) + xp.linalg.slogdet(cov) 
                    + (mean - x).T @ xp.linalg.pinv(cov) @ (mean - x))
        elif pre is not None:
            def func(x):
                if x.ndim==1:
                    x = x.reshape(-1,1)
                -0.5 * (len(mean)*math.log(2*math.pi) - xp.linalg.slogdet(pre) 
                    + (mean - x).T @ pre @ (mean - x))
        else:
            def func(x, cov):
                if x.ndim==1:
                    x = x.reshape(-1,1)
                -0.5 * (len(mean)*math.log(2*math.pi) + xp.linalg.slogdet(cov) 
                    + (mean - x).T @ xp.linalg.pinv(cov) @ (mean - x))
    else:
        if cov is not None:
            def func(x, mean):
                if x.ndim==1:
                    x = x.reshape(-1,1)
                -0.5 * (len(mean)*math.log(2*math.pi) + xp.linalg.slogdet(cov) 
                    + (mean - x).T @ xp.linalg.pinv(cov) @ (mean - x))
        elif pre is not None:
            def func(x, mean):
                if x.ndim==1:
                    x = x.reshape(-1,1)
                -0.5 * (len(mean)*math.log(2*math.pi) - xp.linalg.slogdet(pre) 
                    + (mean - x).T @ pre @ (mean - x))
        else:
            raise ValueError("mean, covariance and precision are None elements.")
    return func



# calculate MSE
def mean_squared_error(x, y, xp_type="numpy"):
    assert x.shape == y.shape
    xp = judge_xp_type(xp_type)
    return xp.square(x - y).mean()
    # return xp.sqrt(xp.sum(xp.square(x - y))) / x.size


# calculate MAE
def mean_absolute_error(x, y, xp_type="numpy"):
    assert x.shape == y.shape
    xp = judge_xp_type(xp_type)
    return xp.mean(xp.absolute(x - y))


# intersection
def _intersect1d(ar1, ar2, assume_unique=False, return_indices=False, xp_type="numpy"):
    xp = judge_xp_type(xp_type)
    ar1 = xp.asanyarray(ar1)
    ar2 = xp.asanyarray(ar2)

    if not assume_unique:
        if return_indices:
            ar1, ind1 = xp.unique(ar1, return_index=True)
            ar2, ind2 = xp.unique(ar2, return_index=True)
        else:
            ar1 = xp.unique(ar1)
            ar2 = xp.unique(ar2)
    else:
        ar1 = ar1.ravel()
        ar2 = ar2.ravel()

    aux = xp.concatenate((ar1, ar2))
    if return_indices:
        aux_sort_indices = xp.argsort(aux)
        aux = aux[aux_sort_indices]
    else:
        aux.sort()

    mask = aux[1:] == aux[:-1]
    int1d = aux[:-1][mask]

    if return_indices:
        ar1_indices = aux_sort_indices[:-1][mask]
        ar2_indices = aux_sort_indices[1:][mask] - ar1.size
        if not assume_unique:
            ar1_indices = ind1[ar1_indices]
            ar2_indices = ind2[ar2_indices]

        return int1d, ar1_indices, ar2_indices
    else:
        return int1d
