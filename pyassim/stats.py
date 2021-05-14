# statistical functions

import math
import numpy as np


### Distribution
## Gaussian Distribution
def _gaussian_distribution(mu, Sigma, cov_type="full", xp=np):
    """
    Attributes:
        x [n_dim] {xp-array, float}
    """
    if cov_type=="full":
        def _calculate_for_x(x):
            return xp.exp(-0.5 * (x - mu).T @ xp.linalg.pinv(Sigma) @ (x - mu)) \
                    / (xp.power(2 * math.pi, len(mu) / 2) * xp.sqrt(xp.linalg.det(Sigma)))
    elif cov_type=="diag":
        def _calculate_for_x(x):
            return xp.exp(-((x - mu) * (x - mu) / (2 * Sigma)).sum()) \
                    / (xp.power(2 * math.pi, len(mu) / 2) * xp.sqrt(xp.prod(Sigma)))
    elif cov_type=="spherical":
        def _calculate_for_x(x):
            return xp.exp(-((x - mu) * (x - mu)).sum() / (2 * Sigma)) \
                    / xp.power(2 * math.pi * Sigma, len(mu) / 2)
    return _calculate_for_x


def _gaussian_distribution_multi(mu, Sigma, cov_type="full", xp=np):
    """
    Attributes:
        x [n_sample, n_dim] {xp-array, float}
    """
    if cov_type=="full":
        def _calculate_for_x(x):
            return xp.diag(xp.exp(-0.5 * (x - mu.reshape(1,-1)) @ xp.linalg.pinv(Sigma) @ (x - mu.reshape(1,-1)).T)
                    / (xp.power(2 * math.pi, len(mu) / 2) * xp.sqrt(xp.linalg.det(Sigma))))
    elif cov_type=="diag":
        def _calculate_for_x(x):
            return xp.exp(-((x - mu.reshape(1,-1)) * (x - mu.reshape(1,-1)) / (2 * Sigma.reshape(1,-1))).sum(axis=1)) \
                    / (xp.power(2 * math.pi, len(mu) / 2) * xp.sqrt(xp.prod(Sigma)))
    elif cov_type=="spherical":
        def _calculate_for_x(x):
            return xp.exp(-((x - mu.reshape(1,-1)) * (x - mu.reshape(1,-1)) / (2 * Sigma)).sum(axis=1)) \
                    / xp.power(2 * math.pi * Sigma, len(mu) / 2)
    return _calculate_for_x


def _log_gaussian_distribution(mu, Sigma, cov_type="full", xp=np):
    """
    Attributes:
        x [n_dim] {xp-array, float}
    """
    if cov_type=="full":
        def _calculate_for_x(x):
            return -0.5 * (x - mu).T @ xp.linalg.pinv(Sigma) @ (x - mu) \
                    - len(mu) * xp.log(2 * math.pi) / 2 - xp.log(xp.linalg.det(Sigma)) / 2
    elif cov_type=="diag":
        def _calculate_for_x(x):
            return -((x - mu) * (x - mu) / (2 * Sigma)).sum() \
                    - len(mu) * xp.log(2 * math.pi) / 2 - xp.log(Sigma).sum() / 2
    elif cov_type=="spherical":
        def _calculate_for_x(x):
            return -((x - mu) * (x - mu)).sum() / (2 * Sigma) \
                    - len(mu) * (xp.log(2 * math.pi) + xp.log(Sigma)) / 2
    return _calculate_for_x


def _log_gaussian_distribution_multi(mu, Sigma, cov_type="full", xp=np):
    """
    Attributes:
        x [n_sample, n_dim] {xp-array, float}
    """
    if cov_type=="full":
        def _calculate_for_x(x):
            return xp.diag(-0.5 * (x - mu.reshape(1,-1)) @ xp.linalg.pinv(Sigma) @ (x - mu.reshape(1,-1)).T
                    - 0.5 * len(mu) * xp.log(2 * math.pi) - 0.5 * xp.log(xp.linalg.det(Sigma)))
    elif cov_type=="diag":
        def _calculate_for_x(x):
            return -((x - mu.reshape(1,-1)) * (x - mu.reshape(1,-1)) / (2 * Sigma.reshape(1,-1))).sum(axis=1) \
                    - 0.5 * len(mu) * xp.log(2 * math.pi) - 0.5 * xp.log(Sigma).sum()
    elif cov_type=="spherical":
        def _calculate_for_x(x):
            return -((x - mu.reshape(1,-1)) * (x - mu.reshape(1,-1)) / (2 * Sigma)).sum(axis=1) \
                    - 0.5 * len(mu) * xp.log(2 * math.pi * Sigma)
    return _calculate_for_x


def _log_gaussian_distribution_kernel_Sigma_multi(Sigma, cov_type="full", xp=np):
    """
    Attributes:
        x [n_sample, n_dim] {xp-array, float}
    """
    if cov_type=="full":
        def _calculate_for_x(x, mu):
            return xp.diag(-0.5 * (x - mu.reshape(1,-1)) @ xp.linalg.pinv(Sigma) @ (x - mu.reshape(1,-1)).T)
    elif cov_type=="diag":
        def _calculate_for_x(x, mu):
            return -((x - mu.reshape(1,-1)) * (x - mu.reshape(1,-1)) / (2 * Sigma.reshape(1,-1))).sum(axis=1)
    elif cov_type=="spherical":
        def _calculate_for_x(x, mu):
            return -((x - mu.reshape(1,-1)) * (x - mu.reshape(1,-1)) / (2 * Sigma)).sum(axis=1)
    return _calculate_for_x


def _log_gaussian_distribution_kernel_all(x, mu, Sigma, cov_type="full", xp=np):
    if cov_type=="full":
        return xp.diag(-0.5 * (x - mu.reshape(1,-1)) @ xp.linalg.pinv(Sigma) @ (x - mu.reshape(1,-1)).T)
    elif cov_type=="diag":
        return -((x - mu.reshape(1,-1)) * (x - mu.reshape(1,-1)) / (2 * Sigma.reshape(1,-1))).sum(axis=1)
    elif cov_type=="spherical":
        return -((x - mu.reshape(1,-1)) * (x - mu.reshape(1,-1)) / (2 * Sigma)).sum(axis=1)
    else:
        raise ValueError("Type of covariance must be \"full\", \"diag\" or \"spherical\".")



## Poisson
def _log_poisson_kernel(x, lamda, xp=np):
    return x * xp.log(lamda) - xp.arange(x).sum()


def _log_poisson_kernel_lambda(lamda, xp=np):
    def _calculate_for_x(x):
        return x * xp.log(lamda) - xp.arange(x).sum()
    return _calculate_for_x


def _log_poisson_kernel_x(x, xp=np):
    def _calculate_for_lamda(lamda):
        return x * xp.log(lamda) - xp.arange(x).sum()
    return _calculate_for_lamda


## For Particle Filter
def _emperical_cummulative_inv(w_cumsum, idx, u, xp=np):
    """calculate inverse map for emperical cummulative function

    Args:
        w_cumsum [n_particle] {xp-array, float}
            : emperical cummulative function for particles
        idx [n_particle] {xp-array, int}
            : array of ID which are assigined to each particle
        u {float}
            : value between 0 and 1

    Returns (int):
        like floor function, ID number which maximize set of ID numbers,
        set are less than `u`
    """
    if xp.any(w_cumsum < u) == False:
        return 0
    k = xp.max(idx[w_cumsum < u])
    return k + 1


def _resampling(weights, xp=np):
        """caluclate standard resampling method
        
        Args:
            weights {xp-array, float} [n_particle]
                : set of likelihoodness for each particle

        Returns:
            k_list {xp-array, float} [n_particle]
                : index set which represent particle number remeining
        """
        w_cumsum = xp.cumsum(weights)

        # generate basic labels
        idx = xp.asanyarray(range(len(weights)))

        # storage for k
        k_list = xp.zeros(len(weights), dtype = int)
        
        # get index for resampling from weights with uniform distribution
        for i, u in enumerate(xp.random.uniform(0, 1, size = len(weights))):
            k = _emperical_cummulative_inv(w_cumsum, idx, u)
            k_list[i] = k
        return k_list


def _stratified_resampling(weights, xp=np):
    """caluclate stratified resampling method
    
    Args:
        weights {xp-array, float} [n_particle]
            : set of likelihoodness for each particle

    Returns:
        k_list {xp-array, float} [n_particle]
            : index set which represent particle number remeining
    """
    idx = xp.asanyarray(range(len(weights)))
    u0 = xp.random.uniform(0, 1 / len(weights))
    # u = [1 / len(weights)*i + u0 for i in range(len(weights))]
    u = u0 + xp.arange(len(weights)) / len(weights)
    w_cumsum = xp.cumsum(weights)
    k = xp.asanyarray([
        _emperical_cummulative_inv(w_cumsum, idx, val, xp) for val in u
        ])
    return k



