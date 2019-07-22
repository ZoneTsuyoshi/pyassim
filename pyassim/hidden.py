"""
===============================
Inference with Hidden Markov Model
===============================
This module implements the Hidden Markov Model
"""

import math

import numpy as np
import numpy.random as rd
from sklearn.cluster import KMeans

from .stats import _gaussian_distribution, _gaussian_distribution_multi, \
                _log_gaussian_distribution, _log_gaussian_distribution_multi


def _log_sum_exp(a, axis=None, keepdims=False, xp=np):
    """Calculate logsumexp like as scipy.special.logsumexp
    """
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


def _judge_minus_infinity(a, xp=np):
    return xp.all(xp.array([xp.isinf(a), a<0]), axis=0)


def _judge_minus_infinity2(a, b, xp=np):
    return xp.all(xp.array([xp.isinf(a), a<0, xp.isinf(b), b<0]), axis=0)


def _log_add_exp(a, b, xp=np):
    """Calculate logaddexp
    """
    # print("xi", a)
    # print("xi_t", b)
    ac = a.copy(); bc = b.copy()
    ac[_judge_minus_infinity(a, xp)] = b[_judge_minus_infinity(a, xp)]
    bc[_judge_minus_infinity(b, xp)] = a[_judge_minus_infinity(b, xp)]
    out = a.copy()
    out[_judge_minus_infinity2(a, b, xp)] = - xp.inf
    out[~_judge_minus_infinity2(a, b, xp)] = (xp.maximum(ac,bc)
        + xp.log(1 + xp.exp(-xp.absolute(ac - bc))))[~_judge_minus_infinity2(a, b, xp)]
    return out



class GaussianHiddenMarkovModel(object):
    """Implements the Hidden Markov Model.
    """
    def __init__(self,
                initial_probability = None, transition_kernel = None,
                n_components = 1, dtype = "float64", use_gpu = False):
        self.use_gpu = use_gpu
        if use_gpu:
            import cupy
            self.xp = cupy
        else:
            self.xp = np

        self.n_components = int(n_components)
        self.dtype = dtype
        self.fit_flag = False
        self.cov_type = "full"

        if initial_probability is None:
            self.pi = self.xp.ones(self.n_components, dtype=self.dtype) / self.n_components
        elif initial_probability.ndim==1 and len(initial_probability)==self.n_components:
            self.pi = self.xp.array(initial_probability).copy()
        else:
            raise ValueError("Lenght of \"initial_probability\" and n_components must be same. "
                            + "Also, number of axes of \"initial_probability\" must be 1.")
            

        if transition_kernel is None:
            self.P = self.xp.ones((self.n_components, self.n_components), dtype=self.dtype) / self.n_components
        elif transition_kernel.ndim==2 and transition_kernel.shape==(self.n_components, self.n_components):
            self.P = self.xp.array(transition_kernel).copy()
        else:
            raise ValueError("Shape of \"transition_kernel\" must be (n_components, n_components)." 
                            + "Also, number of axes of \"transition_kernel\" must be 2.")

    


    def fit(self, X, n_iter=10, em_vars=["pi", "P", "mu", "Sigma"], method="scaling",
            cov_type="full"):
        """estimate parameters by Baum-Welch algorithm
        Args:
            X [n_timesteps, n_dim] {float}
                : observation data
            n_iter {int}
                : number of iterations of EM algorithm
            em_vars {list, string}
                : variable list for update by EM
            method {string}
                : normal, scaling
            cov_type {string}
                : covariance type "full", "diag" or "spherical"
        """
        if cov_type in ["full", "diag", "spherical"]:
            self.cov_type = cov_type
        self.fit_flag = True
        if X.ndim == 2:
            n_data = 1
            n_timesteps, n_dim = X.shape
            X = X.reshape(1, X.shape[0], X.shape[1])
        elif X.ndim == 3:
            n_data, n_timesteps, n_dim = X.shape
        else:
            raise ValueError("X.ndim must be 2 or 3, but input is {}.")

        
        X = self.xp.array(X, dtype=self.dtype) # n_timesteps, n_dim
        alpha = self.xp.zeros((n_timesteps, self.n_components), dtype=self.dtype)
        beta = self.xp.zeros((n_timesteps, self.n_components), dtype=self.dtype)
        # xi: class numbering for n+1, n
        xi = self.xp.zeros((n_timesteps-1, self.n_components, self.n_components), dtype=self.dtype)
        kmeans = KMeans(n_clusters=self.n_components, random_state=0).fit(X.reshape(-1,n_dim))
        # self.mu = X.mean(axis=0) * self.xp.ones((self.n_components, n_dim), dtype=self.dtype)
        self.mu = self.xp.array(kmeans.cluster_centers_, dtype=self.dtype)
        if cov_type=="full":
            self.Sigma = self.xp.zeros((self.n_components, n_dim, n_dim), dtype=self.dtype)
            for k in range(self.n_components):
                self.Sigma[k] = self.xp.cov(X.reshape(-1,n_dim)[kmeans.labels_==k].T)
        elif cov_type=="diag":
            self.Sigma = self.xp.zeros((self.n_components, n_dim), dtype=self.dtype)
            for k in range(self.n_components):
                self.Sigma[k] = self.xp.var(X.reshape(-1,n_dim)[kmeans.labels_==k], axis=0)
        elif cov_type=="spherical":
            self.Sigma = self.xp.zeros(self.n_components, dtype=self.dtype)
            for k in range(self.n_components):
                self.Sigma[k] = self.xp.var(X.reshape(-1,n_dim)[kmeans.labels_==k], axis=0).mean()
        # kmeans = KMeans(n_clusters=self.n_components, random_state=0).fit(X)
        # # self.mu = X.mean(axis=0) * self.xp.ones((self.n_components, n_dim), dtype=self.dtype)
        # self.mu = self.xp.array(kmeans.cluster_centers_, dtype=self.dtype)
        # if cov_type=="full":
        #     self.Sigma = self.xp.zeros((self.n_components, n_dim, n_dim), dtype=self.dtype)
        #     for k in range(self.n_components):
        #         self.Sigma[k] = self.xp.cov(X[kmeans.labels_==k].T)
        # elif cov_type=="diag":
        #     self.Sigma = self.xp.zeros((self.n_components, n_dim), dtype=self.dtype)
        #     for k in range(self.n_components):
        #         self.Sigma[k] = self.xp.var(X[kmeans.labels_==k], axis=0)
        # elif cov_type=="spherical":
        #     self.Sigma = self.xp.zeros(self.n_components, dtype=self.dtype)
        #     for k in range(self.n_components):
        #         self.Sigma[k] = self.xp.var(X[kmeans.labels_==k], axis=0).mean()

        if method=="scaling":
            c = self.xp.zeros(n_timesteps, dtype=self.dtype)

        if method in ["normal", "scaling"]:
            for i in range(n_iter):
                print("\riteration={}/{}".format(i+1,n_iter))
                ## Expectation Step (Baum-Welch Algorithm)
                # calculation for alpha
                output_functions = []
                for k in range(self.n_components):
                    output_functions.append(_gaussian_distribution(self.mu[k], self.Sigma[k], cov_type, self.xp))
                    alpha[0,k] = self.pi[k] * output_functions[k](X[0])

                if method=="scaling":
                    c[0] = alpha[0].sum()
                    alpha[0] /= c[0]

                for t in range(n_timesteps-1):
                    for k in range(self.n_components):
                        alpha[t+1, k] = output_functions[k](X[t+1]) \
                                        * self.xp.sum(alpha[t] * self.P[k])
                    if method=="scaling":
                        c[t+1] = alpha[t+1].sum()
                        alpha[t+1] /= c[t+1]

                # calculation for beta
                beta = self.xp.zeros((n_timesteps, self.n_components), dtype=self.dtype)
                beta[-1] = 1
                if method=="scaling":
                    beta[-1] /= c[-1]
                    # beta[-1] *= np.prod(c)
                for t in reversed(range(n_timesteps-1)):
                    for k in range(self.n_components):
                        for kk in range(self.n_components):
                            beta[t,k] += beta[t+1,kk] * output_functions[kk](X[t+1]) * self.P[kk,k]
                    if method=="scaling":
                        beta[t] /= c[t+1]

                # calculation for xi
                if method=="normal":
                    pX = self.xp.sum(alpha[-1])
                # elif method=="scaling":
                #     pX = c[-1]
                    for t in range(n_timesteps-1):
                        for k in range(self.n_components):
                            xi[t,k] = alpha[t] * output_functions[k](X[t+1]) * self.P[k] * beta[t+1,k] / pX
                elif method=="scaling":
                    # print("c: ", c)
                    # print("alpha: ", alpha)
                    # print("beta: ", beta)
                    # pX = np.prod(c)
                    for t in range(n_timesteps-1):
                        for k in range(self.n_components):
                            xi[t,k] = alpha[t] * output_functions[k](X[t+1]) * self.P[k] * beta[t+1,k] / c[t+1]
                    print("xi", xi.sum(axis=0))


                ## Maximization Step
                gamma = alpha * beta
                # print("gamma: ", gamma)
                # print("xi: ", xi)
                if "pi" in em_vars:
                    self.pi = gamma[0] / np.sum(gamma[0])
                    # self.pi = alpha[0] * beta[0] / np.sum(alpha[0] * beta[0])
                if "P" in em_vars:
                    # self.P = (self.xp.sum(xi, axis=0).T / xi.sum(axis=0).sum(axis=0)).T # sum regarding n+1
                    self.P = (self.xp.sum(xi, axis=0).T / xi.sum(axis=0).sum(axis=0)).T
                    # self.P = self.xp.sum(xi / xi.mean(axis=0).sum(axis=0), axis=0) / xi.shape[0]
                    # xi2 = xi / xi.sum(axis=1)
                    # self.P = self.xp.sum(xi2, axis=0) / xi2.sum(axis=0).sum(axis=0)
                if "mu" in em_vars:
                    for k in range(self.n_components):
                        self.mu[k] = (gamma[:,k] / (gamma[:,k]).sum() * X.T).sum(axis=1) # overflow
                        # self.mu[k] = (gamma[:,k] / (gamma[:,k]).sum() * X.T) # overflow
                if "Sigma" in em_vars:
                    if cov_type=="full":
                        self.Sigma = self.xp.zeros((self.n_components, n_dim, n_dim), dtype=self.dtype)
                        for k in range(self.n_components):
                            for t in range(n_timesteps):
                                self.Sigma[k] += gamma[t,k] * self.xp.outer(X[t] - self.mu[k], X[t] - self.mu[k]) / gamma[:, k].sum()
                    elif cov_type=="diag":
                        for k in range(self.n_components):
                            self.Sigma[k] = (gamma[:,k] * ((X - self.mu[k]) * (X - self.mu[k])).T).sum(axis=1) / gamma[:, k].sum()
                    elif cov_type=="spherical":
                        for k in range(self.n_components):
                            self.Sigma[k] = (gamma[:,k] * ((X - self.mu[k]) * (X - self.mu[k])).T).sum() / (n_dim + gamma[:, k].sum())

                # print("pi: ", self.pi)
                # print("P: ", self.P)
                # print("mu: ", self.mu)
                # print("Sigma: ", self.Sigma)
        elif method=="log":
            for i in range(n_iter):
                print("\riteration={}/{}".format(i+1,n_iter))
                log_P = self.xp.log(self.P)
                log_output = self.xp.zeros((n_timesteps, self.n_components), dtype=self.dtype)
                gamma = self.xp.zeros((n_data, n_timesteps, self.n_components), dtype=self.dtype)
                # gamma_x = self.xp.zeros((n_dim, n_timesteps, self.n_components), dtype=self.dtype)
                xi = self.xp.zeros((self.n_components, self.n_components), dtype=self.dtype)

                for j, Xj in enumerate(X):
                    for k in range(self.n_components):
                        log_output_function = _log_gaussian_distribution_multi(self.mu[k], self.Sigma[k], cov_type, self.xp)
                        log_output[:,k] = log_output_function(Xj)
                    
                    alpha[0] = self.xp.log(self.pi) + log_output[0]
                    beta[-1] = 1

                    for t in range(n_timesteps-1):
                        for k in range(self.n_components):
                            alpha[t+1,k] = log_output[t+1,k] + _log_sum_exp(alpha[t] + log_P[k])

                    for t in reversed(range(n_timesteps-1)):
                        for k in range(self.n_components):
                            beta[t,k] = _log_sum_exp(beta[t+1] + log_output[t+1] + log_P[:,k])

                    log_gamma = alpha + beta
                    gamma[j] = self.xp.exp(log_gamma.T - _log_sum_exp(log_gamma, axis=1)).T 

                    if "P" in em_vars:
                        log_pX = _log_sum_exp(alpha[-1]) 
                        xi_t = self.xp.zeros((2, self.n_components, self.n_components), dtype=self.dtype)
                        xi_t[0,:,:] = - self.xp.inf

                        for t in range(n_timesteps-1):
                            for k in range(self.n_components):
                                xi_t[1,k] = alpha[t] + beta[t+1,k] + log_P[k] + log_output[t+1,k] - log_pX
                            xi_t[0] = _log_add_exp(xi_t[0], xi_t[1])

                        # xi = (xi.T - xi.max(axis=0)).T
                        xi += self.xp.exp(xi_t[0])

                if "pi" in em_vars:
                    self.pi = gamma[:,0].mean(axis=0)
                    # self.pi = alpha[0] * beta[0] / np.sum(alpha[0] * beta[0])
                if "P" in em_vars:
                    self.P = self.xp.where(self.P == 0.0, self.P, xi)
                    # P_tmp = (xi.T / xi.sum(axis=0)).T
                    self.P = (self.P.T / self.P.sum(axis=0)).T
                if "mu" in em_vars:
                    for k in range(self.n_components):
                        # self.mu[k] = (gamma[:,:,k] / (gamma[:,:,k]).sum() * X.T).sum(axis=1).sum(axis=0) # overflow
                        self.mu[k] = (gamma[:,:,k] * self.xp.rollaxis(X,2)).sum(axis=2).sum(axis=1) / gamma[:,:,k].sum()
                if "Sigma" in em_vars:    
                    if cov_type=="full":
                        self.Sigma = self.xp.zeros((self.n_components, n_dim, n_dim), dtype=self.dtype)
                        for k in range(self.n_components):
                            for n in range(n_data):
                                for t in range(n_timesteps):
                                    self.Sigma[k] += gamma[n,t,k] * self.xp.outer(X[n,t] - self.mu[k], X[n,t] - self.mu[k]) / gamma[:,:,k].sum()
                    elif cov_type=="diag":
                        for k in range(self.n_components):
                            self.Sigma[k] = (gamma[:,:,k] * self.xp.rollaxis((X - self.mu[k]) * (X - self.mu[k]), 2)).sum(axis=1).sum(axis=1) / gamma[:,:,k].sum()
                    elif cov_type=="spherical":
                        for k in range(self.n_components):
                            self.Sigma[k] = (gamma[:,:,k] * self.xp.rollaxis((X - self.mu[k]) * (X - self.mu[k]), 2)).sum() / (n_dim * gamma[:,:,k].sum())

                # if "pi" in em_vars:
                #     self.pi = gamma[0]
                # if "P" in em_vars:
                #     self.P = self.xp.where(self.P == 0.0, self.P, xi)
                #     # P_tmp = (xi.T / xi.sum(axis=0)).T
                #     self.P = (self.P.T / self.P.sum(axis=0)).T
                # if "mu" in em_vars:
                #     # for k in range(self.n_components):
                #     #     self.mu[k] = (gamma[:,k] / (gamma[:,k]).sum() * X.T).sum(axis=1)
                #     self.mu = gamma_x.sum(axis=1)
                # if "Sigma" in em_vars:
                #     if cov_type=="full":
                #         self.Sigma = self.xp.zeros((self.n_components, n_dim, n_dim), dtype=self.dtype)
                #         for k in range(self.n_components):
                #             for t in range(n_timesteps):
                #                 self.Sigma[k] += gamma[t,k] * self.xp.outer(X[t] - self.mu[k], X[t] - self.mu[k]) / gamma[:, k].sum()
                #     elif cov_type=="diag":
                #         for k in range(self.n_components):
                #             self.Sigma[k] = (gamma[:,k] * ((X - self.mu[k]) * (X - self.mu[k])).T).sum(axis=1) / gamma[:, k].sum()
                #     elif cov_type=="spherical":
                #         for k in range(self.n_components):
                #             self.Sigma[k] = (gamma[:,k] * ((X - self.mu[k]) * (X - self.mu[k])).T).sum() / (n_dim + gamma[:, k].sum())



    def predict(self, X):
        """predict path by Viterbi algorithm
        Args:
            X [n_timesteps, n_dim] {float}
                : observation data
        """
        if not self.fit_flag:
            # ToDo: make error function
            raise TypeError("Function \"fit\" has not executed, yet.")
        if X.ndim != 2:
            raise ValueError("X.ndim must be 2, but input is {}.")

        n_timesteps, n_dim = X.shape
        X = self.xp.array(X, dtype=self.dtype) # n_timesteps, n_dim
        path = self.xp.zeros((n_timesteps, self.n_components), dtype=int)
        omega_old = self.xp.zeros(self.n_components, dtype=self.dtype)
        omega = self.xp.zeros(self.n_components, dtype=self.dtype)
        new_indices = self.xp.zeros(self.n_components, dtype=int)

        output_functions = []
        for k in range(self.n_components):
            output_functions.append(_log_gaussian_distribution(self.mu[k], self.Sigma[k], self.cov_type, self.xp))
            omega_old[k] = self.xp.log(self.pi[k]) + output_functions[k](X[0])
        path[0] = range(self.n_components)

        for t in range(n_timesteps-1):
            for k in range(self.n_components):
                omega[k] = output_functions[k](X[t+1]) + self.xp.max(self.xp.log(self.P[k]) + omega_old)
                new_indices[k] = self.xp.argmax(self.xp.log(self.P[k]) + omega_old)

            for k in range(self.n_components):
                path[:t+1, k] = path[:t+1, new_indices[k]]
                path[t+1, k] = k

            omega_old = omega.copy()

        return path[:,self.xp.argmax(omega)]



    def fit_predict(self, X, n_iter=10, em_vars=["pi", "P", "mu", "Sigma"], method="scaling",
            cov_type="full"):
        """
        Args:
            X [n_timesteps, n_dim] {float}
                : observation data
            n_iter {int}
                : number of iterations of EM algorithm
            em_vars {list, string}
                : variable list for update by EM
            method {string}
                : normal, scaling
        """
        self.fit(X, n_iter, em_vars, method, cov_type)
        return self.predict(X)



    def fit_multi(self, X, n_iter=10, em_vars=["pi", "P", "mu", "Sigma"], method="scaling",
            cov_type="full"):
        """estimate parameters by Baum-Welch algorithm
        Args:
            X [n_timesteps, n_dim] {float}
                : observation data
            n_iter {int}
                : number of iterations of EM algorithm
            em_vars {list, string}
                : variable list for update by EM
            method {string}
                : normal, scaling
        """
        self.fit_flag = True
        if cov_type in ["full", "diag", "spherical"]:
            self.cov_type = cov_type
        if X.ndim != 3:
            raise ValueError("X.ndim must be 2, but input is {}.")

        n_data, n_timesteps, n_dim = X.shape
        X = self.xp.array(X, dtype=self.dtype) # n_data, n_timesteps, n_dim
        alpha = self.xp.zeros((n_data, n_timesteps, self.n_components), dtype=self.dtype)
        beta = self.xp.zeros((n_data, n_timesteps, self.n_components), dtype=self.dtype)
        # xi: class numbering for n+1, n
        xi = self.xp.zeros((n_data, n_timesteps-1, self.n_components, self.n_components), dtype=self.dtype)
        kmeans = KMeans(n_clusters=self.n_components, random_state=0).fit(X.reshape(-1,n_dim))
        # self.mu = X.mean(axis=0) * self.xp.ones((self.n_components, n_dim), dtype=self.dtype)
        self.mu = self.xp.array(kmeans.cluster_centers_, dtype=self.dtype)
        if cov_type=="full":
            self.Sigma = self.xp.zeros((self.n_components, n_dim, n_dim), dtype=self.dtype)
            for k in range(self.n_components):
                self.Sigma[k] = self.xp.cov(X.reshape(-1,n_dim)[kmeans.labels_==k].T)
        elif cov_type=="diag":
            self.Sigma = self.xp.zeros((self.n_components, n_dim), dtype=self.dtype)
            for k in range(self.n_components):
                self.Sigma[k] = self.xp.var(X.reshape(-1,n_dim)[kmeans.labels_==k], axis=0)
        elif cov_type=="spherical":
            self.Sigma = self.xp.zeros(self.n_components, dtype=self.dtype)
            for k in range(self.n_components):
                self.Sigma[k] = self.xp.var(X.reshape(-1,n_dim)[kmeans.labels_==k], axis=0).mean()


        if method=="scaling":
            c = self.xp.zeros((n_data, n_timesteps), dtype=self.dtype)

        for i in range(n_iter):
            print("\riteration={}/{}".format(i+1,n_iter))
            ## Expectation Step (Baum-Welch Algorithm)
            # calculation for alpha
            output_functions = []
            for k in range(self.n_components):
                output_functions.append(_gaussian_distribution_multi(self.mu[k], self.Sigma[k], cov_type, self.xp))
                alpha[:,0,k] = self.pi[k] * output_functions[k](X[:,0])

            if method=="scaling":
                c[:,0] = alpha[:,0].sum(axis=1)
                alpha[:,0] /= c[:,0].reshape(n_data,1)

            for t in range(n_timesteps-1):
                for k in range(self.n_components):
                    alpha[:, t+1, k] = output_functions[k](X[:, t+1]) \
                                    * self.xp.sum(alpha[:, t] * self.P[k], axis=1)
                if method=="scaling":
                    c[:, t+1] = alpha[:, t+1].sum(axis=1)
                    alpha[:, t+1] /= c[:, t+1].reshape(n_data,1)


            # calculation for beta
            beta = self.xp.zeros((n_data, n_timesteps, self.n_components), dtype=self.dtype)
            beta[:, -1] = 1
            if method=="scaling":
                beta[:, -1] /= c[:, -1].reshape(n_data,1)
                # beta[-1] *= np.prod(c)
            for t in reversed(range(n_timesteps-1)):
                for k in range(self.n_components):
                    for kk in range(self.n_components):
                        beta[:,t,k] += beta[:,t+1,kk] * output_functions[kk](X[:,t+1]) * self.P[kk,k]
                if method=="scaling":
                    beta[:,t] /= c[:,t+1].reshape(n_data,1)

            # calculation for xi
            if method=="normal":
                pX = self.xp.sum(alpha[:,-1])
            # elif method=="scaling":
            #     pX = c[-1]
                for t in range(n_timesteps-1):
                    for k in range(self.n_components):
                        xi[:,t,k] = alpha[:,t] * output_functions[k](X[:,t+1]) * self.P[k] * beta[:,t+1,k] / pX.reshape(n_data,1)
            elif method=="scaling":
                # print("c: ", c)
                # print("alpha: ", alpha)
                # print("beta: ", beta)
                # pX = np.prod(c)
                for t in range(n_timesteps-1):
                    for k in range(self.n_components):
                        xi[:,t,k] = alpha[:,t] * output_functions[k](X[:,t+1]).reshape(n_data,1) * self.P[k] * beta[:,t+1,k].reshape(n_data,1) / c[:,t+1].reshape(n_data,1)


            ## Maximization Step
            gamma = alpha * beta
            # print("gamma: ", gamma)
            # print("xi: ", xi)
            if "pi" in em_vars:
                self.pi = gamma[:,0].sum(axis=0) / gamma[:,0].sum()
                # self.pi = alpha[0] * beta[0] / np.sum(alpha[0] * beta[0])
            if "P" in em_vars:
                # self.P = (self.xp.sum(xi, axis=0).T / xi.sum(axis=0).sum(axis=0)).T # sum regarding n+1
                self.P = (self.xp.sum(xi, axis=1).sum(axis=0).T / xi.sum(axis=0).sum(axis=0).sum(axis=0)).T
                # self.P = self.xp.sum(xi / xi.mean(axis=0).sum(axis=0), axis=0) / xi.shape[0]
                # xi2 = xi / xi.sum(axis=1)
                # self.P = self.xp.sum(xi2, axis=0) / xi2.sum(axis=0).sum(axis=0)
            if "mu" in em_vars:
                for k in range(self.n_components):
                    # self.mu[k] = (gamma[:,:,k] / (gamma[:,:,k]).sum() * X.T).sum(axis=1).sum(axis=0) # overflow
                    self.mu[k] = (gamma[:,:,k] * self.xp.rollaxis(X,2)).sum(axis=2).sum(axis=1) / gamma[:,:,k].sum()
            if "Sigma" in em_vars:    
                if cov_type=="full":
                    self.Sigma = self.xp.zeros((self.n_components, n_dim, n_dim), dtype=self.dtype)
                    for k in range(self.n_components):
                        for n in range(n_data):
                            for t in range(n_timesteps):
                                self.Sigma[k] += gamma[n,t,k] * self.xp.outer(X[n,t] - self.mu[k], X[n,t] - self.mu[k]) / gamma[:,:,k].sum()
                elif cov_type=="diag":
                    for k in range(self.n_components):
                        self.Sigma[k] = (gamma[:,:,k] * self.xp.rollaxis((X - self.mu[k]) * (X - self.mu[k]), 2)).sum(axis=1).sum(axis=1) / gamma[:,:,k].sum()
                elif cov_type=="spherical":
                    for k in range(self.n_components):
                        self.Sigma[k] = (gamma[:,:,k] * self.xp.rollaxis((X - self.mu[k]) * (X - self.mu[k]), 2)).sum() / (n_dim * gamma[:,:,k].sum())



