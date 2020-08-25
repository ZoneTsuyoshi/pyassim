"""
sample code for LOCK and EM algorithm of KF
application these methods to damped oscillation model
"""

import os, sys, math

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append("..")
from pyassim import EnsembleKalmanFilter,  NonlinearEnsembleKalmanFilter, \
                    ParticleFilterGaussian, GaussianParticleFilterGauss,\
                    Lorenz63Model, RungeKuttaScheme, Rodrigues_rotation_matrix


def main():
    result_dir = "figures/Lorenz"
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    seed = 121
    np.random.seed(seed)

    # parameters
    sigma = 10
    rho = 28
    beta = 8/3
    x0 = np.array([5., 5., 5.])
    dt = 0.01
    sys_sd = 0.1
    obs_sd = 0.5
    timestep = 1500
    ds = 10

    model = Lorenz63Model(sigma, rho, beta)
    scheme = RungeKuttaScheme(dt, timestep, model, seed=seed)
    true, obs = scheme.noise_added_simulation(x0, sys_sd, obs_sd)

    scheme2 = RungeKuttaScheme(dt, ds+1, model, seed=seed)
    def transition_func(x, v):
        return scheme2.perfect_simulation(x)[-1] + v

    def observation_func(x):
        Rot = Rodrigues_rotation_matrix(np.ones(3), math.pi/4)
        return (Rot @ x.T).T

    H = Rodrigues_rotation_matrix(np.ones(3), math.pi/4)
    Q = (ds*sys_sd)**2*np.eye(3) #1e-2
    R = obs_sd**2*np.eye(3) #1e-2
    trans_obs = (H @ obs.T).T

    ## EnKF
    enkf = EnsembleKalmanFilter(trans_obs[::ds], [transition_func], H, x0,
                           transition_noise=(np.random.multivariate_normal, [np.zeros(3), Q]),
                           observation_covariance=R, n_particles=8)
    enkf.forward()
    enkf.smooth(5)

    ## NEnKF
    nenkf = NonlinearEnsembleKalmanFilter(trans_obs[::ds], [transition_func],
                                      [observation_func], x0,
                           transition_noise=(np.random.multivariate_normal, [np.zeros(3), Q]),
                           observation_covariance=R, n_particles=8)
    nenkf.forward()

    ## PF/PS
    pf = ParticleFilterGaussian(trans_obs[::ds], [transition_func], [observation_func], x0,
                           transition_noise=(np.random.multivariate_normal, [np.zeros(3), Q]),
                            save_particles=True,
                           observation_covariance=R, n_particles=16)
    # pf.forward()
    pf.smooth(5)

    ## GPF
    gpf = GaussianParticleFilterGauss(trans_obs[::ds], [transition_func], 
                                  [observation_func], x0,
                           transition_noise=(np.random.multivariate_normal, [np.zeros(3), Q]),
                            save_particles=True,
                           observation_covariance=R, n_particles=16)
    gpf.forward()



    fig, ax = plt.subplots(3,1,figsize=(12,8))
    for i in range(3):
        ax[i].scatter(np.arange(timestep//ds), obs[::ds,i], label="obs", c="k",
                      linewidth=1, marker="x")
        ax[i].plot(true[::ds,i], ls="--", label="true")
        ax[i].plot(enkf.get_filtered_value(i, False), label="enkf")
        ax[i].plot(enkf.get_smoothed_value(i, False), label="enks")
        ax[i].plot(nenkf.get_filtered_value(i, False), label="nenkf")
    ax[-1].set_xlabel("Timestep")
    ax[0].legend(loc="upper right", bbox_to_anchor=(1.1,1.))
    fig.savefig(os.path.join(result_dir, "enkf_estimated.pdf"), bbox_inches="tight")

    fig, ax = plt.subplots(3,1,figsize=(12,8))
    for i in range(3):
        ax[i].scatter(np.arange(timestep//ds), obs[::ds,i], label="obs", c="k",
                      linewidth=1, marker="x")
        ax[i].plot(true[::ds,i], ls="--", label="true")
        ax[i].plot(pf.get_filtered_value(i, False), label="pf")
        ax[i].plot(pf.get_smoothed_value(i, False), label="ps")
        ax[i].plot(gpf.get_filtered_value(i, False), label="gpf")
    ax[-1].set_xlabel("Timestep")
    ax[0].legend(loc="upper right", bbox_to_anchor=(1.1,1.))
    fig.savefig(os.path.join(result_dir, "pf_estimated.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()