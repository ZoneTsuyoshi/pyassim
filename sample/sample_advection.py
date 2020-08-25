"""
sample code for LLOCK, SLOCK, LSLOCK
application the method to advection model (periodic boundary condition)
"""

import os, sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("..")
from pyassim import KalmanFilter, LocalLOCK, SpatiallyUniformLOCK, LSLOCK,\
                    PeriodicAdvection, EulerScheme


def main():
    result_dir = "figures/advection"
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    seed = 121
    np.random.seed(seed)

    # parameters
    N = 20
    x0 = np.exp(-(np.arange(N)-N//2)**2/20)
    dt = 0.01
    dx = 1
    c = 1
    sys_sd = 0.001
    obs_sd = 0.1
    timestep = 10000
    ds = 100

    # generate data
    model = PeriodicAdvection(dx, c, dt, scheme="LW")
    scheme = EulerScheme(dt, timestep, model, seed=seed)
    true, obs = scheme.noise_added_simulation(x0, sys_sd, obs_sd)

    # setup matrices
    # adjacency matrix
    A = np.eye(N)
    A[np.arange(N-1), np.arange(1,N)] = 2
    A[np.arange(1,N), np.arange(N-1)] = 3
    A[0,-1] = 3
    A[-1,0] = 2
    # A[np.arange(N-2), np.arange(2,N)] = True
    # A[np.arange(2,N), np.arange(N-2)] = True
    # A[0,-2] = A[-2,0] = A[1,-1] = A[-1,1] = True

    # initial transition matrix
    F = np.eye(N)
    H = np.eye(N)

    # covariance
    Q = obs_sd**2 * np.eye(N)
    R = obs_sd**2 * np.eye(N)
    V0 = obs_sd**2 * np.eye(N)

    # execution
    kf = KalmanFilter(obs[::ds], x0, V0, F, H, Q, R, em_vars=["transition_matrices"])
    kf.em(n_iter=10)
    kf.forward()

    llock = LocalLOCK(obs[::ds], x0, V0, F, H, Q, R, A.astype(bool), method="elementwise",
                 estimation_length=20, estimation_interval=5, eta=1.0,
                 cutoff=10, estimation_mode="forward")
    llock.forward()

    slock = SpatiallyUniformLOCK(obs[::ds], x0, V0, F, H, Q, R, np.zeros(N), A, 
                 estimation_length=1, estimation_interval=1, eta=1.,
                 cutoff=10., estimation_mode="forward")
    slock.forward()

    lslock = LSLOCK(obs[::ds], x0, V0, F, H, Q, R, A, method="gridwise", 
                 estimation_length=10, estimation_interval=5, eta=1.,
                 cutoff=10., estimation_mode="forward")
    lslock.forward()

    # draw results
    dim=0
    plt.figure(figsize=(8,5))
    plt.scatter(np.arange(timestep//ds), obs[::ds,dim], label="obs", c="k")
    plt.plot(true[::ds,dim], label="true", c="cyan", ls="--")
    plt.plot(kf.get_filtered_value(dim), label="kf w/ EM")
    plt.plot(llock.get_filtered_value(dim), label="llock")
    plt.plot(slock.get_filtered_value(dim), label="slock")
    plt.plot(lslock.get_filtered_value(dim), label="lslock")
    plt.legend()
    plt.savefig(os.path.join(result_dir, "dim{}_estimated.pdf".format(dim)), bbox_inches="tight")

    fig, ax = plt.subplots(2,2,figsize=(10,10))
    vmin, vmax = obs.min(), obs.max()
    sns.heatmap(true[::ds], cmap="Blues", vmin=vmin, vmax=vmax, ax=ax[0,0])
    sns.heatmap(llock.get_filtered_value(), cmap="Blues", vmin=vmin, vmax=vmax, ax=ax[0,1])
    sns.heatmap(slock.get_filtered_value(), cmap="Blues", vmin=vmin, vmax=vmax, ax=ax[1,0])
    sns.heatmap(lslock.get_filtered_value(), cmap="Blues", vmin=vmin, vmax=vmax, ax=ax[1,1])

    ax[0,0].set_title("True")
    ax[0,1].set_title("LLOCK")
    ax[1,0].set_title("SLOCK")
    ax[1,1].set_title("LSLOCK")

    for i in range(2):
        for j in range(2):
            ax[i,j].set_xlabel("space")
            ax[i,j].set_ylabel("timestep")
    fig.savefig(os.path.join(result_dir, "estimated.pdf"))


if __name__ == "__main__":
    main()