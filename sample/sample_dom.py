"""
sample code for LOCK and EM algorithm of KF
application these methods to damped oscillation model
"""

import os, sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append("..")
from pyassim import LOCK, KalmanFilter, DampedOscillationModel, EulerScheme


def main():
    result_dir = "figures/dom"
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    seed = 121
    np.random.seed(seed)

    # parameters
    perf_initial = np.array([5.0, 0.0])
    sim_initial = np.array([[6.0, 0.0]])
    m = 1.0
    k = 0.5
    r = 0.52
    dt = 1
    sys_sd = 0.01
    obs_sd = 0.2
    T = 100

    def w (t) :
        # return np.sin(np.pi * t / 5)
        return 0

    wd = np.zeros(T + 2)
    for t in range(T + 2) :
        wd[t] = w(t)

    # setup matrix
    F = np.array([[1, dt], [- k * dt / m, 1 - r * dt / m]]) + 0.1*np.random.randn(2,2)
    G = np.array([[0], [dt / m]])
    H = np.array([[1, 0], [0, 1]])
    Gw = np.dot(np.array([G[:, 0]]).T, [wd]).T
    V0 = np.array([[1, 0], [0, 1]]) # 1,0,0,0
    Q = np.array([[sys_sd**2]])
    R = np.array([[obs_sd**2, 0], [0, obs_sd**2]]) 

    dom = DampedOscillationModel(m,k,r,w)
    es = EulerScheme(dt,T,dom,seed=seed)
    true, obs = es.noise_added_simulation(perf_initial, sys_sd, obs_sd)

    ## KF
    kf = KalmanFilter(obs, sim_initial, V0, F, H, Q, R, G, em_vars=["transition_matrices"])
    kf.em(n_iter=10)
    kf.forward()
    print(kf.F)

    ## LOCK
    lock = LOCK(obs, sim_initial, V0, F, H, G @ Q @ G.T, R, 
        estimation_length=4, estimation_interval=4, eta=0.6, cutoff=0.5)
    lock.forward()


    fig, ax = plt.subplots(1,2,figsize=(15,5))
    for i, value in enumerate(["x", "v"]):
        ax[i].scatter(range(T), obs[:,i], color = "k", marker = "o", label = "obs")
        ax[i].plot(true[:,i], linestyle = "--", color = "c", label = "true")
        ax[i].plot(kf.get_filtered_value(i), color = "g", label =  "KF (EM)")
        ax[i].plot(lock.get_filtered_value(i), color = "b", label =  "LOCK")
        ax[i].set_xlabel("Timestep", fontsize=12)
        ax[i].set_ylabel(value, fontsize=12)
    ax[0].legend(loc = "upper right", bbox_to_anchor=(1.4, -0.15), ncol=4, fontsize=12)
    fig.savefig(os.path.join(result_dir, "lock_estimated.pdf"), bbox_inches="tight")



if __name__ == "__main__":
    main()