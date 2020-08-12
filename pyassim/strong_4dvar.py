"""
=======================================================
Inference with strong-constrained 4D Variational Method
=======================================================
This module implements the strong-constrained 4D Variational method,
for Nonlinear state space models
"""

# import numpy as np
import autograd.numpy as np
from autograd import grad
from .utils import array1d, array2d
from .util_functions import _last_dims, _determine_dimensionality


class Strong4DVar(object):
    def __init__(self, observation = None, 
                initial_mean = None, 
                transition_functions = None,
                observation_functions = None,
                observation_covariance = None,
                n_dim_sys = None, n_dim_obs = None,
                use_gpu = False,
                dtype = "float64", seed = 10):
        self.use_gpu = use_gpu
        if use_gpu:
            import cupy
            self.xp = cupy
        else:
            self.xp = np

        # check order of tensor and mask missing values
        self.y = self.xp.array(observation, dtype=dtype)
        self.n_timestep = len(self.y)

        # determine dimensionality
        self.n_dim_sys = _determine_dimensionality(
            [(initial_mean, array1d, -1)],
            n_dim_sys
        )

        self.n_dim_obs = _determine_dimensionality(
            [(observation, array1d, -1),
            (observation_covariance, array2d, -2)],
            n_dim_obs
        )

        if observation_covariance is None:
            self.R = self.xp.eye(self.n_dim_obs, dtype = dtype)
        else:
            self.R = observation_covariance.astype(dtype)

        self.x_est = self.xp.zeros((self.n_timestep, self.n_dim_sys))
        self.x_est[0] = initial_mean
        self.y_est = self.xp.zeros((self.n_timestep, self.n_dim_obs))

        self.grad_fs = []
        for f in transition_functions:
            self.grad_fs.append(grad(f))

        self.grad_hs = []
        for h in observation_functions:
            self.grad_hs.append(grad(h))

        self.transition_functions = transition_functions
        self.observation_functions = observation_functions




    def estimate(self, iteration=10, eta=0.01, method="gd"):
        costs = np.zeros(iteration + 1)

        for j in range(iteration):
            ## forward
            for t in range(self.n_timestep-1):
                for i, f in enumerate(self.transition_functions):
                    self.x_est[t+1, i] = f(self.x_est[t], t)

            ## backward
            adjoint = self.xp.zeros((self.n_timestep + 1, self.n_dim_sys))
            adjoint[-1] = 0
            F = self.xp.zeros((self.n_dim_sys, self.n_dim_sys))
            H = self.xp.zeros((self.n_dim_obs, self.n_dim_sys))
            for t in reversed(range(self.n_timestep)):
                ## calculate Jacobian
                for i, grad_f in enumerate(self.grad_fs):
                    F[i] = grad_f(self.x_est[t], t)

                for i, grad_h in enumerate(self.grad_hs):
                    H[i] = grad_h(self.x_est[t], t)

                R = _last_dims(self.R, t, 2)
                for i, h in enumerate(self.observation_functions):
                    self.y_est[t, i] = h(self.x_est[t], t)
                adjoint[t] = F.T @ adjoint[t+1] + H.T @ self.xp.linalg.pinv(R) @ (self.y_est[t] - self.y[t])

            costs[j] = self.cost_function()

            if method=="gd":
                g = - adjoint[0]
                self.x_est[0] += eta * g
            elif method=="cg":
                # due to no quadrotic, applying CG method is difficult
                if j==0:
                    search_vec = - adjoint[0]
                else:
                    search_vec = - adjoint[0] + adjoint[0] @ adjoint[0] / g @ g * search_vec
                search_rad = eta
                g = adjoint[0]


        ## forward
        for t in range(self.n_timestep-1):
            for i, f in enumerate(self.transition_functions):
                self.x_est[t+1, i] = f(self.x_est[t], t)

        for t in range(self.n_timestep):
            for i, h in enumerate(self.observation_functions):
                    self.y_est[t, i] = h(self.x_est[t], t)

        costs[-1] = self.cost_function()
        return costs



    def cost_function(self):
        result = 0
        for t in range(self.n_timestep):
            R = _last_dims(self.R, t, 2)
            result += self.y_est[t].reshape(1,self.n_dim_obs) @ R @ self.y_est[t]
        return result


