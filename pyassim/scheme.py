from abc import ABCMeta, abstractmethod
import numpy as np


class Scheme(metaclass=ABCMeta):
    @abstractmethod
    def _forward(self, system):
        pass



class EulerScheme(Scheme):
    def __init__(self, dt, timestep, system, seed=121):
        self.dt = dt
        self.timestep = timestep
        self.system = system
        np.random.seed(seed)


    def _forward(self, x, t):
        return x + self.system.f(x, t) * self.dt


    def perfect_simulation(self, initial_x):
        whole_x = np.zeros((self.timestep, len(initial_x)))
        current_x = initial_x.copy()
        whole_x[0] = current_x

        for s in range(self.timestep-1):
            current_x = self._forward(current_x, s*self.dt)
            whole_x[s+1] = current_x

        return whole_x


    def noise_added_simulation(self, initial_x, sd):
        true_x = np.zeros((self.timestep, len(initial_x)))
        obs_x = np.zeros((self.timestep, len(initial_x)))
        current_x = initial_x.copy()
        true_x[0] = current_x
        obs_x[0] = current_x

        for s in range(self.timestep-1):
            current_x = self._forward(current_x, s*self.dt)
            true_x[s+1] = current_x
            obs_x[s+1] = current_x + np.random.normal(0, sd, size=len(initial_x))

        return true_x, obs_x


    def noise_variant_simulation(self, initial_x, sd_rate):
        true_x = np.zeros((self.timestep, len(initial_x)))
        obs_x = np.zeros((self.timestep, len(initial_x)))
        current_x = initial_x.copy()
        true_x[0] = current_x
        obs_x[0] = current_x

        for s in range(self.timestep-1):
            current_x = self._forward(current_x, s*self.dt)
            true_x[s+1] = current_x
            obs_x[s+1] = current_x + np.random.normal(0,
                                                sd_rate*np.sqrt(np.absolute(true_x[s+1] - true_x[s])),
                                                size=len(initial_x))

        return true_x, obs_x


    def get_transition_function(self, t=0):
        def transition_function(x, noise):
            return x + self.system.f(x, t) * self.dt + noise
        return transition_function