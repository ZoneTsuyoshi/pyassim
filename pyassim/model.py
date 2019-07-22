from abc import ABCMeta, abstractmethod
import numpy as np
import math

class ODE(metaclass=ABCMeta):
    @abstractmethod
    def f(self, x, t):
        pass



class DampedOccilationModel(ODE):
    def __init__(self, m, k, r, w):
        self.m = m
        self.k = k
        self.r = r
        self.w = w


    def f(self, x, t):
        perturbation = np.zeros_like(x)
        perturbation[0] = x[1]
        perturbation[1] = (- self.k * x[0] - self.r * x[1] + self.w(t)) / self.m
        return perturbation


class CoefficientChangedDampedOccilationModel(ODE):
    def __init__(self, m, k, r, w):
        self.m = m
        self.k = k
        self.r = r
        self.w = w


    def f(self, x, t):
        perturbation = np.zeros_like(x)
        perturbation[0] = x[1]
        perturbation[1] = (- self.k(t) * x[0] - self.r(t) * x[1] + self.w(t)) / self.m
        return perturbation


class DuffingModel(ODE):
    def __init__(self, m, alpha, beta):
        self.m = m
        self.alpha = alpha
        self.beta = beta


    def f(self, x, t):
        perturbation = np.zeros_like(x)
        perturbation[0] = x[1]
        perturbation[1] = (- self.alpha * x[0] - self.beta * x[0]**3) / self.m
        return perturbation


class Lorentz63Model(ODE):
    def __init__(self, sigma, rho, beta):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def f(self, x, t):
        perturbation = np.zeros_like(x)
        perturbation[0] = self.sigma * (x[1] - x[0])
        perturbation[1] = x[0] * (self.rho - x[2]) - x[1]
        perturbation[2] = x[0] * x[1] - self.beta * x[2]
        return perturbation


def rotation_matrix_3d(theta):
    result = np.eye(3)
    R = np.array([[1, 0, 0],
                  [0, math.cos(theta[0]), -math.sin(theta[0])],
                  [0, math.sin(theta[0]), math.cos(theta[0])]])
    result = R @ result
    R = np.array([[math.cos(theta[1]), 0, -math.sin(theta[1])],
                  [0, 1, 0],
                  [math.sin(theta[1]), 0, math.cos(theta[1])]])
    result = R @ result
    R = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                  [math.sin(theta[2]), math.cos(theta[2]), 0],
                  [0, 0, 1]])
    return R @ result


def Rodrigues_rotation_matrix(n, theta):
    if np.linalg.norm(n)!=0:
        n = n / np.linalg.norm(n) 
    else:
        raise ValueError("norm of n must be greater than 0.")

    return np.array([[math.cos(theta)+n[0]*n[0]*(1-math.cos(theta)),
                        n[0]*n[1]*(1-math.cos(theta))-n[2]*math.sin(theta),
                        n[0]*n[2]*(1-math.cos(theta))+n[1]*math.sin(theta)],
                     [n[0]*n[1]*(1-math.cos(theta))+n[2]*math.sin(theta),
                        math.cos(theta)+n[1]*n[1]*(1-math.cos(theta)),
                        n[1]*n[2]*(1-math.cos(theta))-n[0]*math.sin(theta)],
                     [n[0]*n[2]*(1-math.cos(theta))-n[1]*math.sin(theta),
                        n[1]*n[2]*(1-math.cos(theta))+n[0]*math.sin(theta),
                        math.cos(theta)+n[2]*n[2]*(1-math.cos(theta))]])
