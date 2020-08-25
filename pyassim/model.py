import numpy as np
import math

class ODE(object):
    def __init__(self, xp_type="numpy"):
        if xp_type=="numpy":
            self.xp = np
        elif xp_type=="cupy":
            import cupy
            self.xp = cupy

    def f(self, t, x):
        pass


class DampedOscillationModel(ODE):
    def __init__(self, m, k, r, w, xp_type="numpy"):
        super(DampedOscillationModel, self).__init__(xp_type)
        self.m = m
        self.k = k
        self.r = r
        self.w = w


    def f(self, t, x):
        perturbation = self.xp.zeros_like(x)
        perturbation[0] = x[1]
        perturbation[1] = (- self.k * x[0] - self.r * x[1] + self.w(t)) / self.m
        return perturbation


class CoefficientChangedDampedOscillationModel(ODE):
    def __init__(self, m, k, r, w, xp_type="numpy"):
        super(CoefficientChangedDampedOscillationModel, self).__init__(xp_type)
        self.m = m
        self.k = k
        self.r = r
        self.w = w


    def f(self, t, x):
        perturbation = self.xp.zeros_like(x)
        perturbation[0] = x[1]
        perturbation[1] = (- self.k(t) * x[0] - self.r(t) * x[1] + self.w(t)) / self.m
        return perturbation


class DuffingModel(ODE):
    def __init__(self, m, alpha, beta, xp_type="numpy"):
        super(DuffingModel, self).__init__(xp_type)
        self.m = m
        self.alpha = alpha
        self.beta = beta


    def f(self, t, x):
        perturbation = self.xp.zeros_like(x)
        perturbation[0] = x[1]
        perturbation[1] = (- self.alpha * x[0] - self.beta * x[0]**3) / self.m
        return perturbation


class Lorenz63Model(ODE):
    def __init__(self, sigma, rho, beta, xp_type="numpy"):
        super(Lorenz63Model, self).__init__(xp_type)
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def f(self, t, x):
        perturbation = self.xp.zeros_like(x)
        perturbation[0] = self.sigma * (x[1] - x[0])
        perturbation[1] = x[0] * (self.rho - x[2]) - x[1]
        perturbation[2] = x[0] * x[1] - self.beta * x[2]
        return perturbation


class Lorenz96Model(ODE):
    def __init__(self, N, F, xp_type="numpy"):
        super(Lorenz96Model, self).__init__(xp_type)
        self.N = N
        self.F = F

    def f(self, t, x):
        perturbation = self.xp.zeros_like(x)
        perturbation[0] = (x[1] - x[-2]) * x[-1] - x[0]
        perturbation[1] = (x[2] - x[-1]) * x[0] - x[1]
        perturbation[-1] = (x[0] - x[-3]) * x[-2] - x[-1]
        perturbation[2:-1] = (x[3:] - x[:-3]) * x[1:-2] - x[2:-1]
        return perturbation + self.F


class PeriodicDiffusion(ODE):
    def __init__(self, N, g=None, xp_type="numpy"):
        super(PeriodicDiffusion, self).__init__(xp_type)
        self.N = N
        if g is None:
            self.g = lambda x:x
        else:
            self.g = g

    def f(self, t, x):
        perturbation = self.xp.zeros_like(x)
        perturbation[0] = (- 2*x[0] + x[-1] + x[1])/4
        perturbation[-1] = (- 2*x[-1] + x[0] + x[-2])/4
        perturbation[1:-1] = (- 2*x[1:-1] + x[:-2] + x[2:])/4
        return perturbation + self.g(x)


class PeriodicAdvection(ODE):
    def __init__(self, dx, c, dt=0.01, scheme="upwind", xp_type="numpy"):
        super(PeriodicAdvection, self).__init__(xp_type)
        self.dx = dx
        self.c = c
        self.dt = dt
        if scheme in ["upwind", "LW"]:
            self.scheme = scheme
        else:
            raise ValueError("no existing scheme")

    def f(self, t, x):
        # upwind scheme
        perturbation = self.xp.zeros_like(x)

        r = self.c / self.dx
        if self.scheme=="upwind":
            if self.c > 0:
                perturbation[0] = r * (x[-1] - x[0])
                perturbation[1:] = r * (x[:-1] - x[1:])
            else:
                perturbation[-1] = r * (x[-1] - x[0])
                perturbation[:-1] = r * (x[:-1] - x[1:])
        elif self.scheme=="LW": # Lax-Wendroff
            perturbation[1:-1] = -r/2*(x[2:]-x[:-2]) + self.dt*r**2/2*(x[2:]-2*x[1:-1]+x[:-2])
            perturbation[0] = -r/2*(x[1]-x[-1]) + self.dt*r**2/2*(x[1]-2*x[0]+x[-1])
            perturbation[-1] = -r/2*(x[0]-x[-2]) + self.dt*r**2/2*(x[0]-2*x[-1]+x[-2])
        return perturbation


class VanderPol(ODE):
    def __init__(self, mu, xp_type="numpy"):
        super(VanderPol, self).__init__(xp_type)
        self.mu = mu

    def f(self, t, x):
        perturbation = self.xp.zeros_like(x)
        perturbation[0] = x[1]
        perturbation[1] = self.mu * (1 - x[0]**2) * v - x[0]
        return perturbation


class FitzHughNagumo(ODE):
    def __init__(self, a, b, c, I, xp_type="numpy"):
        super(FitzHughNagumo, self).__init__(xp_type)
        self.a = a
        self.b = b
        self.c = c
        self.I = I

    def f(self, t, x):
        perturbation = self.xp.zeros_like(x)
        perturbation[0] = self.c * (x[0] - x[1] - x[0]**3 / 3 + self.I(t))
        perturbation[1] = self.a + x[0] - self.b * x[1]
        return perturbation


class LotkaVolterra(ODE):
    def __init__(self, a, b, c, d, xp_type="numpy"):
        super(LotkaVolterra, self).__init__(xp_type)
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def f(self, t, x):
        perturbation = self.xp.zeros_like(x)
        perturbation[0] = self.a * x[0] - self.b * x[0] * x[1]
        perturbation[1] = self.c * x[0] * x[1] - self.d * x[1]
        return perturbation


class ClockReaction(ODE):
    def __init__(self, k1, k2, xp_type="numpy"):
        super(ClockReaction, self).__init__(xp_type)
        self.k1 = k1
        self.k2 = k2

    def f(self, t, x):
        #x0:A, x1:B, x2:T, x3:L
        perturbation = self.xp.zeros_like(x)
        perturbation[0] = - self.k1 * x[0] * x[1]
        perturbation[1] = - self.k1 * x[0] * x[1]
        perturbation[2] = self.k1 * x[0] * x[1] - self.k2 * x[2] * x[3]
        perturbation[3] = - self.k2 * x[2] * x[3]
        return perturbation


class OregonatorTyson(ODE):
    def __init__(self, q, f1, epsilon, xp_type="numpy"):
        super(OregonatorTyson, self).__init__(xp_type)
        self.q = q
        self.f1 = f1
        self.epsilon = epsilon

    def f(self, t, x):
        perturbation = self.xp.zeros_like(x)
        perturbation[0] = ((self.q - x[0]) / (self.q + x[0]) * self.f1 * x[1]
                             + x[0] * (1 - x[0])) / self.epsilon
        perturbation[1] = x[0] - x[1]
        return perturbation


class Oregonator(ODE):
    def __init__(self, q, f1, epsilon1, epsilon2, xp_type="numpy"):
        super(Oregonator, self).__init__(xp_type)
        self.q = q
        self.f1 = f1
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2

    def f(self, t, x):
        perturbation = self.xp.zeros_like(x)
        perturbation[0] = (self.q * x[1] - x[0] * x[1]
                             + x[0] * (1 - x[0])) / self.epsilon1
        perturbation[1] = (- self.q * x[1] - x[0] * x[1]
                             + self.f1 * x[2]) / self.epsilon2
        perturbation[2] = x[0] - x[2]
        return perturbation


class OregonatorCombination(ODE):
    def __init__(self, q, f1, epsilon1, epsilon2, intensity, xp_type="numpy"):
        super(OregonatorCombination, self).__init__(xp_type)
        self.q = q
        self.f1 = f1
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.intensity = intensity

    def f(self, t, x):
        perturbation = self.xp.zeros_like(x)
        perturbation[0] = (self.q * x[1] - x[0] * x[1]
                             + x[0] * (1 - x[0])
                             + self.intensity * (x[3] - x[0])) / self.epsilon1
        perturbation[1] = (- self.q * x[1] - x[0] * x[1]
                             + self.f1 * x[2]
                             + self.intensity * (x[4] - x[1])) / self.epsilon2
        perturbation[2] = x[0] - x[2]
        perturbation[3] = (self.q * x[3] - x[3] * x[4]
                             + x[3] * (1 - x[3])
                             + self.intensity * (x[0] - x[3])) / self.epsilon1
        perturbation[4] = (- self.q * x[4] - x[3] * x[4]
                             + self.f1 * x[5]
                             + self.intensity * (x[1] - x[4])) / self.epsilon2
        perturbation[5] = x[3] - x[5]
        return perturbation



def rotation_matrix_2d(theta):
    R = np.array([[math.cos(theta), -math.sin(theta)],
                  [math.sin(theta), math.cos(theta)]])
    return R        


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
