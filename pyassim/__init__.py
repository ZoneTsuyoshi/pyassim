'''
=============
Assimilation Module
=============
This module provides inference methods for state-space estimation in continuous
spaces.
'''

from .kalman import KalmanFilter
from .ensemble import EnsembleKalmanFilter, NonlinearEnsembleKalmanFilter
from .particle import ParticleFilterGaussian
from .apf import AuxiliaryParticleFilter
from .gpf import GaussianParticleFilter
from .letkf import LocalEnsembleTransformKalmanFilter
from .sukf import SequentialUpdateKalmanFilter
from .osukf import OffsetsSequentialUpdateKalmanFilter
from .mvsukf import MissingValueSequentialUpdateKalmanFilter
from .semkf import SequentialExpectationMaximizationKalmanFilter
from .mtkf import MarkovTransitionKalmanFilter
from .gsmtpf import GaussianSoftMarkovTransitionParticleFilter
from .hidden import GaussianHiddenMarkovModel
from .model import DampedOccilationModel, CoefficientChangedDampedOccilationModel, DuffingModel, Lorentz63Model
from .scheme import EulerScheme

# __all__ = [
#     "KalmanFilter",
#     "EnsembleKalmanFilter",
#     "ParticleFilter",
#     "GaussianParticleFilter",
#     "LocalEnsembleTransformKalmanFilter",
#     "SequentialUpdateKalmanFilter"
# ]