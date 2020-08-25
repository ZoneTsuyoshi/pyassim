'''
=============
Assimilation Module
=============
This module provides inference methods for state-space estimation in continuous
spaces.
'''

from .kalman import KalmanFilter
# from .kalman_me import KalmanFilterME
from .ensemble import EnsembleKalmanFilter, NonlinearEnsembleKalmanFilter
from .particle import ParticleFilterGaussian
from .apf import AuxiliaryParticleFilter
from .gpf import GaussianParticleFilterGauss
from .letkf import LocalEnsembleTransformKalmanFilter
from .lock import LOCK
from .llock import LocalLOCK
from .slock import SpatiallyUniformLOCK
from .lslock import LSLOCK
# from .lslock_me import LSLOCKME
# from .sukf import SequentialUpdateKalmanFilter
# from .bsukf import BayesianSequentialUpdateKalmanFilter
# from .osukf import OffsetsSequentialUpdateKalmanFilter
# from .mvsukf import MissingValueSequentialUpdateKalmanFilter
from .semkf import SequentialExpectationMaximizationKalmanFilter
# from .mtkf import MarkovTransitionKalmanFilter
# from .gsmtpf import GaussianSoftMarkovTransitionParticleFilter
from .hidden import GaussianHiddenMarkovModel
from .vmpf import GaussianVariationalMappingParticleFilter
from .strong_4dvar import Strong4DVar
from .model import *
from .scheme import EulerScheme, RungeKuttaScheme

# __all__ = [
#     "KalmanFilter",
#     "EnsembleKalmanFilter",
#     "ParticleFilter",
#     "GaussianParticleFilter",
#     "LocalEnsembleTransformKalmanFilter",
#     "SequentialUpdateKalmanFilter"
# ]