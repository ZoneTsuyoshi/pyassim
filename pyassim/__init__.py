'''
=============
Assimilation Module
=============
This module provides inference methods for state-space estimation in continuous
spaces.
'''

from .apf import AuxiliaryParticleFilter
from .ensemble import EnsembleKalmanFilter, NonlinearEnsembleKalmanFilter
from .gpf import GaussianParticleFilterGauss
# from .hidden import GaussianHiddenMarkovModel
from .kalman import KalmanFilter
from .letkf import LocalEnsembleTransformKalmanFilter
from .llock import LocalLOCK
from .lock import LOCK
from .lslock import LSLOCK
# from .lslock_me import LSLOCKME
from .model import *
from .particle import ParticleFilterGaussian
from .scheme import EulerScheme, RungeKuttaScheme
from .semkf import SequentialExpectationMaximizationKalmanFilter
from .slock import SpatiallyUniformLOCK
# from .strong_4dvar import Strong4DVar
from .vmpf import GaussianVariationalMappingParticleFilter
from .unscented import UnscentedKalmanFilter