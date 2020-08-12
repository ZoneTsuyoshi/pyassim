Data Assimilation repository

- this repository is under development

# User Guide for This Package
## How to Import
- `python setup.py install`

## How to Test
- `python setup.py test`

## How to Use
- please see the codes under `samples/`
- this repository includes following methods
    - [x] Kalman Filter (KF)
    - [x] Ensemble KF (EnKF)
    - [x] Nonlinear Ensemble KF (NEnKF)
    - [x] Local EnKF (LETKF)
    - [x] Linear Opeartor Construction with the Kalman Filter (LOCK)
    - [x] Local LOCK (LLOCK)
    - [x] Spatially Uniform LOCK (SLOCK)
    - [x] Locally and Spatially Uniform LOCK (LSLOCK)
    - [x] Bayesian LOCK (BLOCK)
    - [x] Local BLOCK (LBLOCK)
    - [x] Locally and Spatially Uniform BLOCK (LSBLOCK)
    - [x] Particle Filter (PF)
    - [x] Gaussian PF (GPF)
    - [x] Variational Mapping PF (VMPF)
    - [x] Hidden Markov Model (HMM)
- if you tackle high-dimensional problems, this repository provides following methods for memory efficiency
    - [x] Kalman Filter (KF)
    - [x] Local LOCK (LLOCK)
    - [x] Spatially Uniform LOCK (SLOCK)
    - [x] Locally and Spatially Uniform LOCK (LSLOCK)
    - [] Local BLOCK (LBLOCK)
    - [] Locally and Spatially Uniform BLOCK (LSBLOCK)
- checked items are already developed, no checked items are under development