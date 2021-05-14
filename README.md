# pyassim
This package provides data assimilation (DA) methods for easy implementation by python.

## How to Import
For a quick installation::
`pip install pyassim`

`pyassim` depends on following packages:
- `numpy >= 1.13.3` (for core functionality)
- `cupy` (for gpu user. This package is not automatically installed as dependencies)


## How to Use
- Please see the codes under `samples/`
    - `sample_dom.py` is an example w.r.t. KF and LOCK through a twin experiment of dample oscillation model.
    - `sample_advection.py` is an example w.r.t. LLOCK, SLOCK, and LSLOCK through a twin experiment of advection equation.
    - `sample_Lorenz.py` is an example w.r.t. EnKF, NEnKF, PF, and GPF through a twin experiment of Lorenz 63 model.
- This repository includes following methods
    - [x] Kalman Filter (KF)
    - [x] Ensemble KF (EnKF)
    - [x] Nonlinear Ensemble KF (NEnKF)
    - [x] Local Ensemble Transform KF (LETKF)
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
    - [ ] Hidden Markov Model (HMM)
- Checked items are already developed, the others are under development