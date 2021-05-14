# pyassim
This package provides data assimilation (DA) methods for easy implementation by python.

## How to Import
For a quick installation::

```
$ pip install pyassim
```

or

```
$ easy_install pyassim
```

Alternatively, you can get the latest and greatest from github::

```
$ git clone https://github.com/ZoneTsuyoshi/pyassim.git
$ cd pyassim
$ sudo python setup.py install
```


`pyassim` depends on following packages:
- `numpy >= 1.13.3` (for core functionality)
- `cupy` (for gpu calculation. This package is not automatically installed as dependencies)


## How to Use
Please see the codes under `samples/`
- `sample_dom.py` is an example w.r.t. KF and LOCK through a twin experiment of dample oscillation model.
- `sample_advection.py` is an example w.r.t. LLOCK, SLOCK, and LSLOCK through a twin experiment of advection equation.
- `sample_Lorenz.py` is an example w.r.t. EnKF, NEnKF, PF, and GPF through a twin experiment of Lorenz 63 model.


This repository includes following methods:
| Methods | Paper |
| :-- | :-- |
| Kalman Filter (KF) | Kalman R. E. (1960): [A New Approach to Linear Filtering and Prediction Problems](https://asmedigitalcollection.asme.org/fluidsengineering/article-abstract/82/1/35/397706/A-New-Approach-to-Linear-Filtering-and-Prediction?redirectedFrom=fulltext)

- [x] [Kalman Filter (KF)](https://asmedigitalcollection.asme.org/fluidsengineering/article-abstract/82/1/35/397706/A-New-Approach-to-Linear-Filtering-and-Prediction?redirectedFrom=fulltext)
- [x] [Ensemble KF (EnKF)](https://link.springer.com/article/10.1007/s10236-003-0036-9)
- [x] Nonlinear Ensemble KF (NEnKF)
- [x] [Local Ensemble Transform KF (LETKF)](https://www.sciencedirect.com/science/article/pii/S0167278906004647)
- [x] Particle Filter (PF)
- [x] [Gaussian PF (GPF)](https://ieeexplore.ieee.org/document/1232326)
- [x] [Variational Mapping PF (VMPF)](https://arxiv.org/abs/1805.11380)
- [x] [Linear Opeartor Construction with the Kalman Filter (LOCK)](https://arxiv.org/abs/2001.11256)
- [x] [Local LOCK (LLOCK)](https://arxiv.org/abs/2001.11256)
- [x] [Spatially Uniform LOCK (SLOCK)](https://arxiv.org/abs/2001.11256)
- [x] [Locally and Spatially Uniform LOCK (LSLOCK)](https://link.springer.com/chapter/10.1007/978-3-030-58653-9_33)
- [x] Bayesian LOCK (BLOCK)
- [x] Local BLOCK (LBLOCK)
- [x] Locally and Spatially Uniform BLOCK (LSBLOCK)
- [ ] Hidden Markov Model (HMM)

Checked items are already developed, the others are under development