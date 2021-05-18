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


`pyassim` provides easy implementation for following methods:
| Methods | Paper | link |
| :-- | :-- | :-- |
| Kalman Filter (KF) | Kalman R.E. (1960): A New Approach to Linear Filtering and Prediction Problems | [ASME](https://asmedigitalcollection.asme.org/fluidsengineering/article-abstract/82/1/35/397706/A-New-Approach-to-Linear-Filtering-and-Prediction?redirectedFrom=fulltext) |
| Ensemble KF (EnKF) | Evensen G. (2003): The Ensemble Kalman Filter: theoretical formulation and practical implementation | [SpringerLink](https://link.springer.com/article/10.1007/s10236-003-0036-9) |
| Local Ensemble Transform KF (LETKF) | Hunt B.R. + (2006): Efficient data assimilation for spatiotemporal chaos: A local ensemble transform Kalman filter | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0167278906004647) |
| Particle Filter (PF) | Kitagawa G. (1996): Monte Carlo Filter and Smoother for Non-Gaussian Nonlinear State Space Models | [JSTOR](https://www.jstor.org/stable/1390750?seq=1#metadata_info_tab_contents) |
| Gaussian PF (GPF) | Kotecha J.H. + (2003): Gaussian particle filtering | [IEEE](https://ieeexplore.ieee.org/document/1232326) |
| Variational Mapping PF (VMPF) | Pulido M. + (2018): Kernel embedding of maps for sequential Bayesian inference: The variational mapping particle filter | [arXiv](https://arxiv.org/abs/1805.11380) |
| Linear Opeartor Construction with the Kalman Filter (LOCK) | Ishizone T. + (2020): Real-time Linear Operator Construction and State Estimation with the Kalman Filter | [arXiv](https://arxiv.org/abs/2001.11256) |
| Local LOCK (LLOCK) | Ishizone T. + (2020): Real-time Linear Operator Construction and State Estimation with the Kalman Filter | [arXiv](https://arxiv.org/abs/2001.11256) |
| Spatially Uniform LOCK (SLOCK) | Ishizone T. + (2020): Real-time Linear Operator Construction and State Estimation with the Kalman Filter | [arXiv](https://arxiv.org/abs/2001.11256) |
| Locally and Spatially Uniform LOCK (LSLOCK) | Ishizone T. + (2020): LSLOCK: A Method to Estimate State Space Model by Spatiotemporal Continuity | [SpringerLink](https://link.springer.com/chapter/10.1007/978-3-030-58653-9_33) |
