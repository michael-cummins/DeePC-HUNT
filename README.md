# DeePC-HUNT
### Data enabled predictive control hyperparameter tuning via differentiable optimization layers

DeePC-HUNT is a method for optimising over the hyperparameters of DeePC, using anlaytical policy gradients. This method has been developed as part of my bachelor thesis, carried out at the [automatic control lab](https://control.ee.ethz.ch/). Supervised by [Alberto Padoan](https://www.albertopadoan.com/), [Keith Moffat](https://www.keithmoffat.com/) and [Florian Dorfler](http://people.ee.ethz.ch/~floriand/). 

Developed in a conda environment on Ubuntu 22.04 with python 3.10. 

Differentiable DeePC layer is inspired by [Differentiable MPC](https://github.com/locuslab/differentiable-mpc) and built using [CvxpyLayers](https://github.com/cvxgrp/cvxpylayers).

## Installation
install via pip
```
pip install deepc_hunt
```
Or clone the repo and install via
```
cd DeePC-HUNT && pip install -e .
```
All extra packages necessary for running the example notebookds are in examples/requirements.txt. If needed,
```
pip install -r examples/requirements.txt
```

DeePC-HUNT has the following dependancies.
* Python3
* [PyTorch](https://pytorch.org/) >= 1.0
* [CvxpyLayers](https://github.com/cvxgrp/cvxpylayers) >= 1.0

## Usage
Data-​enabled Predictive Control ([DeePC](https://arxiv.org/abs/1811.05890)) is a data-​driven non-​parametric algorithm for combined identification (learning) and control of dynamical systems. It leverages on the solution of the following optimization problem in a receding horizon fashion

![Problem Formulation](https://github.com/michael-cummins/DeePC-HUNT/blob/main/videos/deepc_problem.png)

DeePC can achieve performance that rivals MPC on non-linear and stochastic systems ([see here](https://arxiv.org/abs/2101.01273)), but is highly sensitive to the choice of regularization parameters $\theta_i$. DeePC-HUNT addresses this problem by automatically tuning these parameters. The performance of DeePC-HUNT has been validated on a noisy [cartpole](https://github.com/michael-cummins/DeePC-HUNT/ddeepc/cartpole_ddeepc.ipynb) and [LTI](https://github.com/michael-cummins/DeePC-HUNT/ddeepc/linear_ddeepc.ipynb) system. These examples can be viewed in the notebooks in the examples directory.

### Cartpole - before training



https://github.com/michael-cummins/DeePC-HUNT/assets/72135336/1c73ca21-91c1-4669-a04d-1d9a103d9d48



### Cartpole - after training (episode 70)




https://github.com/michael-cummins/DeePC-HUNT/assets/72135336/b6fa9948-a768-4646-b87f-018395fcc6c4

