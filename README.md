# DeePC-HUNT
## Data enabled predictive control hyperparameter tuning via differentiable optimization layers

Source code for bachelor thesis carried out at the automatic control lab, ETH Zurich.
Supervised by [Alberto Padoan](https://www.albertopadoan.com/), [Keith Moffat](https://www.keithmoffat.com/) and [Florian Dorfler](http://people.ee.ethz.ch/~floriand/).

Developed in a conda environment on Ubuntu 22.04 with python 3.10. Source code is in deepc_hunt and all experiments are in the examples directory

Differentiable DeePC layer is inspired by [Differentiable MPC](https://github.com/locuslab/differentiable-mpc) and built using [CvxpyLayers](https://github.com/cvxgrp/cvxpylayers).

## Hyperparameter Tuning
![Problem Formulation](https://github.com/michael-cummins/DeePC-HUNT/blob/main/videos/deepc_problem.png)

DeePC can achieve performance that rivals MPC on non-linear and stochastic systems but is highly sensitive to the choice of regularization parameters. We present a method for automatically tuning these parameters and validate its performance on a noisy [cartpole](https://github.com/michael-cummins/DeePC-HUNT/ddeepc/cartpole_ddeepc.ipynb) and [LTI](https://github.com/michael-cummins/DeePC-HUNT/ddeepc/linear_ddeepc.ipynb) system. Further technical details can be viewed in the report.

## Installing and running the examples
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

### Cartpole - before training



https://github.com/michael-cummins/DeePC-HUNT/assets/72135336/1c73ca21-91c1-4669-a04d-1d9a103d9d48



### Cartpole - after training (episode 70)




https://github.com/michael-cummins/DeePC-HUNT/assets/72135336/b6fa9948-a768-4646-b87f-018395fcc6c4

