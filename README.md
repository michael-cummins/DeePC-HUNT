# DeePC-HUNT
## Data enabled predictive control hyperparameter tuning via differentiable optimization layers

Source code for bachelor thesis carried out at the automatic control lab, ETH Zurich.
Supervised by [Alberto Padoan](https://www.albertopadoan.com/), [Keith Moffat](https://www.keithmoffat.com/) and [Florian Dorfler](http://people.ee.ethz.ch/~floriand/).

Developed in a conda enviornment on Ubuntu 22.04 with python 3.10. Source code and other experiments are all in ddeepc and necessary libraries are in requirements.txt
```
pip install -r requirements.txt
```
Differentiable DeePC layer is inspired by [Differentiable MPC](https://github.com/locuslab/differentiable-mpc) and built using [CvxpyLayers](https://github.com/cvxgrp/cvxpylayers).

## Hyperparameter Tuning
![Problem Formulation](https://github.com/michael-cummins/DeePC-HUNT/blob/main/videos/deepc_problem.png)

DeePC can achieve performance that rivals MPC on non-linear and stochastic systems but is highly sensitive to the choice of regularization parameters. We present a method for automatically tuning these parameters and validate its perforamce on a noisy [cartpole]() and [LTI]() system. Further technical details can be viewed in the report.

### Before training
![](https://github.com/michael-cummins/DeePC-HUNT/blob/main/videos/cartpole_init.mp4)
### After training (episode 70)
![](https://github.com/michael-cummins/DeePC-HUNT/blob/main/videos/cartpole_demo.mp4)

