# DeePC-HUNT
### Data-enabled predictive control hyperparameter tuning via differentiable optimization layers

DeePC-HUNT is a method for optimising over the hyperparameters of DeePC using [analytical policy gradients](https://arxiv.org/abs/2202.00817) and [differentiable optimization layers](https://locuslab.github.io/2019-10-28-cvxpylayers/). This method has been developed as part of my bachelor thesis, carried out at the [Automatic Control Laboratory (IfA)](https://control.ee.ethz.ch/). Supervised by [Alberto Padoan](https://www.albertopadoan.com/), [Keith Moffat](https://www.keithmoffat.com/) and [Florian Dorfler](http://people.ee.ethz.ch/~floriand/). 

Developed in a conda environment on Ubuntu 22.04 with Python 3.10. 

Differentiable DeePC layer is inspired by [Differentiable MPC](https://github.com/locuslab/differentiable-mpc) and built using [CvxpyLayers](https://github.com/cvxgrp/cvxpylayers).

## Installation
Install via pip
```
pip install deepc_hunt
```
Or clone the repo and install from source [Recommended]
```
cd DeePC-HUNT && pip install -e .
```
Extra packages necessary for running the example notebooks are in examples/requirements.txt if needed.
```
pip install -r examples/requirements.txt
```

DeePC-HUNT has the following dependencies.
* Python3
* [PyTorch](https://pytorch.org/) >= 1.0
* [CvxpyLayers](https://github.com/cvxgrp/cvxpylayers) >= 1.0

## Usage
Data-enabled Predictive Control ([DeePC](https://arxiv.org/abs/1811.05890)) is a data-driven non-parametric algorithm for combined identification (learning) and control of dynamical systems. It leverages the solution of the following optimization problem in a receding horizon fashion.

![Problem Formulation](https://github.com/michael-cummins/DeePC-HUNT/blob/main/videos/deepc_problem.png)
<!-- **DeePC Problem Formulation**
$$\min_{y,u,g,\sigma_y,\sigma_u} \sum_{i=0}^{T-1} ||y_i - r_{t+i}||_Q^2 + ||u_i||_R^2 + \theta_0||(I-\Pi)g||_2^2 + \theta_1|g|_1 + \theta_2|\sigma_y|_1 + \theta_3|\sigma_u|_1$$
    
$$\textrm{subject to} \begin{pmatrix} {U_p \\ Y_p \\ U_f \\ Y_f} \end{pmatrix}g = \begin{pmatrix} u_\textrm{ini} \\ y_\textrm{ini}  \\ u \\ y \end{pmatrix} + \begin{pmatrix} \sigma_u \\ \sigma_y \\ 0 \\ 0 \end{pmatrix} $$

$$\begin{pmatrix} 1 & 2 & 3 \end{pmatrix}$$
    
$$u \in \mathcal{U}, y \in \mathcal{Y}$$ -->

DeePC can achieve performance that rivals MPC on non-linear and stochastic systems ([see here](https://arxiv.org/abs/2101.01273)) but is highly sensitive to the choice of regularization parameters $\theta_i$. DeePC-HUNT addresses this problem by automatically tuning these parameters. The performance of DeePC-HUNT has been validated on a [rocket lander](https://github.com/michael-cummins/DeePC-HUNT/examples/rocket.ipynb) modelling the falcon 9, a noisy [cartpole](https://github.com/michael-cummins/DeePC-HUNT/examples/cartpole_ddeepc.ipynb) and a [LTI](https://github.com/michael-cummins/DeePC-HUNT/examples/linear_deepc.ipynb) system. To run these example notebooks, you can clone this directory and open it in a VS-Code environment with the Jupyter Notebook extension

### Rocket - before training

https://github.com/michael-cummins/DeePC-HUNT/assets/72135336/eacd384e-b69d-428e-aa3e-1b4e669a3485




### Rocket - after training (episode 70)

After running DeePC-HUNT for 70 episodes, the controller now stabilizes the system.




https://github.com/michael-cummins/DeePC-HUNT/assets/72135336/b88aedf8-b816-4a89-a9ef-3bfa9b9c9d0b


