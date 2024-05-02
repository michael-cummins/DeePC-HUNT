import torch.nn as nn
from numpy import genfromtxt
import numpy as np
import torch
from deepc_hunt.dynamics import CartpoleDx, RocketDx
from deepc_hunt import DeePC, Trainer
from deepc_hunt.controllers import npDeePC
from deepc_hunt.utils import tensor2np
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
from tqdm import tqdm

import coco_rocket_lander  # need to import to call gym.make()
from coco_rocket_lander.env import SystemModel


if __name__ == '__main__':
    
    q = torch.Tensor([40,10,1,1,3000,30]) # 6-tuple (x, y, x_dot, y_dot, theta, theta_dot)
    r = torch.Tensor([0.01,0.01,0.01]) # 3-tuple (F_E, F_S, phi)
    q_np = tensor2np(q)
    r_np = tensor2np(r)

    n = 6 # states
    m = 3 # inputs
    p = 6 # outputs

    Tini = 1 # number of past measurements (also called T_ini)
    Tf = 10 # number of future measurements (also called K)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_batch = 1

    ud = np.genfromtxt('../data/rocket_ud.csv', delimiter=',')
    yd = np.genfromtxt('../data/rocket_yd.csv', delimiter=',')

    y_constraints = np.ones(Tf*p)*1e5 #Unconstrained
    u_constraints = np.ones(Tf*m)

    # For training
    controller = DeePC(
        ud=ud, yd=yd, u_constraints=u_constraints, y_constraints=y_constraints,
        Tini=Tini, N=Tf, m=m, p=p, n_batch=n_batch, device=device,
        linear=False, stochastic_y=True, stochastic_u=False, q=q, r=r,
        # lam_g1=torch.Tensor([1e-5]), lam_g2=torch.Tensor([10]), 
        lam_y=torch.Tensor([1e5])
    ).to(device)

    epochs = 100
    time_steps = 30

    # Tune regularization params
    controller.initialise(lam_g1=0.1, lam_g2=0.1, lam_y=1000)
    deepc_tuner = Trainer(controller=controller, env=RocketDx())
    final_params = deepc_tuner.run(epochs=epochs, time_steps=time_steps)
    print(final_params)
