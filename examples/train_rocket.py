import torch.nn as nn
import numpy as np
import torch
from deepc_hunt.dynamics import RocketDx
from deepc_hunt import DeePC, Trainer
from deepc_hunt.controllers import npDeePC, npMPC
from deepc_hunt.utils import tensor2np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    q = torch.Tensor([100,10,5,1,3000,30])
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

    ud = np.genfromtxt('data/rocket_ud.csv', delimiter=',')
    yd = np.genfromtxt('data/rocket_yd.csv', delimiter=',')

    # y_constraints = np.ones(Tf*p)*1e5 #Unconstrained
    # u_constraints = np.ones(Tf*m)
    y_upper = np.kron(np.ones(Tf), np.array([33,26.6,100,100,0.6,100]))
    y_lower = np.kron(np.ones(Tf), np.array([0,7,-100,-100,-0.6,-100]))
    u_upper = np.kron(np.ones(Tf), np.array([1,1,1]))
    u_lower = np.kron(np.ones(Tf), np.array([0,-1,-1]))
    y_constraints = (y_lower, y_upper)
    u_constraints = (u_lower, u_upper)

    # For training
    controller = DeePC(
        ud=ud, yd=yd, u_constraints=u_constraints, y_constraints=y_constraints,
        Tini=Tini, N=Tf, m=m, p=p, n_batch=n_batch, device=device,
        linear=False, stochastic_y=True, stochastic_u=False, q=q, r=r,
        # lam_g1=torch.Tensor([1e-5]), lam_g2=torch.Tensor([10]), lam_y=torch.Tensor([1e5])
    ).to(device)

    epochs = 100
    time_steps = 20
    rocket = RocketDx(true_model=False)

    # Tune regularization params
    controller.initialise(lam_g1=50, lam_g2=50, lam_y=1000)
    deepc_tuner = Trainer(controller=controller, env=rocket)
    final_params = deepc_tuner.run(epochs=epochs, time_steps=time_steps)