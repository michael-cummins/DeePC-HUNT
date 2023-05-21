

import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import os
import io
import base64
import tempfile
from IPython.display import HTML
from numpy import loadtxt
from mpc import util
from numpy import savetxt

from controller_utils import CartpoleDx, sample_initial_signal, WeightClipper, episode_loss
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from mpc import mpc
from mpc.mpc import GradMethods, QuadCost, LinDx
import mpc.util as eutil
from controllers import DDeePC


Tini = 4
m = 1
p = 4
Tf = 20
T = (m+1)*(Tini + Tf + p) + 14
n_batch = 20
device = 'cuda' if torch.cuda.is_available() else 'mps'
# device = 'cpu'
print(device)
ud = loadtxt('../badcartpole_ud.csv', delimiter=',')
yd = loadtxt('../badcartpole_yd.csv', delimiter=',')
yd = yd.reshape(T*p,)
noise =  np.random.randn(*yd.shape)*0.001
noiseu =  np.random.randn(*ud.shape)*0.01
yd = yd + noise
ud = ud + noiseu
dx_deepc = CartpoleDx().to(device)
dx_mpc = CartpoleDx().to(device)
clipper = WeightClipper()
def uniform(shape, low, high):
    r = high-low
    return torch.rand(shape)*r+low

u_constraints = np.ones(Tf)*3
y_constraints = np.kron(np.ones(Tf), np.array([0.15, 0.5, 0.15, 0.5]))
r = torch.ones(m)*0.01
q = torch.Tensor([100,10,100,10])
lambdag = np.logspace(start=-3, stop=3, base=10, num=20)
lam_y = torch.Tensor([250.258]).to(device)

controller = DDeePC(
    ud=ud, yd=yd, u_constraints=u_constraints, y_constraints=y_constraints,
    Tini=Tini, T=T, N=Tf, m=m, p=p, n_batch=n_batch, device=device,
    linear=False, stochastic=True, q=q, r=r, lam_y=lam_y, lam_u=lam_y, lam_g1=lam_y, lam_g2=lam_y
).to(device)

lamy = [250]
episodes = 20
ref = torch.zeros(size=(n_batch,p))
perfect = torch.kron(torch.ones(episodes+Tini), ref).to(device)
ref = torch.kron(torch.ones(Tf), ref).to(device)
n_row = np.sqrt(n_batch).astype(np.uint8)
n_col = n_row
zero = torch.zeros(n_batch)
performance = np.zeros((20,20)) 
# (0,0) - l1=10e-4, l2=10e-4
# (0,20) - l1=10e4, l2=10e-4
loss = torch.nn.MSELoss()
I, PI = controller.get_PI()
PI = torch.Tensor(I-PI).to(device)

for i, ly in enumerate(lamy):
    
    controller.lam_y = torch.Tensor([ly]).to(device)

    for j, l2 in enumerate(lambdag):
        controller.lam_g2 = torch.Tensor([l2]).to(device)
        pbar = tqdm(lambdag)

        for k, l1 in enumerate(pbar):
            controller.lam_g1 = torch.Tensor([l1]).to(device)
            
            uini = torch.zeros(size=(n_batch, Tini*m)).to(device)
            noise = torch.randn(uini.shape).to(device)*0.01
            uini += noise

            th = uniform((n_batch), -0.01, 0.01)
            y =  torch.stack((zero, zero, th, zero), dim=1).to(device)
            yini = y.repeat(1,Tini).to(device)
            noise = torch.randn(yini.shape).to(device)*0.001
            deepc_traj = yini
            yini = yini + noise

            G, Ey, Eu, Y, U = torch.Tensor().to(device), torch.Tensor().to(device), torch.Tensor().to(device), torch.Tensor().to(device), torch.Tensor().to(device)

            violated = False
            for l in range(episodes):

                noise = torch.randn(y.shape).to(device)*0.001   

                pbar.set_description(f'violated = {violated}, episode = {l}, lam1 = {controller.lam_g1}, lam2={controller.lam_g2}')
                g, u_pred, _, sig_y, sig_u = controller(ref=ref, y_ini=yini, u_ini=uini)
                G, Ey, Eu = torch.cat((G, g.unsqueeze(1)), axis=1), torch.cat((Ey, sig_y.unsqueeze(1)), axis=1), torch.cat((Eu, sig_u.unsqueeze(1)), axis=1)
                input = u_pred[:,:m]
                U = torch.cat((U, input.unsqueeze(1)), axis=1)
                input += torch.randn(u_pred[:,:m].shape).to(device)*0.01
                y = dx_deepc(y, input)
                Y = torch.cat((Y, y.unsqueeze(1)), axis=1)
                
                if any(torch.abs(y[:,0]) >= 0.15) or any(torch.abs(y[:,1]) >= 0.5) or any(torch.abs(y[:,2]) >= 0.15) or any(torch.abs(y[:,3]) >= 0.5): 
                    violated = True
                    pbar.set_description(f'violated = {violated}, episode = {l}, lam1 = {controller.lam_g1}, lam2={controller.lam_g2}')
                    break
                
                yini = torch.cat((yini[:, p:], y+noise), axis=1)
                uini = torch.cat((uini[:, m:], input), axis=1)
                deepc_traj = torch.cat((deepc_traj, y), axis=1)

            if not violated:
                loss = episode_loss(G=G, U=U, Y=Y, Ey=Ey, Eu=Eu, controller=controller, PI=PI)
                performance[j,k] = loss
                # performance[j,k] = loss(input=deepc_traj, target=torch.zeros(deepc_traj.shape).to(device))
            elif violated: performance[j,k] = 0
            savetxt('performance_ly370.csv', performance, delimiter=',')
