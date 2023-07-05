import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML
from numpy import loadtxt
from numpy import savetxt

from controller_utils import CartpoleDx, sample_initial_signal, Projection, episode_loss
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from mpc import mpc
from mpc.mpc import GradMethods, QuadCost
import mpc.util as eutil
from controllers import DDeePC

Tini = 4
m = 1
p = 4
Tf = 25
T = (m+1)*(Tini + Tf + p) + 4
n_batch = np.ones(100).astype(np.uint8)*30
device = 'cuda' if torch.cuda.is_available() else 'mps'
print(device)

ud = loadtxt('../badcartpole_ud.csv', delimiter=',')
yd = loadtxt('../badcartpole_yd.csv', delimiter=',')
yd = yd.reshape(T*p,)

noise =  np.random.randn(*yd.shape)*0.001
noiseu =  np.random.randn(*ud.shape)*0.01
yd = yd + noise
ud = ud + noiseu
dx = CartpoleDx().to(device)
projection = Projection()

def uniform(shape, low, high):
    r = high-low
    return torch.rand(shape)*r+low

u_constraints = np.ones(Tf)*4
y_constraints = np.kron(np.ones(Tf), np.array([0.25, 0.2, 0.15, 0.2]))
r = torch.ones(m)*0.01
q = torch.ones(p)*100 
q = torch.Tensor([100,10,100,10])
lam_g1 = torch.Tensor([500.409]).to(device)
lam_g2 = torch.Tensor([0.01]).to(device)
lam_y = torch.Tensor([210.258]).to(device)

episodes = 20
epochs = 20
print(n_batch)
for batch in n_batch:

    controller = DDeePC(
        ud=ud, yd=yd, u_constraints=u_constraints, y_constraints=y_constraints,
        Tini=Tini, T=T, N=Tf, m=m, p=p, n_batch=batch, device=device,
        linear=False, stochastic=True, q=q, r=r
    ).to(device)

    I, PI = controller.get_PI()
    PI = torch.Tensor(I-PI).to(device)

    pbar = tqdm(range(epochs))
    loss_params = np.zeros((epochs,5))
    opt = optim.Rprop(controller.parameters(), lr=0.01, step_sizes=(1e-4,50))
    # opt = optim.RMSprop(controller.parameters(), lr=100)

    for j in pbar:
                                                                                                                                            
        uini = (torch.zeros(size=(batch, Tini*m)) + torch.randn(batch, Tini*m)*0.001).to(device)
        uini += torch.randn(uini.shape).to(device)*0.01
        zero = torch.zeros(batch)
        th = uniform((batch), -0.01, 0.01)
        yini = torch.stack((zero, zero, th, zero), dim=1).repeat(1,Tini)
        y = yini[:,-p:].to(device)
        yini += torch.randn(yini.shape)*0.001
        yini = yini.to(device)
        traj = yini
        G, Ey, Eu, Y, U = torch.Tensor().to(device), torch.Tensor().to(device), torch.Tensor().to(device), torch.Tensor().to(device), torch.Tensor().to(device)
        
        for i in range(episodes):
            
            g, u_pred, _, sig_y, sig_u = controller(ref=None, uref=None, y_ini=yini, u_ini=uini)
            
            G, Ey, Eu = torch.cat((G, g.unsqueeze(1)), axis=1), torch.cat((Ey, sig_y.unsqueeze(1)), axis=1), torch.cat((Eu, sig_u.unsqueeze(1)), axis=1)
            input = u_pred[:,:m] + torch.randn(u_pred[:,:m].shape).to(device)*0.01
            U = torch.cat((U, input.unsqueeze(1)), axis=1)
            y = dx(y, input)
            Y = torch.cat((Y, y.unsqueeze(1)), axis=1)

            noise = torch.randn(y.shape).to(device)*0.001
            y += noise
            
            yini = torch.cat((yini[:, p:], y), axis=1)
            uini = torch.cat((uini[:, m:], input), axis=1)

        loss = episode_loss(G=G, U=U, Y=Y, Ey=Ey, Eu=Eu, controller=controller, PI=PI)
        
        loss_params[j,0] = loss.item()
        loss_params[j,1] = controller.lam_g1.data.item()
        loss_params[j,2] = controller.lam_g2.data.item()
        loss_params[j,3] = controller.lam_y.data.item()
        loss_params[j,4] = controller.lam_u.data.item()
        
        opt.zero_grad()   
        loss.backward()
        opt.step()
        controller.apply(projection)
        
        savetxt(f'data/rprop_{batch}.csv', loss_params, delimiter=',')
        
        pbar.set_description(f'l={loss.item():.3f}, ly={controller.lam_y.data.item():.3f},\
    l1={controller.lam_g1.data.item():.4f}, l2={controller.lam_g2.data.item():.3f}, lu={controller.lam_u.data.item():.3f}')

