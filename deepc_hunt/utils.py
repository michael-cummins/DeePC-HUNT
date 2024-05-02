import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import Parameter

def episode_loss(Y : torch.Tensor, U : torch.Tensor, G : torch.Tensor, controller, PI,
                 Ey=None, Eu=None) -> torch.Tensor:
    
    """
    Calculate loss for for batch trajectory - pretty inificient, look into vectorizing
    Y should be shape(batch, T, p) - T is length of trajectory
    G should be shape(batch, T, Td-Tini-N+1)
    If doing reference tracking, Y and U are expected to be in delta formulation
    """
    
    PI = torch.Tensor(PI) if isinstance(PI, np.ndarray) else PI
    n_batch = G.shape[0]
    T = Y.shape[1]
    phi = torch.Tensor().to(controller.device)
    Q, R = torch.diag(controller.q).to(controller.device), torch.diag(controller.r).to(controller.device)
    
    for i in range(n_batch):
        Ct, Cr = 0, 0
        for j in range(T):
            Ct += (Y[i,j,:].T @ Q @ Y[i,j,:] + U[i,j,:].T @ R @ U[i,j,:]).reshape(1)
            # if not controller.linear:
            #     Cr += torch.norm((PI)@G[i,j,:], p=2)**2
            #     Cr += torch.norm(G[i,j,:], p=1)
            # if Ey is not None:
            #     Cr += torch.norm(Ey[i,j,:], p=1) 
            # if Eu is not None:
            #     Cr += torch.norm(Eu[i,j,:], p=1)
        phi = torch.cat((phi, Ct), axis=0)
    loss = torch.sum(phi)/n_batch
    return loss

def sample_initial_signal(Tini : int, p : int, m : int, batch : int, ud : np.array, yd : np.array) -> torch.Tensor:
    
    """
    Samples initial signal trajectory from system data
    args:
        Tini = Initial time
        p = Dimension of output signal
        m = Dimension of input signal
        batch = nunmber of batches
        ud  = System input data
        yd = system output data
    """
    
    T = yd.shape[0]
    high=T-Tini
    if batch>T:
        raise Exception('Biased estimate of closed loop cost')
    index = np.random.uniform(size=(batch,), low=0, high=high).astype(np.uint8)
    if ud.ndim > 1:
        sampled_uini = np.array([ud[ind:Tini + ind, :].reshape((Tini*m,)) for ind in index])
    else:
        sampled_uini = np.array([ud[ind:Tini*m + ind].reshape((Tini*m,)) for ind in index])
    if yd.ndim > 1:
        sampled_yini = np.array([yd[ind:Tini + ind, :].reshape((Tini*p,)) for ind in index])
    else:
        sampled_yini = np.array([yd[ind:Tini*p + ind].reshape((Tini*p,)) for ind in index])
    u_ini, y_ini = torch.Tensor(sampled_uini), torch.Tensor(sampled_yini)
    return u_ini, y_ini

def block_hankel(w: np.ndarray, L: int, d: int) -> np.ndarray:
    """
    Builds block Hankel matrix for column vector w of order L
    args:
        w = column vector
        p = dimension of each block in w
        L = order of hankel matrix
    """
    T = int(len(w)/d)
    if L > T:
        raise ValueError('L must be smaller than T')
    H = np.zeros((L*d, T-L+1))
    for i in range(0, T-L+1):
        H[:,i] = w[d*i:d*(L+i)]
    return H

def block_hankel_torch(w: torch.Tensor, L: int, d: int) -> torch.Tensor:
    T = int(len(w)/d)
    if L > T:
        raise ValueError('L must be smaller than T')
    H = torch.zeros((L*d, T-L+1))
    for i in range(0, T-L+1):
        H[:,i] = w[d*i:d*(L+i)]
    return H

class Projection(object):

    def __init__(self, frequency=1, lower=1e-5, upper=1e5):
        self.frequency = frequency
        self.lower = lower 
        self.upper = upper

    def __call__(self, module):
        # filter the variables to get the ones you want
        # if hasattr(module, 'weight'):
        for param in module.parameters():
            w = param.data
            w = w.clamp(self.lower,self.upper)
            param.data = w


def tensor2np(tensor : torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()