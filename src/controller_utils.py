import numpy as np
import matplotlib.pyplot as plt
import torch
from mpc import util
from torch import nn
from torch.autograd import Variable

def episode_loss(Y : torch.Tensor, U : torch.Tensor, G : torch.Tensor, E : torch.Tensor, controller, PI) -> torch.Tensor:
    """
    Calculate loss for for batch trajectory - pretty inificient, look into vectorizing
    Y should be shape(batch, T, p) - T is length of trajectory
    G should be shape(batch, T, Td-Tini-N+1)
    If doing reference tracking, Y and U are expected to be in delta formulation
    """
    n_batch = G.shape[0]
    T = G.shape[1]
    phi = torch.Tensor()

    # Not sure if I should include the cost/regularisation weights
    Q, R = torch.diag(controller.q), torch.diag(controller.r)
    ly = controller.lam_y.data if controller.stochastic else 0
    (lg1, lg2) = (controller.lam_g1, controller.lam_g2) if not controller.linear else (0, 0) 

    for i in range(n_batch):
        Ct, Cr = 0, 0
        for j in range(T):
            Ct += (Y[i,j,:].T @ Q @ Y[i,j,:] + U[i,j,:].T @ R @ U[i,j,:]).reshape(1)
            if not controller.linear:
                Cr += (torch.norm((PI)@G[i,j,:], p=2)**2)*lg1
                Cr += torch.norm((PI)@G[i,j,:], p=2)*lg2
            if controller.stochastic:
                Cr += torch.norm(E[i,j,:], p=1)*ly
        phi = torch.cat((phi, Ct+Cr), axis=0)

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
    high = 15
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

def block_hankel(w: np.array, L: int, d: int) -> np.array:
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

class Clamp(torch.autograd.Function):
    """
    https://discuss.pytorch.org/t/regarding-clamped-learnable-parameter/58474/4
    """
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=1e-3, max=1e6) # the value in iterative = 2

    @staticmethod 
    def backward(ctx, grad_output):
        return grad_output.clone()

class RechtDx(nn.Module):
    """
    torch enviornment for simple recht temperature control system
    """
    def __init__(self) -> None:
        super().__init__()
        self.A = torch.Tensor([[1.01, 0.01, 0.00], 
                [0.01, 1.01, 0.01], 
                [0.00, 0.01, 1.01]])
        
    def forward(self, x : torch.Tensor, u : torch.Tensor) -> torch.Tensor:
        if x.ndim > 1:
            batch_size = x.shape[0]
            self.A = self.A.repeat(batch_size, 1, 1)
        y = self.A @ x + u
        return y
        

class CartpoleDx(nn.Module):
    def __init__(self, params=None):
        super().__init__()

        self.n_state = 4
        self.n_ctrl = 1

        # model parameters
        if params is None:
            # gravity, masscart, masspole, length
            self.params = Variable(torch.Tensor((9.8, 1.0, 0.1, 0.5)))
        else:
            self.params = params
        assert len(self.params) == 4
        self.force_mag = 100.

        self.theta_threshold_radians = np.pi#12 * 2 * np.pi / 360
        self.x_threshold = 2.4
        self.max_velocity = 10

        self.dt = 0.05

        self.lower = -self.force_mag
        self.upper = self.force_mag

        # 0  1      2        3   4
        # x dx cos(th) sin(th) dth
        self.goal_state = torch.Tensor(  [ 0.,  0., 0.,   0.])
        self.goal_weights = torch.Tensor([0.1, 0.1,  1., 0.1])
        self.ctrl_penalty = 0.001

        self.mpc_eps = 1e-4
        self.linesearch_decay = 0.5
        self.max_linesearch_iter = 2

    def forward(self, state, u):
        squeeze = state.ndimension() == 1

        if squeeze:
            state = state.unsqueeze(0)
            u = u.unsqueeze(0)

        if state.is_cuda and not self.params.is_cuda:
            self.params = self.params.cuda()
        gravity, masscart, masspole, length = torch.unbind(self.params)
        total_mass = masspole + masscart
        polemass_length = masspole * length

        u = torch.clamp(u[:,0], -self.force_mag, self.force_mag)

        x, dx, th, dth = torch.unbind(state, dim=1)
        # th = torch.atan2(sin_th, cos_th)
        cos_th, sin_th = torch.cos(th), torch.sin(th)

        cart_in = (u + polemass_length * dth**2 * sin_th) / total_mass
        th_acc = (gravity * sin_th - cos_th * cart_in) / \
                 (length * (4./3. - masspole * cos_th**2 /
                                     total_mass))
        xacc = cart_in - polemass_length * th_acc * cos_th / total_mass

        x = x + self.dt * dx
        dx = dx + self.dt * xacc
        th = th + self.dt * dth
        dth = dth + self.dt * th_acc

        state = torch.stack((
            x, dx, th, dth
        ), 1)

        return state

    def get_frame(self, state, ax=None):
        state = util.get_data_maybe(state.view(-1))
        assert len(state) == 4
        x, dx, th, dth = torch.unbind(state)
        cos_th, sin_th = torch.cos(th), torch.sin(th)
        gravity, masscart, masspole, length = torch.unbind(self.params)
        th = np.arctan2(sin_th, cos_th)
        th_x = sin_th*length
        th_y = cos_th*length

        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))
        else:
            fig = ax.get_figure()
        ax.plot((x,x+th_x), (0, th_y), color='k')
        ax.set_xlim((-length*2, length*2))
        ax.set_ylim((-length*2, length*2))
        return fig, ax

    def get_true_obj(self):
        q = torch.cat((
            self.goal_weights,
            self.ctrl_penalty*torch.ones(self.n_ctrl)
        ))
        assert not hasattr(self, 'mpc_lin')
        px = -torch.sqrt(self.goal_weights)*self.goal_state #+ self.mpc_lin
        p = torch.cat((px, torch.zeros(self.n_ctrl)))
        return Variable(q), Variable(p)
    

class AffineDynamics(nn.Module):
    def __init__(self, A, B, c=None):
        super(AffineDynamics, self).__init__()

        assert A.ndimension() == 2
        assert B.ndimension() == 2
        if c is not None:
            assert c.ndimension() == 1

        self.A = A
        self.B = B
        self.c = c

    def forward(self, x, u):
        if not isinstance(x, Variable) and isinstance(self.A, Variable):
            A = self.A.data
            B = self.B.data
            c = self.c.data if self.c is not None else 0.
        else:
            A = self.A
            B = self.B
            c = self.c if self.c is not None else 0.

        x_dim, u_dim = x.ndimension(), u.ndimension()
        if x_dim == 1:
            x = x.unsqueeze(0)
        if u_dim == 1:
            u = u.unsqueeze(0)

        z = A@x + B@u + c

        if x_dim == 1:
            z = z.squeeze(0)

        return z
