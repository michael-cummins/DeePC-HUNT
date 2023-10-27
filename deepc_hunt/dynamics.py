import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np
from deepc_hunt.utils import tensor2np
import matplotlib.pyplot as plt

class RocketDx(nn.Module):

    """
    Dynamics for rocket lander
        x = [x, y, x_dot, y_dot, theta, theta_dot]
        u = [F_E, F_s, phi]
    """

    def __init__(self, x_init : torch.Tensor):
        super().__init__()
        squeeze = x_init.ndimension() == 1
        if squeeze:
            x_init = x_init.unsqueeze(0)
        # Params from the COCO rocket lander env
        self.Ts : float = 1/60
        self.g = 9.81
        self.lander_scaling : float = 4 
        self.scale : int = 30
        self.mass = 30000 # kg
        # self.main_engine_thrust = (6.8e6*(self.lander_scaling**3))/(self.scale**3)
        # self.side_engine_thrust = (6.8e6*(self.lander_scaling**3)/50)/(self.scale**3)
        # self.nozzle_torque: float = 500 * self.lander_scaling
        self.main_engine_thrust = 6.8e6
        self.side_engine_thrust = 6.8e6/50
        self.nozzle_torque = 500
        self.max_nozzle_angle: float = 15 * np.pi / 180
        self.l1 : float = 42.7/2
        self.l2 : float = 42.7/2
        self.ln :float = 42.7/32
        self.x = x_init
        print(f'F_e = {self.main_engine_thrust}, F_s = {self.side_engine_thrust}, Nozzle Torque : {self.nozzle_torque}')

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:

        squeeze_x = x.ndimension() == 1
        squeeze_u = u.ndimension() == 1

        if squeeze_x:
            x = x.unsqueeze(0)
        if squeeze_u:
            u = u.unsqueeze(0)
        
        # print(self.x)
        # Rescale inputs
        u[:,0] = torch.clamp(u[:,0], min=0, max=1)*self.main_engine_thrust
        u[:,1] = torch.clamp(u[:,1], min=-1, max=1)*self.side_engine_thrust
        u[:,2] = torch.clamp(u[:,2], min=-self.max_nozzle_angle, max=self.max_nozzle_angle)
        # print(u)
        
        z = x
        # Roll sim forward
        z[:,0] = x[:,0] + self.Ts*x[:,2] # X
        z[:,1] = x[:,1] + self.Ts*x[:,3] # Y
        z[:,4] = x[:,4] + self.Ts*x[:,5] # Theta
        z[:,2] = x[:,2] + self.Ts*(u[:,0]*(x[:,4] + u[:,2]) + u[:,1])/self.mass
        z[:,3] = x[:,3] + self.Ts*(u[:,0]*(1 - u[:,2]*x[:,4]) - u[:,1]*x[:,4] - self.mass*self.g)/self.mass
        z[:,5] = x[:,5] + self.Ts*(-u[:,0]*u[:,2]*(self.l1 + self.ln) + self.l2*u[:,1])/self.nozzle_torque
        return z

class Env(nn.Module):

    def __init__(self, f, discrete=False, Ts=None):
        if Ts is not None and discrete==True:
            raise AssertionError('If not discrete, then you must supply a sample time Ts')
        if discrete and Ts:
            raise AssertionError('Discrete does not require a sample time -> Ts=None')

        self.Ts = Ts
        self.discrete = discrete
        self.f = f
        
    
    def f(x : torch.Tensor, u : torch.Tensor) -> torch.Tensor:
        return x
    
    def forward(self, x : torch.Tensor, u : torch.Tensor) -> torch.Tensor:
        
        # x is shape (n_batch, p)
        # u is shape (n_batch, m)

        x_dim, u_dim = x.ndimension(), u.ndimension()
        if x_dim == 1:
            x = x.unsqueeze(0)
        if u_dim == 1:
            u = u.unsqueeze(0)

        if self.discrete:
            z = self.f(x, u)
        else:
            z = x + self.Ts*self.f(x, u)
        
        if x_dim == 1:
            z = z.squeeze(0)
        
        return z

class dynamics(Env):
    def __init__(self, discrete=False, Ts=None):
        super().__init__()
        self.discrete = discrete
        print(self.discrete)
        self.A = torch.Tensor([[1.01, 0.01, 0.00], # A - State-space matrix
                                [0.01, 1.01, 0.01], 
                                [0.00, 0.01, 1.01]])
    def f(self, x, u):
        return x@self.A + u
    
class AffineDynamics(nn.Module):
    def __init__(self, A, B, c=None):
        super(AffineDynamics, self).__init__()

        assert A.ndimension() == 2
        assert B.ndimension() == 2
        if c is not None:
            assert c.ndimension() == 1

        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)
        self.c = nn.Parameter(c) if c is not None else c
        self.obs_noise_std = 0.1
        self.input_noise_std = 0.1

    def forward(self, x, u):


        x_dim, u_dim = x.ndimension(), u.ndimension()
        if x_dim == 1:
            x = x.unsqueeze(0)
        if u_dim == 1:
            u = u.unsqueeze(0)
        
        # u += torch.randn(u.shape) * self.input_noise_std

        z = x@self.A.T + u@self.B.T
        z += self.c if self.c is not None else 0

        if x_dim == 1:
            z = z.squeeze(0)

        return z + torch.randn(z.shape).to(z.device) * self.obs_noise_std

class CartpoleDx(nn.Module):
    def __init__(self, params=None):
        super().__init__()

        self.n_state = 4
        self.n_ctrl = 1
        self.input_noise_std = 0.01
        self.output_noise_std = 0.001

        # model parameters
        if params is None:
            # gravity, masscart, masspole, length
            self.params = Variable(torch.Tensor((9.8, 1.0, 0.1, 0.5)))
        else:
            self.params = params
        assert len(self.params) == 4
        self.force_mag = 3.

        self.theta_threshold_radians = np.pi#12 * 2 * np.pi / 360
        self.x_threshold = 2.4
        self.max_velocity = 10

        self.dt = 0.05

    def forward(self, state, u):
        squeeze = state.ndimension() == 1

        if squeeze:
            state = state.unsqueeze(0)
            u = u.unsqueeze(0)
        
        device = 'cuda' if state.is_cuda else 'cpu'
        u += torch.randn(u.shape).to(device)*self.input_noise_std
        
        if state.is_cuda and not self.params.is_cuda:
            self.params = self.params.cuda()
        gravity, masscart, masspole, length = torch.unbind(self.params)
        total_mass = masspole + masscart
        polemass_length = masspole * length

        u = torch.clamp(u[:,0], -self.force_mag, self.force_mag)

        x, dx, th, dth = torch.unbind(state, dim=1)
        cos_th, sin_th = torch.cos(th), torch.sin(th)

        cart_in = (u + polemass_length * dth**2 * sin_th) / total_mass
        th_acc = (gravity * sin_th - cos_th * cart_in) / (length * (4./3. - masspole * cos_th**2 / total_mass))
        xacc = cart_in - polemass_length * th_acc * cos_th / total_mass

        x = x + self.dt * dx
        dx = dx + self.dt * xacc
        th = th + self.dt * dth
        dth = dth + self.dt * th_acc

        state = torch.stack((
            x, dx, th, dth
        ), 1)

        state += torch.randn(state.shape).to(device)*self.output_noise_std

        return state
    
    def get_data_maybe(self, x):
        return x if not isinstance(x, Variable) else x.data

    def get_frame(self, state, ax=None):
        state = self.get_data_maybe(state.view(-1))
        assert len(state) == 4
        x, dx, th, dth = torch.unbind(state)
        cos_th, sin_th = torch.cos(th), torch.sin(th)
        gravity, masscart, masspole, length = torch.unbind(self.params)
        th = torch.arctan2(sin_th, cos_th)
        th_x = sin_th*length
        th_y = cos_th*length

        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))
        else:
            fig = ax.get_figure()
        x, th_x, th_y, length = tensor2np(x), tensor2np(th_x), tensor2np(th_y), tensor2np(length)
        ax.plot((x,x+th_x), (0, th_y), color='k')
        ax.set_xlim((-length*2, length*2))
        ax.set_ylim((-length*2, length*2))
        return fig, ax