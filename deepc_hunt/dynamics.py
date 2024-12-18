import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np
from deepc_hunt.utils import tensor2np
import matplotlib.pyplot as plt
from typing import Tuple
import scipy
class RocketDx(nn.Module):

    """
    Dynamics for rocket lander
        x = [x, y, x_dot, y_dot, theta, theta_dot]
        u = [F_E, F_s, phi]
    """

    def __init__(self, true_model=True):
        super().__init__()
        
        # Params from the COCO rocket lander env
        self.lander_scaling : float = 4 
        self.scale : int = 30
        self.const: float = (self.lander_scaling**3)/(self.scale**3)
        self.nozzle_torque = 500*self.lander_scaling
        self.nozzle_inertia = 0.21696986258029938
        self.true_model = true_model
        self.Ts : float = 1/60
        self.g = 9.81
        self.mass = 530.4058532714844 # kg
        self.main_engine_thrust = 16118.518518518518
        self.side_engine_thrust = 322.3703703703704
        self.inertia = 1209.53515625
        self.max_nozzle_angle: float = 0.2617993877991494
        self.l1 : float = 2.8466666666666667
        self.l2 : float = 2.1350000000000002
        self.ln : float = 0

        if not true_model:
            self.l1, self.l2 = self.l2, self.l1
            self.mass = self.mass/2

        self.state_shape = 6
        self.action_shape = 3

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:

        squeeze_x = x.ndimension() == 1
        squeeze_u = u.ndimension() == 1

        if squeeze_x:
            x = x.unsqueeze(0)
        if squeeze_u:
            u = u.unsqueeze(0)
        
        # Rescale inputs
        F_e = torch.clamp(u[:,0], min=0, max=1)*self.main_engine_thrust
        F_s = torch.clamp(u[:,1], min=-1, max=1)*self.side_engine_thrust
        phi = torch.clamp(u[:,2], min=-1, max=1)*self.max_nozzle_angle
        sin_phi = torch.sin(phi)

        z = x.clone()

        # Theta dot
        z[:,5] = x[:,5] + self.Ts*(-F_e*sin_phi*self.l1 - self.l2*F_s)/self.inertia
        # x dot
        z[:,2] = x[:,2] + self.Ts*(-F_e*torch.sin(x[:,4] + phi) + F_s*torch.cos(x[:,4]))/self.mass
        # y dot
        z[:,3] = x[:,3] + self.Ts*(F_e*torch.cos(x[:,4] + phi) + F_s*torch.sin(x[:,4]) - self.mass*self.g)/self.mass

        # Roll sim forward
        z[:,0] = x[:,0] + self.Ts*x[:,2] # X
        z[:,1] = x[:,1] + self.Ts*x[:,3] # Y
        z[:,4] = x[:,4] + self.Ts*x[:,5] # Theta

        return z

    def linearise(self, x_eq: np.ndarray, u_eq: np.ndarray, discrete: bool) -> Tuple[np.ndarray]:
        # linearized state dynamics
        a24 = (
            -u_eq[1] * np.sin(x_eq[4]) - u_eq[0] * np.cos(x_eq[4] + u_eq[2])
        ) / self.mass
        a34 = (
            u_eq[1] * np.cos(x_eq[4]) - u_eq[0] * np.sin(x_eq[4] + u_eq[2])
        ) / self.mass

        # linearized input dynamics
        b20 = -np.sin(x_eq[4] + u_eq[2]) / self.mass
        b21 = np.cos(x_eq[4]) / self.mass
        b22 = -u_eq[0] * np.cos(x_eq[4] + u_eq[2]) / self.mass

        b30 = np.cos(x_eq[4] + u_eq[2]) / self.mass
        b31 = np.sin(x_eq[4]) / self.mass
        b32 = -u_eq[0] * np.sin(x_eq[4] + u_eq[2]) / self.mass

        b50 = -self.l1 * np.sin(u_eq[2]) / self.inertia
        b51 = -self.l2 / self.inertia
        b52 = -self.l1 * u_eq[0] * np.cos(u_eq[2]) / self.inertia

        # A matrix
        A = np.zeros((self.state_shape, self.state_shape))
        A[0, 2] = 1
        A[1, 3] = 1
        A[2, 4] = a24
        A[3, 4] = a34
        A[4, 5] = 1

        # B matrix
        B = np.zeros((self.state_shape, self.action_shape))
        B[2, 0] = b20
        B[2, 1] = b21
        B[2, 2] = b22

        B[3, 0] = b30
        B[3, 1] = b31
        B[3, 2] = b32

        B[5, 0] = b50
        B[5, 1] = b51
        B[5, 2] = b52

        # we normalize the applied actions within the allowable input range
        normalization_u = np.diag([self.main_engine_thrust, self.side_engine_thrust, self.max_nozzle_angle])

        B = B @ normalization_u
        
        if discrete:
            # integrate matrix exponential, multiply with B
            Ad_int, _ = scipy.integrate.quad_vec(
                lambda tau: scipy.linalg.expm(A * tau), 0, self.Ts
            )
            B = Ad_int @ B

            # exact discretization using matrix exponential
            A = scipy.linalg.expm(A * self.Ts)

        return A, B
    
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