import torch
import torch.nn as nn

class Env(nn.Module):

    def __init__(self, discrete=False, Ts=None):
        if Ts is not None and discrete==True:
            raise AssertionError('If not discrete, then you must supply a sample time Ts')
        if discrete and Ts:
            raise AssertionError('Discrete does not require a sample time -> Ts=None')

        self.Ts = Ts
        self.discrete = discrete
        
    
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
        
        u += torch.randn(u.shape) * self.input_noise_std
        
        z = x@self.A + u@self.B
        z += self.c if self.c is not None else 0

        if x_dim == 1:
            z = z.squeeze(0)

        return z + torch.randn(z.shape).to(z.device) * self.obs_noise_std