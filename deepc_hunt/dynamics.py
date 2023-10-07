import torch
import torch.nn as nn

class Env(nn.Module):

    def __init__(self, f, exact=False, Ts=None):
        
        if not Ts and not exact:
            raise AssertionError('If not an exact discretization, then you must supply a sample time Ts')
        if exact and Ts:
            raise AssertionError('Exact discretization does not require a sample time -> Ts=None')
        
        self.f = f
        self.Ts = Ts
        self.exact = exact
    
    def forward(self, x : torch.Tensor, u : torch.Tensor) -> torch.Tensor:

        if self.exact:
            x = self.f(x, u)
        else:
            x = x + self.Ts*self.f(x, u)
        
        return x
    
def affine_dynamics(x : torch.Tensor, u : torch.Tensor) -> torch.Tensor:
    pass
    
