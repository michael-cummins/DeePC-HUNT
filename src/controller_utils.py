import numpy as np
import matplotlib.pyplot as plt
import torch

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
        return input.clamp(min=1e-4, max=1e9) # the value in iterative = 2

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()