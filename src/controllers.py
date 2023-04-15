from controller_utils import block_hankel
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from cvxpylayers.torch import CvxpyLayer
import numpy as np
import cvxpy as cp


class DeePC:

    """
    Vanilla regularized DeePC module
    """

    def __init__(self, ud: np.array, yd: np.array, y_constraints: np.array, u_constraints: np.array, 
                 N: int, Tini: int, n: int, T: int, p: int, m: int) -> None:
       
        """
        Initialise variables
        args:
            ud = Inpiut signal data
            yd = output signal data
            N = predicition horizon
            n = dimesnion of system
            p = output signla dimension
            m = input signal dimension
        """

        self.T = T
        self.Tini = Tini
        self.n = n 
        self.N = N
        self.p = p
        self.m = m
        self.y_constraints = y_constraints
        self.u_constraints = u_constraints

        # Check for full row rank
        H = block_hankel(w=ud.reshape((m*T,)), L=Tini+N+n, d=m)
        rank = np.linalg.matrix_rank(H)
        if rank != H.shape[0]:
            raise ValueError('Data is not persistently exciting')
        
        # Construct data matrices
        U = block_hankel(w=ud.reshape((m*T,)), L=Tini+N, d=m)
        Y = block_hankel(w=yd.reshape((p*T,)), L=Tini+N, d=p)
        self.Up = U[0:m*Tini,:]
        self.Yp = Y[0:p*Tini,:]
        self.Uf = U[Tini*m:,:]
        self.Yf = Y[Tini*p:,:]

        # Initialise Optimisation variables
        self.u = cp.Variable(self.N*self.m)
        self.g = cp.Variable(self.T-self.Tini-self.N+1)
        self.y = cp.Variable(self.N*self.p)
        self.sig_y = cp.Variable(self.Tini*self.p)

        # Regularization Variables
        PI = np.vstack([self.Up, self.Yp, self.Uf])
        PI = np.linalg.pinv(PI)@PI
        I = np.eye(PI.shape[0])
        self.PI = I - PI
        
    
    def setup(self, ref: np.array, u_ini: np.array, y_ini: np.array, Q : np.array, R : np.array,
               lam_g1=None, lam_g2=None, lam_y=None) -> None:
       
        """
        Set up controller constraints and cost function.
        Also used online during sim to update u_ini, y_ini, reference and regularizers
        args:
            ref = reference signal
            u_ini = initial input trajectory
            y_ini = initial output trajectory
            lam_g1, lam_g2 = regularization params for nonlinear systems
            lam_y = regularization params for stochastic systems
        """

        self.lam_y = lam_y
        self.lam_g1 = lam_g1
        self.lam_g2 = lam_g2
        self.Q = np.kron(np.eye(self.N), Q)
        self.R = np.kron(np.eye(self.N), R)
        self.cost = cp.quad_form(self.y-ref, cp.psd_wrap(self.Q)) + cp.quad_form(self.u, cp.psd_wrap(self.R))

        if self.lam_y != None:
            self.cost += cp.norm(self.sig_y, 1)*self.lam_y
            self.constraints = [
                self.Up@self.g == u_ini,
                self.Yp@self.g == y_ini + self.sig_y,
                self.Uf@self.g == self.u,
                self.Yf@self.g == self.y,
                cp.abs(self.u) <= self.u_constraints,
                cp.abs(self.y) <= self.y_constraints
            ]
        else:
            self.constraints = [
                self.Up@self.g == u_ini,
                self.Yp@self.g == y_ini,
                self.Uf@self.g == self.u,
                self.Yf@self.g == self.y,
                cp.abs(self.u) <= self.u_constraints,
                cp.abs(self.y) <= self.y_constraints
            ]

        if self.lam_g1 != None or self.lam_g2 != None:
            self.cost += cp.sum_squares(self.PI@self.g)*lam_g1 + cp.norm(self.g, 1)*lam_g2
            

    def solve(self, verbose=False, solver=cp.MOSEK) -> np.array:
        
        """
        Call once the controller is set up with relevenat parameters.
        Returns the first action of input sequence.
        args:
            solver = cvxpy solver, usually use OSQP or ECOS
            verbose = bool for printing status of solver
        """

        prob = cp.Problem(cp.Minimize(self.cost), self.constraints)
        assert prob.is_dpp()
        assert prob.is_dcp()
        prob.solve(solver=solver, verbose=verbose)
        action = prob.variables()[1].value[:self.m]
        g = prob.variables()[2].value # For imitation loss
        return action

class Clamp(torch.autograd.Function):
    """
    https://discuss.pytorch.org/t/regarding-clamped-learnable-parameter/58474/4
    """
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0, max=100000) # the value in iterative = 2

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class DDeePC(nn.Module):

    """
    Differentiable DeePC Module
    """

    def __init__(self, ud: np.array, yd: np.array, y_constraints: np.array, u_constraints: np.array, 
                 N: int, Tini: int, T: int, p: int, m: int, n_batch: int,
                 stochastic : bool, linear : bool,
                 q=None, r=None, lam_y=None, lam_g1=None, lam_g2=None):
        super().__init__()

        """
        Initialise differentiable DeePC
        args:
            - ud : time series vector of input signals 
            - yd : time series vector of output signals 
            - y_constraints : State-wise Constraints on output signal
            - u_constraints : State-wise Constraints on input signal
            - N : Future Time horizon
            - Tini : Initial time horizon
            - T : Length of data
            - p : Dimension of output signal
            - m : Dimension of input signal

            - stochastic : Set true if noise if output signals contain noise
            - linear : Set true if input and putput signals are collected from a linear system

            - q : vector of diagonal elemetns of Q,
                if passed as none -> randomly initialise as torch parameter from N(1, 0.1) in Rp
            - r : vector of diagonal elemetns of R,
                if passed as none -> randomly initialise as torch parameter from N(1, 0.1) in Rm
            - lam_y : regularization paramter for sig_y 
                    -> if left as none, randomly initialise as torch parameter from N(1, 0.1)
            - lam_g1 : regularization paramter for sum_squares regularization on g 
                    -> if left as none, randomly initialise as torch parameter from N(1, 0.1)
            - lam_g2 : regularization paramter for norm1 regularization on g 
                    -> if left as none, randomly initialise as torch parameter from N(1, 0.1)
        """

        self.T = T
        self.Tini = Tini
        self.N = N
        self.p = p
        self.m = m
        self.y_constraints = y_constraints
        self.u_constraints = u_constraints
        self.stochastic = stochastic
        self.linear = linear
        self.n_batch = n_batch

        # Initialise torch parameters
        clamper = Clamp()
        if isinstance(q, torch.Tensor):
            self.q = q
        else : 
            self.q = Parameter(torch.randn(size=(3,))*0.1+1)
        if isinstance(r, torch.Tensor):
            self.r = r
        else : 
            self.r = Parameter(torch.randn(size=(3,))*0.1+1)

        if stochastic:
            if isinstance(lam_y, torch.Tensor):
                self.lam_y = lam_y 
            else:
                self.lam_y = Parameter(torch.randn((1,))*0.1 + 100)
        else: self.lam_y = 0 # Initialised but won't be used

        if not linear:
            if isinstance(lam_g1, torch.Tensor) and isinstance(lam_g2, torch.Tensor):
                self.lam_g1, self.lam_g2 = lam_g1, lam_g2
            else:
                self.lam_g1 = Parameter(torch.randn((1,))*0.1 + 1)
                self.lam_g2 = Parameter(torch.randn((1,))*0.1 + 1)
        else: self.lam_g1, self.lam_g2 = 0, 0 # Initialised but won't be used

        # Check for full row rank
        H = block_hankel(w=ud.reshape((m*T,)), L=Tini+N+p, d=m)
        rank = np.linalg.matrix_rank(H)
        if rank != H.shape[0]:
            raise ValueError('Data is not persistently exciting')
        
        # Construct data matrices
        U = block_hankel(w=ud.reshape((m*T,)), L=Tini+N, d=m)
        Y = block_hankel(w=yd.reshape((p*T,)), L=Tini+N, d=p)
        self.Up = U[0:m*Tini,:]
        self.Yp = Y[0:p*Tini,:]
        self.Uf = U[Tini*m:,:]
        self.Yf = Y[Tini*p:,:]

        # Initialise Optimisation variables
        g = cp.Variable(self.T-self.Tini-self.N+1)
        self.y = cp.Variable(N*p)
        e = cp.Variable(N*p)
        self.u = cp.Variable(N*m)
        sig_y = cp.Variable(self.Tini*self.p)

        # Constant for sum_squares regularization on G
        PI = np.vstack([self.Up, self.Yp, self.Uf])
        PI = np.linalg.pinv(PI)@PI
        I = np.eye(PI.shape[0])

        # Initalise optimization parameters and cost
        l_g1, l_g2 = cp.Parameter(shape=(1,), nonneg=True), cp.Parameter(shape=(1,), nonneg=True)
        l_y = cp.Parameter(shape=(1,), nonneg=True)
        Q_block_sqrt, R_block_sqrt = cp.Parameter((p*N,p*N)), cp.Parameter((m*N,m*N))
        ref = cp.Parameter((N*p,))
        u_ini, y_ini = cp.Parameter(Tini*m), cp.Parameter(Tini*p)
        cost = cp.sum_squares(cp.psd_wrap(Q_block_sqrt) @ e) + cp.sum_squares(cp.psd_wrap(R_block_sqrt) @ self.u)
        assert cost.is_dpp()

        # Set constraints and cost function according to system (nonlinear / stochastic)
        if not linear:
            cost += cp.sum_squares((I - PI)@g)*l_g1 + cp.norm1(g)*l_g2 
            assert cost.is_dpp()

        if stochastic:
            cost += cp.norm1(sig_y)*l_y
            assert cost.is_dpp()
            constraints = [
                e == self.y - ref,  # necessary for paramaterized programming
                self.Up@g == u_ini,
                self.Yp@g == y_ini + sig_y,
                self.Uf@g == self.u,
                self.Yf@g == self.y,
                cp.abs(self.u) <= self.u_constraints,
                cp.abs(self.y) <= self.y_constraints
            ]
        else:
            constraints = [
                e == self.y - ref,
                self.Up@g == u_ini,
                self.Yp@g == y_ini,
                self.Uf@g == self.u,
                self.Yf@g == self.y,
                cp.abs(self.u) <= self.u_constraints,
                cp.abs(self.y) <= self.y_constraints
            ]
        
        # Initialise optimization problem
        problem = cp.Problem(cp.Minimize(cost), constraints)
        assert problem.is_dcp()
        assert problem.is_dpp()

        variables = [g, e, self.u, self.y]
        params = [Q_block_sqrt, R_block_sqrt, u_ini, y_ini, ref]
        if not linear:
            params.append(l_g1)
            params.append(l_g2)
        if stochastic:
            variables.append(sig_y)
            params.append(l_y)
        
        self.QP_layer = CvxpyLayer(problem=problem, parameters=params, variables=variables)


    def forward(self, ref: torch.Tensor, u_ini: torch.Tensor, y_ini: torch.Tensor) -> torch.Tensor:

        """
        Forward call
        args :
            - ref : Reference trajectory 
            - u_ini : Initial input signal
            - y_ini : Initial Output signal

        Returns : 
            input : optimal input signal
            output : optimal output signal
            cost : optimal cost
        """
        clamper = Clamp()
        self.lam_y.data = clamper.apply(self.lam_y)
        # for param in self.parameters():
        #     clamper.apply(param)

        # Construct Q and R matrices 
        Q = torch.diag(torch.kron(torch.ones(self.N), torch.sqrt(self.q)))
        R = torch.diag(torch.kron(torch.ones(self.N), torch.sqrt(self.r)))
        params = [Q, R, u_ini, y_ini, ref]

        # Add paramters and system
        if not self.linear:
            params.append(self.lam_g1)
            params.append(self.lam_g2)
        if self.stochastic:
            params.append(self.lam_y)

        out = self.QP_layer(*params)
        input, output = out[2], out[3]

        traj_cost = input.T @ R @ input + (output - ref).T @ Q @ (output - ref)

        return input, output, traj_cost
