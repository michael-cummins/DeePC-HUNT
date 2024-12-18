from .utils import block_hankel, block_hankel_torch
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from cvxpylayers.torch import CvxpyLayer
import numpy as np
import cvxpy as cp
import time
from typing import Tuple

class DeePC(nn.Module):

    """
    Differentiable DeePC Module
    """

    def __init__(self, ud: np.ndarray, yd: np.ndarray, 
                 y_constraints: Tuple[np.ndarray, np.ndarray], u_constraints: Tuple[np.ndarray, np.ndarray], 
                 N: int, Tini: int, p: int, m: int, device : str,
                 stochastic_y=False, stochastic_u=False, linear=True, n_batch=1,
                 q=None, r=None, lam_y=None, lam_g1=None, lam_g2=None, lam_u=None):
        super().__init__()

        """
        Initialise differentiable DeePC
        args:
            - ud : time series vector of input signals - always pass as shape (T, m)
            - yd : time series vector of output signals 
            - y_constraints : State-wise Constraints on output signal
            - u_constraints : State-wise Constraints on input signal
            - N : Future Time horizon
            - Tini : Initial time horizon
            - p : Dimension of output signal
            - m : Dimension of input signal

            - stochastic : Set true if noise if output signals contain noise
            - linear : Set true if input and putput signals are collected from a linear system

            - u_constraints : Tuple of (upper constraints, lower constraints) with shape (2, N*m)
            - y_constraints : Tuple of (upper constraints, lower constraints) with shape (2, N*p)

            - q : vector of diagonal elemetns of Q,
                if passed as none -> randomly initialise as torch parameter in R^p
            - r : vector of diagonal elemetns of R,
                if passed as none -> randomly initialise as torch parameter in R^m
            - lam_y : regularization paramter for sig_y 
                    -> if left as none, randomly initialise as torch parameter 
            - lam_g1 : regularization paramter for sum_squares regularization on g 
                    -> if left as none, randomly initialise as torch parameter 
            - lam_g2 : regularization paramter for norm1 regularization on g 
                    -> if left as none, randomly initialise as torch parameter 
        """
        
        self.T = ud.shape[0]
        self.ud = ud
        self.yd = yd
        self.Tini = Tini
        self.N = N
        self.p = p
        self.m = m
        self.y_lower = y_constraints[0]
        self.y_upper = y_constraints[1]
        self.u_lower= u_constraints[0]
        self.u_upper = u_constraints[1]
        self.stochastic_y = stochastic_y
        self.stochastic_u = stochastic_u
        self.device = device # TODO: Shouldn't have to do this
        self.linear = linear
        self.n_batch = n_batch
        self.lam_g1 = lam_g1
        self.lam_g2 = lam_g2
        self.lam_u = lam_u
        self.lam_y = lam_y

        # Initialise torch parameters
        if isinstance(q, torch.Tensor):
            self.q = q.to(self.device)
        else: 
            self.q = Parameter(torch.randn(size=(self.p,))*0.01 + 100)
        
        if isinstance(r, torch.Tensor):
            self.r = r.to(self.device)
        else : 
            self.r = Parameter(torch.randn(size=(self.m,))*0.001 + 0.01)

        if stochastic_y:
            if isinstance(lam_y, torch.Tensor):
                self.lam_y = lam_y.to(self.device)
            else:
                self.lam_y = Parameter(torch.randn((1,))*0.001 + 200)
       
        if stochastic_u:
            if isinstance(lam_u, torch.Tensor):
                self.lam_u = lam_u.to(self.device)
            else:
                self.lam_u = Parameter(torch.randn((1,))*0.01 + 200)

        if not linear:
            if isinstance(lam_g1, torch.Tensor):
                self.lam_g1 = lam_g1.to(self.device)
            else:
                self.lam_g1 = Parameter(torch.randn((1,))*0.0001 + 200)
            if isinstance(lam_g2, torch.Tensor):
                self.lam_g2 = lam_g2.to(self.device)
            else:
                self.lam_g2 = Parameter(torch.randn((1,))*0.001 + 200)

        # Check for full row rank
        H = block_hankel(w=ud.reshape((m*self.T,)), L=Tini+N+p, d=m)
        rank = np.linalg.matrix_rank(H)
        if rank != H.shape[0]:
            raise ValueError('Data is not persistently exciting')
        
        # Construct data matrices
        U = block_hankel(w=ud.reshape((m*self.T,)), L=Tini+N, d=m)
        Y = block_hankel(w=yd.reshape((p*self.T,)), L=Tini+N, d=p)
        self.Up = U[0:m*Tini,:]
        self.Yp = Y[0:p*Tini,:]
        self.Uf = U[Tini*m:,:]
        self.Yf = Y[Tini*p:,:]

        # Initialise Optimisation variables
        g = cp.Variable(self.T-self.Tini-self.N+1)
        self.y = cp.Variable(N*p)
        ey = cp.Variable(N*p)
        eu = cp.Variable(N*m)
        self.u = cp.Variable(N*m)
        sig_y = cp.Variable(self.Tini*self.p) 
        sig_u = cp.Variable(self.Tini*self.m) 

        # Constant for sum_squares regularization on G
        PI = np.vstack([self.Up, self.Yp, self.Uf])
        PI = np.linalg.pinv(PI)@PI
        I = np.eye(PI.shape[0])

        # Initalise optimization parameters and cost
        l_g1, l_g2 = cp.Parameter(shape=(1,), nonneg=True), cp.Parameter(shape=(1,), nonneg=True)
        l_y = cp.Parameter(shape=(1,), nonneg=True)
        l_u = cp.Parameter(shape=(1,), nonneg=True)
        Q_block_sqrt, R_block_sqrt = cp.Parameter((p*N,p*N)), cp.Parameter((m*N,m*N))
        yref = cp.Parameter((N*p,))
        uref = cp.Parameter((N*m,))
        
        u_ini, y_ini = cp.Parameter(Tini*m), cp.Parameter(Tini*p)
        cost = cp.sum_squares(cp.psd_wrap(Q_block_sqrt) @ ey) + cp.sum_squares(cp.psd_wrap(R_block_sqrt) @ eu)
        assert cost.is_dpp()

        # Set constraints and cost function according to system (nonlinear / stochastic)
        if not linear:
            cost += cp.sum_squares((I - PI)@g)*l_g1 + cp.norm1(g)*l_g2 
            assert cost.is_dpp()

        cost += cp.norm1(sig_y)*l_y if self.stochastic_y else 0
        cost += cp.norm1(sig_u)*l_u if self.stochastic_u else 0
        assert cost.is_dpp()

        constraints = [
            ey == self.y - yref,  # necessary for paramaterized programming
            eu == self.u - uref,  # necessary for paramaterized programming
            self.Uf@g == self.u,
            self.Yf@g == self.y,
            self.u <= self.u_upper, self.u >= self.u_lower,
            self.y <= self.y_upper, self.y >= self.y_lower
        ]
        
        constraints.append(self.Up@g == u_ini + sig_u) if self.stochastic_u else constraints.append(self.Up@g == u_ini)
        constraints.append(self.Yp@g == y_ini + sig_y) if self.stochastic_y else constraints.append(self.Yp@g == y_ini)
        
        # Initialise optimization problem
        problem = cp.Problem(cp.Minimize(cost), constraints)
        assert problem.is_dcp()
        assert problem.is_dpp()

        variables = [self.u, self.y, g, ey, eu]
        params = [Q_block_sqrt, R_block_sqrt, u_ini, y_ini, yref, uref]
        
        if not linear:
            params.append(l_g1)
            params.append(l_g2)
        
        if stochastic_y:
            variables.append(sig_y)
            params.append(l_y)
        
        if stochastic_u:
            variables.append(sig_u)
            params.append(l_u)

        self.QP_layer = CvxpyLayer(problem=problem, parameters=params, variables=variables)
    
    def forward(self, yref: torch.Tensor, uref: torch.Tensor, u_ini: torch.Tensor, y_ini: torch.Tensor) -> list[torch.Tensor]:

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

        # Construct Q and R matrices 
        if u_ini.ndim > 1 or y_ini.ndim > 1:
            Q = torch.diag(torch.kron(torch.ones(self.N).to(self.device), torch.sqrt(self.q))).repeat(self.n_batch, 1, 1).to(self.device)
            R = torch.diag(torch.kron(torch.ones(self.N).to(self.device), torch.sqrt(self.r))).repeat(self.n_batch, 1, 1).to(self.device)
        else :
            Q = torch.diag(torch.kron(torch.ones(self.N).to(self.device), torch.sqrt(self.q))).to(self.device)
            R = torch.diag(torch.kron(torch.ones(self.N).to(self.device), torch.sqrt(self.r))).to(self.device)

        params = [Q, R, u_ini, y_ini, yref, uref]
        
        # Add paramters and system
        if not self.linear:
            params.append(self.lam_g1.repeat(self.n_batch,1))
            params.append(self.lam_g2.repeat(self.n_batch,1))
        if self.stochastic_y:
            params.append(self.lam_y.repeat(self.n_batch,1))
        if self.stochastic_u:
            params.append(self.lam_u.repeat(self.n_batch,1))

        try:
            out = self.QP_layer(*params, solver_args={"solve_method": "Clarabel"})
        except:
            out = self.QP_layer(*params, solver_args={"solve_method": "ECOS"})
            
        input, output = out[0], out[1]
        vars = [input, output]
        
        if self.stochastic_y : vars.append(out[-2])
        if self.stochastic_u : vars.append(out[-1])

        return vars

    def get_PI(self):
        # Constant for sum_squares regularization on g
        PI = np.vstack([self.Up, self.Yp, self.Uf])
        PI = np.linalg.pinv(PI)@PI
        I = np.eye(PI.shape[0])
        return I, PI

    def initialise(self, lam_y=None, lam_u=None, lam_g1=None, lam_g2=None):
        if self.lam_g1 is not None:
            self.lam_g1.data = torch.Tensor([lam_g1]) + torch.randn((1,))*0.01
        if self.lam_g2 is not None:
            self.lam_g2.data = torch.Tensor([lam_g2]) + torch.randn((1,))*0.01
        if self.lam_y is not None:
            self.lam_y.data = torch.Tensor([lam_y]) + torch.randn((1,))*0.01
        if self.lam_u is not None:
            self.lam_u.data = torch.Tensor([lam_u]) + torch.randn((1,))*0.01


class npDeePC:

    """
    Vanilla regularized DeePC module
    """

    def __init__(self, ud: np.ndarray, yd: np.ndarray, 
                 y_constraints: Tuple[np.ndarray, np.ndarray], u_constraints: Tuple[np.ndarray, np.ndarray], 
                 N: int, Tini: int, n: int, p: int, m: int) -> None:
       
        """
        Initialise variables
        args:
            ud = Inpiut signal data
            yd = output signal data
            N = predicition horizon
            Tini = estimation horizon
            n = dimesnion of system
            p = output signla dimension
            m = input signal dimension
        """

        self.T = ud.shape[0]
        self.Tini = Tini
        self.n = n 
        self.N = N
        self.p = p
        self.m = m
        self.y_lower = y_constraints[0]
        self.y_upper = y_constraints[1]
        self.u_lower= u_constraints[0]
        self.u_upper = u_constraints[1]
        self._solver_swtich = False
        # Check for full row rank
        H = block_hankel(w=ud.reshape((m*self.T,)), L=Tini+N+n, d=m)
        rank = np.linalg.matrix_rank(H)
        if rank != H.shape[0]:
            raise ValueError('Data is not persistently exciting')
        
        # Construct data matrices
        U = block_hankel(w=ud.reshape((m*self.T,)), L=Tini+N, d=m)
        Y = block_hankel(w=yd.reshape((p*self.T,)), L=Tini+N, d=p)
        self.Up = U[0:m*Tini,:]
        self.Yp = Y[0:p*Tini,:]
        self.Uf = U[Tini*m:,:]
        self.Yf = Y[Tini*p:,:]

        # Initialise Optimisation variables and parameters
        self.u = cp.Variable(self.N*self.m)
        self.g = cp.Variable(self.T-self.Tini-self.N+1)
        self.y = cp.Variable(self.N*self.p)
        self.sig_y = cp.Variable(self.Tini*self.p)

        self.y_ref = cp.Parameter((self.N*self.p,))
        self.u_ref = cp.Parameter((self.N*self.m,))
        self.u_ini = cp.Parameter(self.Tini*self.m)
        self.y_ini = cp.Parameter(self.Tini*self.p)

        # Regularization Variables
        PI = np.vstack([self.Up, self.Yp, self.Uf])
        PI = np.linalg.pinv(PI)@PI
        I = np.eye(PI.shape[0])
        self.PI = I - PI
        
    
    def setup(self, Q : np.array, R : np.array, lam_g1=None, lam_g2=None, lam_y=None) -> None:
       
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
        
        self.cost = cp.quad_form(self.y-self.y_ref, cp.psd_wrap(self.Q)) + cp.quad_form(self.u-self.u_ref, cp.psd_wrap(self.R))


        if self.lam_y != None:
            self.cost += cp.norm(self.sig_y, 1)*self.lam_y
            self.constraints = [
                self.Up@self.g == self.u_ini,
                self.Yp@self.g == self.y_ini + self.sig_y,
                self.Uf@self.g == self.u,
                self.Yf@self.g == self.y,
                self.u <= self.u_upper, self.u >= self.u_lower,
                self.y <= self.y_upper, self.y >= self.y_lower
            ]
        else:
            self.constraints = [
                self.Up@self.g == self.u_ini,
                self.Yp@self.g == self.y_ini,
                self.Uf@self.g == self.u,
                self.Yf@self.g == self.y,
                self.u <= self.u_upper, self.u >= self.u_lower,
                self.y <= self.y_upper, self.y >= self.y_lower
            ]

        if self.lam_g1 != None:
            self.cost += cp.sum_squares(self.PI@self.g)*lam_g1 
        if self.lam_g2 != None:
            self.cost += cp.norm(self.g, 1)*lam_g2
        assert self.cost.is_dpp

        self.problem = cp.Problem(cp.Minimize(self.cost), self.constraints)
        return self

    def solve(self, y_ref, u_ref, u_ini, y_ini, verbose=False, solver=cp.MOSEK) -> np.ndarray:
        
        """
        Call once the controller is set up with relevenat parameters.
        Returns the first action of input sequence.
        args:
            solver = cvxpy solver, best is MOSEK but other good options are cp.ECOS, cp.CLARABEL and cp.OSQP
            verbose = bool for printing status of solver
        """
        if not self._solver_switch: self._solver = solver
        self.y_ref.value = y_ref
        self.u_ref.value = u_ref
        self.u_ini.value = u_ini
        self.y_ini.value = y_ini
        try:
            self.problem.solve(solver=self._solver, verbose=verbose)
        except:
            # MOSEK requires license, switch if error occurs
            # License is free for academics
            self._solver_switch = True
            self._solver = cp.ECOS
        action = self.problem.variables()[1].value[:self.m]
        obs = self.problem.variables()[0].value # For imitation loss
        return action, obs
    
class npMPC:

    """
    Vanilla Linear MPC module using NumPy and CvxPy for efficient implementation
    """

    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, N: int, 
                 u_constraints: Tuple[np.ndarray,np.ndarray], y_constraints: Tuple[np.ndarray,np.ndarray]) -> None:
        
        """
        (A,B): Linear system Matrices
        (Q,R): State and input cost matrices
        N: Predicition horizon
        u_constraints: have shape (2, dimension of input)
            - u_constraints[0] should contain lower box constraints
            - u_constraints[1] should contain upper box constraints
        y_constraints: have shape (2, dimension of state)
            - y_constraints[0] should contain lower box constraints
            - y_constraints[1] should contain upper box constraints
        """
        
        self.N = N
        self.p = B.shape[0] 
        self.N = N
        self.m = B.shape[1]
        self.y_lower = y_constraints[0]
        self.y_upper = y_constraints[1]
        self.u_lower= u_constraints[0]
        self.u_upper = u_constraints[1]
        self.A = cp.Parameter(A.shape)
        self.B = cp.Parameter(B.shape)
        self.Q = Q
        self.R = R
        self.A.value = A
        self.B.value = B
        # Initialise Optimisation variables and parameters
        self.u = cp.Variable(self.N*self.m)
        self.y = cp.Variable(self.N*self.p)
        self.y_ref = cp.Parameter((self.N*self.p,))
        self.u_ref = cp.Parameter((self.N*self.m,))
        self.y_ini = cp.Parameter(self.p)

    def setup(self):
        """
        Call once to initialise the optimisation problem.
        """

        self.Q = np.kron(np.eye(self.N), self.Q)
        self.R = np.kron(np.eye(self.N), self.R)
        self.cost = cp.quad_form(self.y-self.y_ref, cp.psd_wrap(self.Q)) + cp.quad_form(self.u-self.u_ref, cp.psd_wrap(self.R))
        
        self.constraints = [
            self.y[:self.p] == self.y_ini,
            self.u <= self.u_upper, self.u >= self.u_lower,
            self.y <= self.y_upper, self.y >= self.y_lower
        ]

        for i in range(1,self.N):
            self.constraints.append(
                self.y[self.p*i:self.p*(i+1)] == self.A@self.y[self.p*(i-1):self.p*i] + self.B@self.u[self.m*(i-1):self.m*i]
            )

        self.problem = cp.Problem(cp.Minimize(self.cost), self.constraints)
        return self
    
    def solve(self, y_ref: np.ndarray, u_ref: np.ndarray, y_ini: np.ndarray, 
              u_ini=None, verbose=False, solver=cp.OSQP) -> np.ndarray:
        """
        Call to solve the MPC problem.
        y_ref, u_ref, y_ini are instatiated as parameters of the optimisation problem. 
        They only need to be passed to the solver rather than calling setup() again.
        """
        self.y_ref.value = y_ref
        self.u_ref.value = u_ref
        self.y_ini.value = y_ini
        self.problem.solve(solver=solver, verbose=verbose)
        action = self.problem.variables()[1].value[:self.m]
        obs = self.problem.variables()[0].value # For imitation loss
        return action, obs