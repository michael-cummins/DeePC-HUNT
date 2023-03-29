from controller_utils import block_hankel
import numpy as np
import cvxpy as cp


class DeePC:

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
        self.cost = cp.quad_form(self.y-ref,self.Q) + cp.quad_form(self.u,self.R)

        if self.lam_y != None:
            self.cost += cp.norm1(self.sig_y)*self.lam_y
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
            self.cost += cp.sum_squares(self.PI@self.g)*lam_g1 + cp.norm1(self.g)*lam_g1
            

    def solve(self, verbose=False, solver=cp.OSQP, max_iter=10000) -> np.array:
        
        """
        Call once the controller is set up with relevenat parameters.
        Returns the first action of input sequence.
        args:
            solver = cvxpy solver, usually use OSQP or ECOS
            verbose = bool for printing status of solver
        """
       
        prob = cp.Problem(cp.Minimize(self.cost), self.constraints)
        prob.solve(solver=solver, verbose=verbose, max_iter=max_iter)
        action = prob.variables()[1].value[:self.m]
        return action


