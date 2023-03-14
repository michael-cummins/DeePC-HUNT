from deepc_utils import block_hankel
import numpy as np
import cvxpy as cp


class DeePC:

    def __init__(self, ud: np.array, yd: np.array, y_constraints: np.array, u_constraints: np.array, 
                 N: int, Tini: int, n: int, T: int, p: int, m: int,
                 Q: np.array, R: np.array) -> None:
        
        self.T = T
        self.Tini = Tini
        self.n = n 
        self.N = N
        self.p = p
        self.m = m
        self.y_constraints = y_constraints
        self.u_constraints = u_constraints
        self.Q = np.kron(np.eye(N), Q)
        self.R = np.kron(np.eye(N), R)

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
        
    
    def setup(self, ref: np.array, u_ini: np.array, y_ini: np.array, lam_g=None, lam_y=None) -> None:
        self.lam_y = lam_y
        self.lam_g = lam_g
        
        self.constraints = [
            self.Up@self.g == u_ini,
            None,
            self.Uf@self.g == self.u,
            self.Yf@self.g == self.y,
            cp.abs(self.u) <= self.u_constraints,
            cp.abs(self.y) <= self.y_constraints
        ]

        self.cost = cp.quad_form(self.y-ref,self.Q) + cp.quad_form(self.u,self.R)
        if self.lam_y != None:
            self.cost += cp.sum_squares(self.sig_y)*self.lam_y
            self.constraints[1] = self.Yp@self.g == y_ini + self.sig_y
        else:
            self.constraints[1] = self.Yp@self.g == y_ini


    def solve(self) -> np.array:
        prob = cp.Problem(cp.Minimize(self.cost), self.constraints)
        prob.solve(solver=cp.OSQP, verbose=False)
        action = prob.variables()[1].value[:self.m]
        return action


        
