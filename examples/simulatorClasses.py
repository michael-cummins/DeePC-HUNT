
import numpy as np
import scipy as sp

from scipy.integrate import solve_ivp
from scipy.signal import cont2discrete
import control

import copy


'''
NOTE FOR MICHAEL: 
    Hopefully this code helps get you going with the double integrator, rocket, and perhaps crazy fly. It simulats either linear or nonlinear differential equations. 
    You can ignore the "LTV" variables in this code. Those variables are designated at time-varying variables for some of my own research, but that is not pertinent for DeePC-HUNT
'''

class env:

    def __init__(self, dt, printProgress):
        # how to use inherited properties
        # https://towardsdatascience.com/master-class-inheritance-in-python-c46bfda63374
        self.dt = dt
        self.printProgress = printProgress


    def step_measurement(self, u):
        '''
        System:
        x[k+1] = A[k]x[k] + B[k]u[k]
        y[k+1] = C[k+1]x[k+1]
        Input
        - u[k]
        Output
        - y[k+1]

        This is the standard form for output feedback control.
        Does not allow for a D to be used.
        '''

        if self.linear:
            self.x = self.A@self.x + self.B@u
            if self.D is None:
                y = self.C@self.x
            else:
                raise Exception('Cant do step after if there is a D because u[k+1] is not known yet. If u[k+1] is a function of y[k+1] then including a D will not work.')
        else:
            if self.useBasicEuler:
                self.u = u # self.systemDynamicEquations uses self.u
                self.x = self.x + self.dt*self.systemDynamicEquations(self.x, u)
            else:
                def f(t, y): return self.systemDynamicEquations(y, u) # t is not used. u for f() is set to be what it is when f() is defined.
                sol = solve_ivp(f, [0, 0+self.dt], self.x) # LATER : create a timer for this
                if not sol.success:
                    raise Exception('the integrator/system solver failed, perhaps because the system went unstable')
                self.x = sol.y[:,-1].flatten() # sol.y is the name of the solve_ivp state
            if self.C is not None:
                if self.D is None:
                    y = self.C@self.x
                else:
                    raise Exception('Cant do step after if there is a D because u[k+1] is not known yet. If u[k+1] is a function of y[k+1] then including a D will not work.')
            else:
                raise Exception('Havent implemented this yet')

        return y, self.x

    def getMeas(self, u=None):
        '''
        Gives the output at the given state with the given input
        '''
        if self.D is None or u is None:
            y = self.C@self.x
        else:
            y = self.C@self.x + self.D@u
        return y


    def reset(self, x_ini=None):
        if len(x_ini)==0 and x_ini == None:
            self.x = self.x_ini
        else:
            self.x = x_ini
        return



    def getLQRmat(self, A_d, B_d, Q, R):

        K, _, _ = control.dlqr(A_d, B_d, Q, R)

        if self.printProgress:
            print(f'K_lqr: {K}')

        return K
    
    


class env_1D_doubleIntegrator(env):
    '''
    This is a standard double-integrator system.
    Notes:
    - The discretization of continuous time double integrator systems is described in:
    https://jckantor.github.io/ND-Pyomo-Cookbook/notebooks/02.06-Model-Predictive-Control-of-a-Double-Integrator.html
    '''
    def __init__(self, mass=1, p=1, dt=1, x_ini=None, printProgress=False):
        '''
        mass : mass
        p=1 : number of inputs
        dt=1 : timestep length
        x_ini : initial condition (default to zeros)
        u : force
        x1 : displacement
        x2 : velocity

        Notes
        - The LTV parameter has to be a list in order to bind it to the object the objects inside self.LTVparameters_listOfLists
            - lists are mutable in python, most other objects are not
            - this allows lists to behave like pointers in c (bug or feature??)
            - Resources that explain this
                https://realpython.com/pointers-in-python/
                https://stackoverflow.com/questions/3106689/pointers-in-python
        '''

        super().__init__(dt, printProgress)

        self.linear = True

        n=2
        self.mass_list = [mass] #LTV parameters have to be lists so that they are mutable/can use generic "self.LTVparameters_listOfLists" as a pointer
        self.p = p
        self.dt = dt
        self.setSystemMatricesFromParameters()

        self.LTVparameters_listOfLists = [self.mass_list] #list of lists to allow multiple time-varying parameters
        self.LTVparameters_listOfLists_original = copy.deepcopy(self.LTVparameters_listOfLists)

        if x_ini is None:
            self.x = np.zeros(n)
            self.x_ini = np.zeros(n)
        else:
            assert(len(x_ini)==n)
            self.x = copy.deepcopy(x_ini)
            self.x_ini = x_ini

    def getSystemMatrices(self):
        A = np.array([[1, self.dt],
                           [0, 1]])
        B = np.array([[0],
                      [self.dt/self.mass_list[0]]])
        # B = np.array([[(self.dt/self.mass_list[0])**2/2],
        #                    [self.dt/self.mass_list[0]]])
        # A = np.array([[1, 1],
        #                    [0, 1]])
        # B = np.array([[1/2],
        #                    [1]])
        if self.p == 1:
            C = np.array([[1, 0]])
        elif self.p == 2:
            C = np.eye(2)
        else:
            raise Exception('can only be one or two outputs')
        D = None
        return A, B, C, D




class env_3D_doubleIntegrator(env):
    '''
    This is a standard double-integrator system.
    Notes:
    - The discretization of continuous time double integrator systems is described in:
    https://jckantor.github.io/ND-Pyomo-Cookbook/notebooks/02.06-Model-Predictive-Control-of-a-Double-Integrator.html
    '''
    def __init__(self, mass=1, p=3, dt=1, x_ini=None, printProgress=False):
        '''
        mass=1 : mass
        p=3 : number of inputs
        dt=1 : timestep length
        x_ini : initial condition (default to zeros)
        u : force in all 3 directions
        x1 : displacement in all 3 directions
        x2 : velocity in all 3 directions

        Notes
        - The LTV parameter has to be a list in order to bind it to the object the objects inside self.LTVparameters_listOfLists
            - lists are mutable in python, most other objects are not
            - this allows lists to behave like pointers in c (bug or feature??)
            - Resources that explain this
                https://realpython.com/pointers-in-python/
                https://stackoverflow.com/questions/3106689/pointers-in-python
        '''

        super().__init__(dt, printProgress)

        self.linear = True

        n=6
        self.mass_list = [mass] #LTV parameters have to be lists so that they are mutable/can use generic "self.LTVparameters_listOfLists" as a pointer
        self.p = p
        self.dt = dt
        self.setSystemMatricesFromParameters()

        self.LTVparameters_listOfLists = [self.mass_list] #list of lists to allow multiple time-varying parameters
        self.LTVparameters_listOfLists_original = copy.deepcopy(self.LTVparameters_listOfLists)

        if x_ini is None:
            self.x = np.zeros(n)
            self.x_ini = np.zeros(n)
        else:
            assert(len(x_ini)==n)
            self.x = copy.deepcopy(x_ini)
            self.x_ini = x_ini

    def getSystemMatrices(self):
        Ablock = np.array([[1, self.dt],
                           [0, 1]])
        Bblock = np.array([[0],
                      [self.dt/self.mass_list[0]]])
        if self.p == 3:
            Cblock = np.array([[1, 0]])
        elif self.p == 6:
            Cblock = np.eye(2)
        else:
            raise Exception('can only be 3 or 6 outputs')
        A = np.kron(np.eye(3),Ablock)
        B = np.kron(np.eye(3),Bblock)
        C = np.kron(np.eye(3),Cblock)
        D = None

        return A, B, C, D




class env_invertedPendulum(env):
    '''
    Model from:
    http://jstc.de/blog/uploads/Control-of-an-Inverted-Pendulum.pdf
    https://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=SystemModeling

    nonlinear system
    x[0] : Theta
    x[1] : Theta_dot
    x[2] : position
    x[3] : pos_dot

    u is the lateral force on the cart in the x-direction

    sample parameters
    cartMass = 0.5 kg
    bobMass = 0.2 kg
    poleLength = 0.3 m #length of the
    frictCoeff = 0.1 N/m/s #coeff of friction

    '''
    def __init__(self,  bobMass=0.1, cartMass=1, poleLength=0.3, frictCoeff=0.1, p=2, dt =1, x_ini=None, printProgress=False):
        '''
        LTVparam: TODO
        x_ini : initial condition (default to zeros)

        LTVparam is the bobMass

        Notes
        - The LTV parameter has to be a list in order to bind it to the object the objects inside self.LTVparameters_listOfLists
            - lists are mutable in python, most other objects are not
            - this allows lists to behave like pointers in c (bug or feature??)
            - Resources that explain this
                https://realpython.com/pointers-in-python/
                https://stackoverflow.com/questions/3106689/pointers-in-python
        '''

        super().__init__(dt, printProgress)

        self.linear = False
        self.useBasicEuler = False
        if self.useBasicEuler and dt >= .1:
            raise Exception('Cant use basic euler with a dt this big. Probably best to avoid euler altogether.')
        self.u = None
        

        n = 4
        self.bobMass_list = [bobMass] #LTV parameters have to be lists so that they are mutable/can use generic "self.LTVparameters_listOfLists" as a pointer
        self.cartMass = cartMass # not time-varying
        self.poleLength = poleLength # not time-varying
        self.frictCoeff = frictCoeff # not time-varying

        self.LTVparameters_listOfLists = [self.bobMass_list] #list of lists to allow multiple time-varying parameters
        self.LTVparameters_listOfLists_original = copy.deepcopy(self.LTVparameters_listOfLists)

        self.p = p
        if p == 2:
            self.C = np.array([[1, 0, 0, 0],
                               [0, 0, 1, 0]])
        elif p == 4:
            self.C = np.eye(p)
        else:
            raise Exception('Havent implemented this yet')
        self.D = None
        
        if x_ini is None:
            self.x = np.zeros(n)
            self.x_ini = np.zeros(n)
        else:
            assert(len(x_ini)==n)
            self.x = copy.deepcopy(x_ini)
            self.x_ini = x_ini


    # def getSystemMatrices(self):
    def systemDynamicEquations(self, x, u):
        '''
        Theta = x[0]
        Theta_dot = x[1]
        pos = x[2]
        pos_dot = x[3]
        
        Notes
        - I checked the nonlinear dynamics equations (code is in matlab) and the Jacobian gives the linearization that Im using
        '''

        g = 9.8
        cartMass = self.cartMass
        bobMass = self.bobMass_list[0]
        b = self.frictCoeff
        l = self.poleLength

        dx_dt = np.array([x[1],
                          ((cartMass+bobMass)*g*np.sin(x[0]) + b*np.cos(x[0])*x[3] - bobMass*l*np.sin(x[0])*np.cos(x[0])*x[1]**2 - np.cos(x[0])*u[0])/(l*(cartMass + bobMass*(np.sin(x[0]))**2)),
                          x[3],
                          (-bobMass*g*np.sin(x[0])*np.cos(x[0]) - b*x[3] + bobMass*l*np.sin(x[0])*x[1]**2 + u[0])/(cartMass + bobMass*(np.sin(x[0]))**2)])


        # dx_dt = np.array([Theta_dot,
        #                   ((cartMass+bobMass)*g*np.sin(Theta) + b*np.cos(Theta)*pos_dot - bobMass*l*np.sin(Theta)*np.cos(Theta)*Theta_dot**2 - np.cos(Theta)*u)/(l*(cartMass + bobMass*(np.sin(Theta))**2)),
        #                   pos_dot,
        #                   (-bobMass*g*np.sin(Theta)*np.cos(Theta) - b*pos_dot + bobMass*l*np.sin(Theta)*Theta_dot**2 + u)/(cartMass + bobMass*(np.sin(Theta))**2)])

        return dx_dt


    def getDisctretizedLinearizedSystemAtVertical(self):
        '''
        http://jstc.de/blog/uploads/Control-of-an-Inverted-Pendulum.pdf

        Notes:
        - I checked the linearized continuous and discretized matrices that you get with the discretized matrices in the paper above and they match.
        '''
        g = 9.8
        cartMass = self.cartMass
        bobMass = self.bobMass_list[0]
        b = self.frictCoeff
        l = self.poleLength

        A = np.array([[0, 1, 0, 0],
                      [g*(cartMass + bobMass)/(l*cartMass), 0, 0, b/(l*cartMass)],
                      [0, 0, 0, 1],
                      [-bobMass*g/cartMass, 0, 0, -b/cartMass]])

        B = np.array([[0],
                      [-1/(l*cartMass)],
                      [0],
                      [1/cartMass]])

        C = self.C

        D = np.zeros((len(C), 1))

        d_system = cont2discrete((A, B, C, D), self.dt)
        A_d = d_system[0]
        B_d = d_system[1]
        C_d = d_system[2]

        return A_d, B_d, C_d


    def getKforInvertedPendulum(self, p, **kwargs):

        if 'useLQR' in kwargs and kwargs['useLQR']:
            if p != 4:
                raise Exception('need full state to implement LQR, havent built a state observer yet')
            K = self.getLQRmat(kwargs['A_d'], kwargs['B_d'], kwargs['LQR_Q'], kwargs['LQR_R'])
            # K_lqr = self.getLQRmat(A_d, B_d, Q, R)
            # # print(K_lqr)
            # # K_lqr[0,2] = 0
            # # K_lqr = K_lqr*10
            # # print(K_lqr)
            # # exit()
            # Acl = A_d - B_d@K_lqr
        else:
            if 'offlineDataGathering_Kp_angle' not in kwargs or kwargs['offlineDataGathering_Kp_angle'] == None:
                offlineDataGathering_Kp_angle = 0
            else:
                offlineDataGathering_Kp_angle = kwargs['offlineDataGathering_Kp_angle']
            if 'offlineDataGathering_Kd_angle' not in kwargs or kwargs['offlineDataGathering_Kd_angle'] == None:
                offlineDataGathering_Kd_angle = 0
            else:
                offlineDataGathering_Kd_angle = kwargs['offlineDataGathering_Kd_angle']
            if 'offlineDataGathering_Kp_pos' not in kwargs or kwargs['offlineDataGathering_Kp_pos'] == None:
                offlineDataGathering_Kp_pos = 0
            else:
                offlineDataGathering_Kp_pos = kwargs['offlineDataGathering_Kp_pos']
            if 'offlineDataGathering_Kd_pos' not in kwargs or kwargs['offlineDataGathering_Kd_pos'] == None:
                offlineDataGathering_Kd_pos = 0
            else:
                offlineDataGathering_Kd_pos = kwargs['offlineDataGathering_Kd_pos']
            if p == 2:
                K = np.array([[offlineDataGathering_Kp_angle, offlineDataGathering_Kp_pos]])
            elif p == 4:
                K = np.array([[offlineDataGathering_Kp_angle, offlineDataGathering_Kd_angle, offlineDataGathering_Kp_pos, offlineDataGathering_Kd_pos]])
            else:
                raise Exception('havent implemented for other ps yet')

        return K
    
    
    
    
class env_gymRocket(env):
    '''
    Model from:
    The simulator for the 2023 COCO II class project, which is based on the simulator in the masters project below
    https://project-archive.inf.ed.ac.uk/msc/20172139/msc_proj.pdf
    The nonlinear dynamics are given on page 21 of his Thesis
    
    system states
    x[0] : x (lateral position)
    x[1] : x_dot
    x[2] : z (vertical position)
    x[3] : z_dot
    x[4] : theta
    x[5] : theta_dot (rocket angle)

    u[0] : u_Fe (vertical thrust)
    u[1] : u_Fs (lateral thrust at the top of the rocket)
    u[2] : u_phi (angle of the vertical thrust nozzle)

    parameters
    m = m (25 comes from page 43 of his report)
    
    eq point (at which a vertical rocket hovers, this is where the linearization should be calced):
    x_bar = zeros (I think)
    u_bar = [mg, 0, 0]

    LTV param:
        In the simulator the mass decreases as the simulator progresses (p 43 of his report)
        Michael can you figure out how to turn this off in the gym environment (not this sim) so that the rocket mass is constant? If not, dont worry aobut it. If so, please put a note for how to do this here.
    '''
    def __init__(self,  m=25, p=6, dt =1/60, x_ini=None, printProgress=False):
        '''
        m : rocket mass. Default is 25 kg I think
        p : number of ouptuts 
        dt : length of discrete timesteps
        x_ini : initial condition (default to zeros)


        '''

        super().__init__(dt, printProgress)

        self.linear = False
        self.useBasicEuler = False
        if self.useBasicEuler and dt >= .1:
            raise Exception('Cant use basic euler with a dt this big. Probably best to avoid euler altogether.')
        self.u = None
        
        n = 6
        # set parameters
        self.m = m

        self.p = p
        if p == 6:
            self.C = np.eye(p)
        # elif p == 4:
        #     self.C = np.array([[1, 0, 1, 0, 1, 1],
        #                        [0, 0, 1, 0, 1, 1]]) # or something.. dont worry about this
        else:
            raise Exception('Havent implemented this yet')
        self.D = None
        
        if x_ini is None:
            self.x = np.zeros(n)
            self.x_ini = np.zeros(n)
        else:
            assert(len(x_ini)==n)
            self.x = copy.deepcopy(x_ini)
            self.x_ini = x_ini # remember the initial state in case you want to reset the env


    # def getSystemMatrices(self):
    def systemDynamicEquations(self, x, u):
        '''
        ssee env description above for defs of x and u
        '''
        g = 9.8
        m = self.m
        # J = [moment of inertia?]

        dx_dt = np.array([TODO: Diff Eqns here, see invertedPend env for an example])

        return dx_dt


    def getDisctretizedLinearizedSystemAtVertical(self):
        '''
        See page 28 of Rueben Ferrante's masters project:
            https://project-archive.inf.ed.ac.uk/msc/20172139/msc_proj.pdf
            
        eq point (at which a vertical rocket hovers, this is where the linearization should be calced):
        x_bar = zeros (I think)
        u_bar = [mg, 0, 0]
        '''
        g = 9.8
        m = self.m
        # other params

        A = np.array([TODO])

        B = np.array([TODO])

        C = self.C

        # D = np.zeros((len(C), m))

        d_system = cont2discrete((A, B, C, D), self.dt)
        A_d = d_system[0]
        B_d = d_system[1]
        C_d = d_system[2]

        return A_d, B_d, C_d
    
    
    
    
    
    
class env_crazyFly(env):
    '''
    Model from:
    [Ezzats Matlab code?]
    
    system states
    x[0] : ...
    ...
    
    u[0] : ...
    ...

    parameters
    ...
    
    eq point (at which a stabilized crazyFly hovers, this is where the linearization should be calced):
    x_bar = 
    u_bar = 
    '''
    def __init__(self,  params=TODO, p=TODO, dt =TODO, x_ini=None, printProgress=False):
        '''
        params : passed as args or kwargs?
        p : number of ouptuts 
        dt = length of discrete timesteps
        x_ini : initial condition (default to zeros)


        '''

        super().__init__(dt, printProgress)

        self.linear = False
        self.useBasicEuler = False
        if self.useBasicEuler and dt >= .1:
            raise Exception('Cant use basic euler with a dt this big. Probably best to avoid euler altogether.')
        self.u = None
        
        n = TODO
        # set parameters
        # self.m = m

        self.p = p
        # if p == 6:
        #     self.C = np.eye(p)
        # elif p == 4:
        #     self.C = np.array([[1, 0, 1, 0, 1, 1],
        #                        [0, 0, 1, 0, 1, 1]]) # or something.. dont worry about this
        # else:
        #     raise Exception('Havent implemented this yet')
        # self.D = None
        
        if x_ini is None:
            self.x = np.zeros(n)
            self.x_ini = np.zeros(n)
        else:
            assert(len(x_ini)==n)
            self.x = copy.deepcopy(x_ini)
            self.x_ini = x_ini # remember the initial state in case you want to reset the env


    # def getSystemMatrices(self):
    def systemDynamicEquations(self, x, u):
        '''
        see env description above for defs of x and u
        '''
        g = 9.8
        # m = self.m
        # TODO rest of params

        dx_dt = np.array([[TODO: Diff Eqns here, see invertedPend env for an example])

        return dx_dt


    def getDisctretizedLinearizedSystemAtVertical(self):
        '''
        see env description above for defs of x and u
        '''
        g = 9.8
        # m = self.m
        # other params

        A = np.array([TODO])

        B = np.array([TODO])

        C = self.C

        # D = np.zeros((len(C), m))

        d_system = cont2discrete((A, B, C, D), self.dt)
        A_d = d_system[0]
        B_d = d_system[1]
        C_d = d_system[2]

        return A_d, B_d, C_d
    
    