import numpy as np
import math
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import control
import control.matlab
import qpsolvers


class Model:

    kp = -np.array([1, 1, 1])
    ki = np.array([0,0,0])
    x0 = np.array([0,0,0])
    # xd = np.array([0,0,10*math.pi/180])
    xd = 10*math.pi/180
    dt = 0.1
    tt = np.array([0, dt])
    T = 10
    TT = np.arange(0,T+dt,dt)
    u0 = 0
    xx = np.array(x0)
    uu = np.array([0])
    ei = np.array([0,0,0])

    x_prev = x0
    u_prev = u0

    kplace = []
    klqr = []
    kdlqr = np.array([-0.3510, 150.5355, 264.5792])
    
    A = np.array([[-0.3176, 0.852, 0],
                 [-0.0102, -0.1383, 0],
                 [0, 1, 0]])
    B = np.vstack((-0.005,
                 -0.0217,
                 0))
    # C = np.diag([1,1,1])
    C = np.array([[0,0,1]])
    P = 10
    r_line = np.tile(xd, P)
    Ad = np.array([[0.9687, 0.0833, 0],
                   [-0.0010, 0.9862, 0],
                   [-0.0001, 0.0993, 1.0000]])
    Bd = np.vstack((-0.0006,
                   -0.0022,
                   -0.0001))
    Q0_lqr = np.diag(np.array([1, 1, 1e3]))
    R0_lqr = 1e-2
    # Q0_mpc = np.diag(np.array([1, 1, 1e3]))
    Q0_mpc = 1e3
    R0_mpc = 1e-2
    desired_roots = np.array([-1,-2,-3])
    H = []
    M = []
    L = []
    Q_hor = []
    Au = []
    bu = []
    Kmpc = []
    Tmpc = []

    def make_kplace(self):
        self.kplace = -control.matlab.place(self.A,self.B,self.desired_roots)
        print(f"kplace: {self.kplace}")
        w,v = np.linalg.eig(self.A + self.B @ self.kplace)
        print(f"place eig: {w}")

    def make_klqr(self):
        Q0 = self.Q0_lqr
        R0 = self.R0_lqr
        self.klqr = -control.lqr(self.A,self.B,Q0,R0)[0]
        print(f"klqr: {self.klqr}")
        w,v = np.linalg.eig(self.A + self.B @ self.klqr)
        print(f"lqr eig: {w}")   

    def make_kdlqr(self):
        pass

    def make_kt_mpc_lin(self, astatic=True):
        P = self.P
        C = self.C
        (r, n) = C.shape
        A = self.Ad
        B = self.Bd
        if not astatic:
            A_line = A
            B_line = B
            C_line = C
        else:
            A_line = np.block([
                [A, np.zeros((n, r))],
                [C @ A, np.eye(r)]
            ])
            B_line = np.vstack((B, C @ B))
            C_line = np.hstack((np.zeros((r, n)), np.eye(r)))
        Q0 = self.Q0_mpc
        R0 = self.R0_mpc
        Q = np.kron(np.eye(P), Q0)
        R = np.kron(np.eye(P), R0)
        L = C_line @ A_line
        M0 = C_line @ B_line
        (m,n) = M0.shape
        
        for i in range(2, P+1):
            L = np.vstack((L, C_line @ np.linalg.matrix_power(A_line, i)))
            M0 = np.vstack((M0, C_line @ np.linalg.matrix_power(A_line, i-1) @ B_line))
        
        M = M0
        for i in range(1,P):
           M = np.concatenate((M, np.vstack((np.zeros((i*m,n)), M0[0:m*(P-i),:]))),axis=1)
        
        H = M.T @ Q @ M + R
        f = M.T @ Q
        K = -np.linalg.inv(H) @ f @ L 
        T = np.linalg.inv(H) @ f
        (n,m) = B.shape

        self.H = H
        self.M = M
        self.L = L
        self.Q_hor = Q
        self.Kmpc = K[0:m,:]
        self.Tmpc = T[0:m,:]
    

    def du(self, in_,t,ep):
        return ep
    
    def place(self, x):
        ep = x-self.xd
        u = self.kplace @ ep
        return u
    
    def lqr(self, x):
        ep = x-self.xd
        u = self.klqr @ ep
        return u
    
    def dlqr(self, x):
        ep = x-self.xd
        u = self.kdlqr @ ep
        return u
    
    def pid(self, x):
        ep = x-self.xd
        self.ei = odeint(self.du,self.ei,self.tt,args=(ep,))[-1]
        u = self.kp @ ep + self.ki @ self.ei
        return u

    def mpc(self, x):
        u = self.Kmpc @ x + self.Tmpc @ self.r_line
        return u

    def mpc_astatic(self, x):
        y = self.C @ x
        # print(y)
        x_delta = x - self.x_prev
        x_line = np.hstack((x_delta, y))
        u_delta = self.Kmpc @ x_line + self.Tmpc @ self.r_line
        #print(u_delta)
        u = self.u_prev + u_delta
        self.x_prev = x
        self.u_prev = u
        return u

    def mpc_constrained(self, x):
        rho = 1e1
        H = np.block([[self.H, np.zeros((self.P,1))],
                      [np.zeros((1,self.P)), rho]])
        f = self.M.T @ self.Q_hor @ self.L @ x - self.M.T @ self.Q_hor @ self.r_line
        f = np.block([f,0])
        # self.Au = np.block([[np.eye(self.P)],[-np.eye(self.P)]])
        # self.bu = np.block([np.tile(1.0, self.P), np.tile(1.0, self.P)])
        Ay = np.block([[-self.M, -np.ones((self.P,1))],
                       [np.zeros((1, self.P)), -1]])
        by = np.block([np.tile(-1,self.P), 0])
        bdu = np.block([[self.L],
                        [np.zeros((1,3))]])
        # u = qpsolvers.solve_qp(self.H, f, self.Au, self.bu)
        u = qpsolvers.solve_qp(H, f, Ay, by+bdu@x)
        return u[0]

    def ctrl(self, x):
        return self.mpc_constrained(x)

    def rp(self, in_,t,u):
        # print(in_[1])
        x = in_.reshape((3,1))
        
        if (u.shape == (1,)):
            u = u[0]
        if (u.shape == (1,1)):
            u = u[0,0]

        aa = self.A @ x
        bb = self.B * u
        # print(aa)
        dx = aa + bb + 0*np.array([[0, 0.1, 0]]).T
        
        return dx[:,0]

    def state(self,u,x0):
        x = odeint(self.rp,x0,self.tt,args=(u,))[-1,:]
        return x
    
    def step(self,x0,u0):
        u = self.ctrl(x0)
        x = self.state(u,x0)
        return (x,u)

    def main_cycle(self):
        x = self.x0
        u = self.u0
        for t in self.TT[1:]:
            (x,u) = self.step(x,u)
            self.xx = np.vstack((self.xx,x))
            self.uu = np.vstack((self.uu,u))
            
    def linearize(self):
        self.make_kplace()
        self.make_klqr()
        self.make_kdlqr()
        self.make_kt_mpc_lin(astatic=False)
        '''
        p = 1*np.array([-1,-2,-3])
        self.kplace = -control.matlab.place(self.A,self.B,p)
        print(f"kplace: {self.kplace}")
        w,v = np.linalg.eig(self.A + self.B @ self.kplace)
        print(f"place eig: {w}")
        
        Q0 = self.Q0_lqr
        R0 = self.R0_lqr
        self.klqr,S,E = control.lqr(self.A,self.B,Q0,R0)
        self.klqr = -self.klqr
        print(f"klqr: {self.klqr}")
        w,v = np.linalg.eig(self.A + self.B @ self.klqr)
        print(f"lqr eig: {w}")   
        Q0 = self.Q0_mpc
        R0 = self.R0_mpc
        Kmpc, Tmpc = self.kt_mpc_lin(Q0, R0)
        print(f'Kmpc: {Kmpc}, Tmpc: {Tmpc}')
        
        ''' 
        pass

    def __init__(self):
        self.linearize()
        pass


#xmin = qpsolvers.solve_qp(np.array([[2.0]]), np.array([1.0]))
#print(f'xmin: {xmin}')

m = Model()
m.main_cycle()
m.xx *= 180/math.pi

plt.figure(1)
plt.subplot(3,1,1)
plt.plot(m.TT,m.xx[:,0],label='w')
plt.legend()
plt.grid()

plt.subplot(3,1,2)
plt.plot(m.TT,m.xx[:,1],label='beta')
plt.legend()
plt.grid()

plt.subplot(3,1,3)
plt.plot(m.TT,m.xx[:,2],label='phi')
plt.legend()
plt.grid()

plt.figure(2)
plt.plot(m.TT,m.uu,label='ctrl')
plt.legend()
plt.grid()

plt.show()
