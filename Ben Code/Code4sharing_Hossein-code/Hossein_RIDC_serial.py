'''
This code uses Tianming's scripts & RIDC algorithm (CHRISTLIEB et al. 2010)
This is a serial version and for simplicity the OOP are removed
Modified and personalised by Hossein Kafiabad
Modified and personalised again by Ben Cookman
'''

# import time
import numpy as np
from scipy.interpolate import lagrange
from matplotlib import pyplot as plt
from Hossein_RIDC_parallable import RIDCsolverP

def QuadratureMatrix(M):
    """ Forming the quadraure matrix S[m,i] """
    S = np.zeros([M, M + 1])
    for m in range(M):  # Calculate qudrature weights
        for i in range(M + 1):
            x = np.arange(M + 1)  # Construct a polynomial
            y = np.zeros(M + 1)   # which equals to 1 at i, 0 at other points
            y[i] = 1
            p = lagrange(x, y)
            para = np.array(p)  # Compute its integral
            P = np.zeros(M + 2)
            for k in range(M + 1):
                P[k] = para[k]/(M + 1-k)
            P = np.poly1d(P)
            S[m, i] = P(m + 1) - P(m)
    return S


def RIDCsolverS(S, ff, T, y0, N, M, K):
    '''
    Inputs:
    ff: the RHS of the system of ODEs y'=f(t,y)
    T:  integration interval[0,T]
    y0: initial condition
    N:  number of nodes
    M:  number of correction loops
    K:  length of grouped intervals for RIDC method, K > M

    Output:
    y: as function of time
    '''
    J = int(N/K)                # number of invervals in each group
    h = float(T)/N              # time step
    d = len(y0)                 # number of equations in ODE (aka degree of freedom, dimension of space)
    yy = np.zeros([d, N+1])     # storing the final answer in yy
    t = np.arange(0, T+h, h)    # time vector
    yy[:, 0] = y0               # putting the initial condition in y

    # loop over each group of intervals j
    for j in range(J):
        y1 = np.zeros([d, K+1])
        y1[:, 0] = yy[:, j*K]   # predictor starts w last point in j-1 interval
        # Predictor loop using forward Euler method
        for m in range(K):
            # t[m+1] = (j*K+m+1) * h
            y1[:, m+1] = y1[:, m] + h*ff(t[j*K+m], y1[:, m])
        # Corrector loops using Lagrange polynomials
        for l in range(1, M+1):
            y2 = np.zeros([d, K+1])
            y2[:, 0] = y1[:, 0]
            for m in range(M):
                y2[:, m+1] = y2[:, m] + \
                    h*(ff(t[j*K+m], y2[:, m])-ff(t[j*K+m], y1[:, m])) + \
                    h*sum([S[m, i]*ff(t[j*K+i], y1[:, i]) for i in range(M+1)])
            for m in range(M, K):
                y2[:, m+1] = y2[:, m] + \
                    h*(ff(t[j*K+m], y2[:, m])-ff(t[j*K+m], y1[:, m])) + \
                    h*sum([S[M-1, i]*ff(t[j*K+m-M+i+1], y1[:, m-M+i+1]) for i in range(M+1)])
            y1 = y2
        yy[:, j*K+1:(j+1)*K+1] = y2[:, 1:]
    return t, yy


if __name__=="__main__":
    # Set up the test system and its parameters
    # def func(t, y):
    #     return (y - 2*t*y*y)/(1 + t)
    # def y_exact(t):
    #     return (1 + t)/(t*t + 1/y0[0])
    # y0 = np.array([0.4])
    # T = 1.0
    def func(t, y):
        return 4*t*np.sqrt(y)
    def y_exact(t):
        return (1 + t*t)*(1 + t*t)
    y0 = np.array([1.0])
    T = 5.0

    p = 6
    K = 100
    N = K*np.arange(1, 19)
    M = p - 1

    # Run approximations
    S = QuadratureMatrix(M)
    err_array = []
    for N_value in N:
        _, yy = RIDCsolverS(S, func, T, y0, N_value, M, K)
        y_T = y_exact(T)
        err = abs((yy[0, -1] - y_T)/y_T)    # Relative global error
        err_array.append(err)

    # Make convergence plots
    dt = T/N
    fig, ax = plt.subplots(1, 1)
    ax.plot(dt, err_array, color="b", marker="o", label="Approximate solution")
    for l in range(1, p + 1):
        ax.plot(dt, dt**l, linestyle="dashed", label=f"$1\\cdot(\\Delta t)^{l}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    plt.show()



