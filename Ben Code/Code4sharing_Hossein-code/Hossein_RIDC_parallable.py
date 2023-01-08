
'''
This code uses Tianming's scripts & RIDC algorithm (CHRISTLIEB et al. 2010)
This is a serial version and for simplicity the OOP are removed
Modified and personalised by Hossein Kafiabad
'''

# import threading
import time
import numpy as np
from scipy.interpolate import lagrange


def func(t, y):
    y1 = y[0]
    y2 = y[1]
    for i in range(1000):
        y1_p = -y2 + y1*(1-y1**2-y2**2)
        y2_p = y1 + 3*y2*(1-y1**2-y2**2)
    return np.array([y1_p, y2_p])


def RIDCsolverP(S, ff, T, y0, N, M, K):
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
    # define the nested functions
    def corrector(Y2, Y1vec, Tvec):
        Y2plus = Y2+h*(ff(Tvec[-2], Y2)-ff(Tvec[-2], Y1vec[:, -2])) + \
            h*sum([Svec[i]*ff(Tvec[i], Y1vec[:, i]) for i in range(M+1)])
        return Y2plus

    def predictor(Ypred, Tpred):
        return Ypred+h*ff(Tpred, Ypred)
    # number of invervals in each group
    J = int(N/K)
    # time step
    h = float(T)/N
    # Forming the quadraure matrix S[m,i]
    # number of equations in ODE (aka degree of freedom, dimension of space)
    d = len(y0)

    Svec = S[M-1, :]
    # storing the final answer in yy
    yy = np.zeros([d, N+1])
    # the time vector
    t = np.arange(0, T+h, h)
    t_ext = np.arange(0, T+h+M*h, h)
    # putting the initial condition in y
    yy[:, 0] = y0
    for j in range(J):   # loop over each group of intervals j
        # print(j)
        # ----------------------------------------------------------------------
        # Phase 1: compute to the point every threads can start simultaneously
        Ybegin = np.zeros([d, M+1, 2*M])
        # predictor starts w last point in j-1 interval
        Ybegin[:, 0, 0] = yy[:, j*K]
        # predictor loop usig forward Euler method
        for m in range(2*M-1):
            # t[m+1] = (j*K+m+1) * h
            Ybegin[:, 0, m+1] = Ybegin[:, 0, m]+h*ff(t[j*K+m], Ybegin[:, 0, m])
        # corrector loops using Lagrange polynomials
        for l in range(1, M+1):
            Ybegin[:, l, :] = np.zeros([d, 2*M])
            Ybegin[:, l, 0] = yy[:, j*K]
            for m in range(M):
                Ybegin[:, l, m+1] = Ybegin[:, l, m] + \
                    h*(ff(t[j*K+m], Ybegin[:, l, m])-ff(t[j*K+m], Ybegin[:, l-1, m])) + \
                    h*sum([S[m, i]*ff(t[j*K+i], Ybegin[:, l-1, i]) for i in range(M+1)])
            for m in range(M, 2*M-l):
                Ybegin[:, l, m+1] = Ybegin[:, l, m] + \
                    h*(ff(t[j*K+m], Ybegin[:, l, m])-ff(t[j*K+m], Ybegin[:, l-1, m])) + \
                    h*sum([S[M-1, i]*ff(t[j*K+m-M+i+1], Ybegin[:, l-1, m-M+i+1])
                           for i in range(M+1)])
        Ypred = Ybegin[:, 0, -1]
        yy[:, j*K:j*K+M] = Ybegin[:, M, 0:M]
        # declare and fill up Y1corr and Y2corr for phase two
        Y1corr = np.zeros([d, M, M+1])
        Y2corr = np.zeros([d, M])
        for l in range(1, M+1):
            # 'lm' is for corrector 'l' (trying to save space for Y1corr&Y2corr
            lm = l - 1
            Y1corr[:, lm, :] = Ybegin[:, l-1, M-l:2*M-l+1]
            Y2corr[:, lm] = Ybegin[:, l, 2*M-l-1]
        # ----------------------------------------------------------------------
        # Phase 2: all threads can go simultaneously now
        for m in range(M-1, K):
            # predictor
            Tpred = t_ext[j*K+m+M]
            Ypred = predictor(Ypred, Tpred)
            # correctors
            for l in range(1, M+1):
                lm = l - 1
                Tvec = t_ext[j*K+m-l+1:j*K+m-l+1+M+1]
                Y2corr[:, lm] = corrector(Y2corr[:, lm], Y1corr[:, lm, :], Tvec)
            # update the stencil
            Y1corr[:, 0, 0:M] = Y1corr[:, 0, 1:M+1]
            Y1corr[:, 0, M] = Ypred
            for lm in range(1, M):
                Y1corr[:, lm, 0:M] = Y1corr[:, lm, 1:M+1]
                Y1corr[:, lm, M] = Y2corr[:, lm-1]
            # put the most corrected point in the final answer
            yy[:, j*K+m+1] = Y2corr[:, M-1]
    return t, yy


if __name__ == "__main__":
    T = 10.0
    y0 = np.array([1.0, 0])
    p = 4  # RIDC(6,40)
    M = p-1
    K = 200
    N = 1000
    start = time.perf_counter()
    tt, yy = RIDCsolverP(func, T, y0, N, M, K)
    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')
    print(tt[-1:])
    print(yy[:, -1])
