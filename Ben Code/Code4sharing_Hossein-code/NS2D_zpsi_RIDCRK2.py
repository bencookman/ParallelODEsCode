import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from scipy.interpolate import lagrange

# ++++++++++ setting up parameters + +++++++++
nu = 0                              # Viscosity
Nx = 128                            # number of nodes in one dimension
Ny = 128                            #
dt = 0.02                           # time step
Tend = 16                           # end time (integratino length)
Nt = int(Tend/dt)                   # number of steps in time

# ++++++++++ forming physical (spatial) coordinates ++++++++++
dx = 2*np.pi/Nx                     # spatial grid spacing
dy = 2*np.pi/Ny                     #
x_vec = (np.arange(0, Nx)+1)*dx     # vector of spatial grid points in x
y_vec = (np.arange(0, Ny)+1)*dy     #
xx, yy = np.meshgrid(x_vec, y_vec)  # meshgrid of spatial grid points


# ++++++++++ forming spectral coordinates ++++++++++
Kx = np.zeros(Nx)
Kx[:int(Nx / 2)] = np.arange(int(Nx / 2))
Kx[int(Nx / 2):] = np.arange(-int(Nx / 2), 0)
Ky = Kx
Kxx, Kyy = np.meshgrid(Kx, Ky)


# ++++++++++ calculating operators used in time loop ++++++++++
# this helps the code to be faster
# and avoid unncessary computation inside the time loop
# >>> calculating spectral operators <<<
k2poisson = Kxx**2+Kyy**2
k2poisson[0, 0] = 1
ikx_k2 = 1j*Kxx / k2poisson
iky_k2 = 1j*Kyy / k2poisson

# de-aliasing mask: forces the nonlinear terms for kx,ky>2/3 to be zero
# depending on the problem and the type of dissipation can be relaxed ...
L = np.ones(np.shape(k2poisson))
for i in range(Nx):
    for j in range(Ny):
        if (abs(Kxx[j, i]) > max(Kx)*2./3.):
            L[j, i] = 0
        elif (abs(Kyy[j, i]) > max(Ky)*2./3.):
            L[j, i] = 0


# ++++++++++ initial condition ++++++++++
zr = 2*(-np.exp(-((xx-np.pi)**2+(yy-np.pi+np.pi/10)**2)*5) +
        +np.exp(-((xx-np.pi)**2+(yy-np.pi-np.pi/10)**2)*5))
zk = fft2(zr)


# ++++++++++ setting up RIDC stuff before time loop ++++++++++
# RIDC parameters
M = 5  # number of corrector + 1
# >>>> forming quadraure integral's coefficients
# M: the number of points in calculating quadraure integral
# Note Mm is the number of correctors
Mm = M - 1

# Forming the quadraure matrix S[m,i]
S = np.zeros([Mm, Mm+1])
for m in range(Mm):  # Calculate qudrature weights
    for i in range(Mm+1):
        x = np.arange(Mm+1)  # Construct a polynomial
        y = np.zeros(Mm+1)   # which equals to 1 at i, 0 at other points
        y[i] = 1
        p = lagrange(x, y)
        para = np.array(p)    # Compute its integral
        P = np.zeros(Mm+2)
        for k in range(Mm+1):
            P[k] = para[k]/(Mm+1-k)
        P = np.poly1d(P)
        S[m, i] = P(m+1) - P(m)
Svec = S[Mm-1, :]


# function for calculating the quadraure integral
def Quad_int1(RHSvec, iM):
    integ_sum = np.zeros((Nx, Ny), dtype=complex)
    for ii in range(M):
        integ_sum += S[iM, ii]*RHSvec[ii, :]
    return integ_sum


def Quad_int2(RHSvec):
    integ_sum = np.zeros((Nx, Ny), dtype=complex)
    for ii in range(M):
        integ_sum += Svec[ii]*RHSvec[ii, :]
    return integ_sum


# ---> The array of RHS <---
# This array is very essential to RIDC method,
# where we store the values of RHS at different correction level and time
# first index from the left is the order of correctin; 0 -> predictor
# second index from the left is the time (in a moving window)
# other indices to the right are dimensions of space
RHSz_mat = np.zeros((Mm, M, Nx, Ny), dtype=complex)
# ---> answer at different correction level <---
# We store the answer (i.e. the desired field which vorticity here) in zkCor
# Note at the intial phase (1) zkCor contains the asnwer
# at different correction levels but the same time. After the inital phase,
# corrector of order (l) is (l) time steps behind predictor (i.e. zKCor[0,:])
# the first index of zkCor is the order of corrector/predictor
zkCor = np.zeros((M, Nx, Ny), dtype=complex)


# ++++++++++ RHS 2D Navier-Stokes Equation ++++++++++
def RHS(zk):
    '''
    calculate the RHS of momentum eq written for vorticity
    '''
    ur = np.real(ifft2(iky_k2*zk))
    z_xr = np.real(ifft2(1j*Kxx*zk))
    vr = np.real(ifft2(-ikx_k2*zk))
    z_yr = np.real(ifft2(1j*Kyy*zk))
    nlr = ur*z_xr+vr*z_yr
    return -L*fft2(nlr)


# ~~~~~~~~~~~ Fill up RHSz_mat & zkCor with inital condition ~~~~~~~~~~~
for iCor in range(Mm):
    RHSz_mat[iCor, 0, :] = RHS(zk)
for iCor in range(M):
    zkCor[iCor, :] = zk


# ++++++++++ TIME INTEGRATION ++++++++++

# ================== INITIAL PHASE (1) ==================
# preditor
for iTime in range(0, M-1):
    K1 = RHSz_mat[0, iTime, :]
    K2 = RHS(zkCor[0, :] + dt*K1)
    zkCor[0, :] = zkCor[0, :] + (dt/2)*(K1 + K2)
    RHSz_mat[0, iTime+1, :] = RHS(zkCor[0, :])

# correctors: use Integral Deffered Correction (not RIDC)
for iCor in range(1, M-1):
    ll = iCor - 1
    for iTime in range(0, M-1):
        QSum = Quad_int1(RHSz_mat[ll, :], iTime)
        K1 = RHSz_mat[iCor, iTime, :]-RHSz_mat[ll, iTime, :]
        K2 = RHS(zkCor[iCor, :] + dt*(K1 + QSum))-RHSz_mat[ll, iTime+1, :]
        zkCor[iCor, :] = zkCor[iCor, :] + dt*((K1 + K2)/2 + QSum)
        RHSz_mat[iCor, iTime+1, :] = RHS(zkCor[iCor, :])
# treat the last correction loop a little different
for iTime in range(0, M-1):
    QSum = Quad_int1(RHSz_mat[-1, :], iTime)
    K1 = RHS(zkCor[-1, :])-RHSz_mat[-1, iTime, :]
    K2 = RHS(zkCor[-1, :] + dt*(K1 + QSum))-RHSz_mat[-1, iTime+1, :]
    zkCor[-1, :] = zkCor[-1, :] + dt*((K1 + K2)/2 + QSum)

# ================== INITIAL PHASE (2) ==================
for iTime in range(M-1, 2*M-2):
    iStep = iTime - (M-1)
    # prediction loop
    K1 = RHSz_mat[0, M-1, :]
    K2 = RHS(zkCor[0, :] + dt*K1)
    zkCor[0, :] = zkCor[0, :] + (dt/2)*(K1 + K2)
    # correction loops
    for ll in range(iStep):
        iCor = ll + 1
        QSum = Quad_int2(RHSz_mat[ll, :])
        K1 = RHSz_mat[iCor, -1, :]-RHSz_mat[ll, -2, :]
        K2 = RHS(zkCor[iCor, :] + dt*(K1 + QSum))-RHSz_mat[ll, -1, :]
        zkCor[iCor, :] = zkCor[iCor, :] + dt*((K1 + K2)/2 + QSum)
    RHSz_mat[0, 0:M-1, :] = RHSz_mat[0, 1:M, :]
    RHSz_mat[0, M-1, :] = RHS(zkCor[0, :])
    for ll in range(iStep):
        iCor = ll + 1
        RHSz_mat[iCor, 0:M-1, :] = RHSz_mat[iCor, 1:M, :]
        RHSz_mat[iCor, M-1, :] = RHS(zkCor[iCor, :])

# ================== MAIN LOOP FOR TIME ==================
for iTime in range(2*M-2, Nt+M-1):
    t = (iTime-(M-1))*dt
    # prediction loop
    K1 = RHSz_mat[0, M-1, :]
    K2 = RHS(zkCor[0, :] + dt*K1)
    zkCor[0, :] = zkCor[0, :] + (dt/2)*(K1 + K2)
    # correction loops up to but not including last one
    for ll in range(M-2):
        iCor = ll + 1
        QSum = Quad_int2(RHSz_mat[ll, :])
        K1 = RHSz_mat[iCor, -1, :]-RHSz_mat[ll, -2, :]
        K2 = RHS(zkCor[iCor, :] + dt*(K1 + QSum))-RHSz_mat[ll, -1, :]
        zkCor[iCor, :] = zkCor[iCor, :] + dt*((K1 + K2)/2 + QSum)
    # last correction loop
    QSum = Quad_int2(RHSz_mat[-1, :])
    K1 = RHS(zkCor[-1, :])-RHSz_mat[-1, -2, :]
    K2 = RHS(zkCor[-1, :] + dt*(K1 + QSum))-RHSz_mat[-1, -1, :]
    zkCor[-1, :] = zkCor[-1, :] + dt*((K1 + K2)/2 + QSum)
    # ~~~~~~~~~~~ Updating Stencil ~~~~~~~~~~~
    # ---> updating correctors stencil
    for ll in range(1, M-1):
        RHSz_mat[ll, 0:M-1, :] = RHSz_mat[ll, 1:M, :]
        RHSz_mat[ll, M-1, :] = RHS(zkCor[ll, :])
    # ---> updating predictor stencil
    RHSz_mat[0, 0:M-1, :] = RHSz_mat[0, 1:M, :]
    RHSz_mat[0, M-1, :] = RHS(zkCor[0, :])


zk = zkCor[-1, :]


zk_exact = np.load('zk_dipole_RIDCFE_dt0d00002.npy')
zr_exact = np.real(ifft2(zk_exact))
# ++++++++++ Post-processing the results ++++++++++


def error_norm(zr, zr_exact):
    ez = zr - zr_exact
    return np.sqrt(np.sum(ez**2))*np.sqrt(dx*dy)

# >>> calculate the velocities at the end of time integration


# zr_exact = load('data.npy')
# >>> plot the results
fig, axs = plt.subplots(1, 2)
ax = axs[0]
zr = np.real(ifft2(zk))
c = ax.pcolor(xx, yy, zr, cmap='RdBu')
fig.colorbar(c, ax=ax)

ax = axs[1]
c = ax.pcolor(xx, yy, zr_exact, cmap='RdBu')
fig.colorbar(c, ax=ax)


print(error_norm(zr, zr_exact))
plt.show()
