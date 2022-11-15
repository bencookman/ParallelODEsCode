import numpy as np
from scipy.interpolate import lagrange

def integration_matrix_equispaced(M):
    S = np.zeros([M, M + 1])
    for m in range(M):
        for i in range(M + 1):
            x = np.arange(M + 1)
            y = np.zeros(M + 1)
            y[i] = 1.0

            p = lagrange(x, y)
            p_array = np.array(p)
            q = np.zeros(M + 2)
            for k in range(M + 1):
                q[k] = p_array[k]/(M + 1 - k)
            q = np.poly1d(q)
            S[m, i] = q(m + 1) - q(m)
    return S

if __name__ == "__main__":
    M_str = input("M = ")
    S = integration_matrix_equispaced(int(M_str))
    print(S)