## Functions for Different Methods ##
import math
import numpy as np
import scipy
import scipy.optimize as so

## Adams-Bashforth ##
"""
Inputs:
f - Differential Equation
y0 - Initial Conditions
t - Time Array
h - Timestep
k - Desired Order

Outputs:
y - Numerical solution using Explicit Adams method
"""
def AdamsBashforth(f, y0, t, h, k):
    y = np.zeros((len(t), y0.shape[0]))
    beta = np.zeros((5, 6))
    beta[:1, 0] = 1
    beta[:2, 1] = np.array([3, -1])/2
    beta[:3, 2] = np.array([23, -16, 5])/12
    beta[:4, 3] = np.array([55, -59, 37, -9])/24
    beta[:5, 4] = np.array([1901, -2774, 2616, -1274, 251])/720

    y[0, :] = y0

    for i in range(len(t) - 1):
        y[i + 1, :] = y[i, :]
        n = 1
        for j in range(n):
            y[i + 1, :] += h*beta[j, n - 1]*f(y[i - j], t[i - j])
        if n < k:
            n += 1
    return y

## Adams Moulton ##
"""
Inputs:
f - Differential Equation
y0 - Initial Conditions
t - Time Array
h - Timestep
k - Desired Order

Outputs:
y - Numerical solution using Implicit Adams method
"""
def AdamsMoulton(f, y0, t, h, k):
    y = np.zeros((len(t), y0.shape[0]))
    beta = np.zeros((5, 6))
    beta[:2, 0] = np.array([1, 1])/2
    beta[:3, 1] = np.array([5, 8, -1])/12
    beta[:4, 2] = np.array([9, 19, -5, 1])/24
    beta[:5, 3] = np.array([251, 646, -264, 106, -19])/720

    y[0, :] = y0

    for i in range(len(t) - 1):
        y[i + 1, :] = y[i, :]
        n = 1
        def g_AM(y0):
            g = y0 - y[i,:] - h*beta[0, n - 1]*f(y0, t[i + 1])
            for j in range(n):
                g += -h*beta[j + 1, n - 1]*f(y[i - j], t[i - j])
            return g
        y[i + 1, :] = so.fsolve(g_AM,y[i,:])
        if n < k:
            n += 1
    return y