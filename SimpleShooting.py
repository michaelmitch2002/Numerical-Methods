import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import NumericalMethods as nm

# Parameters
N = 1000
h = 1 / N

# Arrays for solution and time values
y = np.zeros((N + 1, 2))
t = h * np.arange(N + 1)

# Differential equations
def fuv(yin,t):
    u, v = yin
    f1 = v
    f2 = -4/t*v - (t*u-1)*u
    return np.array([f1, f2])

# Boundary Value Problem function for a given initial condition
def BVP(initial_u):
    y[0, 0] = initial_u[0]     # Boundary condition for u(0)
    y[0, 1] = 0  # Boundary condition for v(0)
    
    y = nm.BE(fuv, y[0, :], t, h)
    # # Runge-Kutta 4th order method
    # for i in range(N):
    #     if i == 0:

    #     else:
    #         y1 = y[i]
    #         y2 = y[i] + 0.5 * h * fuv(y1,t[i])
    #         y3 = y[i] + 0.5 * h * fuv(y2,t[i]+1/2*h)
    #         y4 = y[i] + h * fuv(y3,t[i]+1/2*h)
    #         y[i + 1] = y[i] + (h / 6) * (fuv(y1,t[i]) + 2 * fuv(y2,t[i]+1/2*h) + 2 * fuv(y3,t[i]+1/2*h) + fuv(y4,t[i+1]))
    
    return y[N, 0]   # Return the last value of u (the target boundary condition)

# Initial guess for fsolve
initial_guess = 1
c = fsolve(BVP, initial_guess)

# Output results
print("Solution for c:", c)
print("Final time:", t[N])

# Plot the results
plt.plot(t, y[:, 0], label="u(t)")
plt.xlabel("Time t")
plt.ylabel("u(t)")
plt.title("Solution of the BVP")
plt.legend()
plt.show()