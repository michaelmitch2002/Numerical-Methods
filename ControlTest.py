## testing simple methods ##

import multistep as ms
import numpy as np
import math
import matplotlib.pyplot as plt

def dydt3(y,t):
    phi = 1.0#/180.0*math.pi
    L = 11.5
    a = 100
    gamma = 1/6
    f0 = y[3]*np.cos(y[2])
    f1 = y[3]*np.sin(y[2])
    f2 = y[3]*np.tan(phi)/L
    f3 = a - gamma*y[3]
    f = np.array([f0,f1,f2,f3])
    return f

def dydt(y,t):
    f1 = -5*y*np.cos(t)-np.log(t)*y #15*np.cos(10*t) + -0.01*y
    f = np.array([f1])
    return f1


t0 = 0
tend = 0.8
h = 0.0001
timesteps = int(tend/h)
t = np.linspace(0,tend,num = timesteps+1)
y0 = np.array([10,10,1,0])
y = ms.AdamsMoulton(dydt3, y0, t, h, 4)
print(y)

plt.plot(y[:,0], y[:,1])
plt.show()