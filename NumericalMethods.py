## Functions for Different Methods ##
import math
import numpy as np
import scipy
import scipy.optimize

######## Implicit Conditionals ###########################################

#Jacobian (df/dy)
def Jacobian(f, y, t):
    err = 10.0**(-8.0)
    emergencysizing = y.shape[0]
    fdev = np.zeros((emergencysizing, emergencysizing))
    for i in range((emergencysizing)):
        err_array = np.zeros((emergencysizing))
        err_array[i] = err
        ylarge = y + err_array
        ysmall = y - err_array
        fdev[:, i] = (f(ylarge, t) - f(ysmall, t))/(2.0*err)
    return fdev

##################################################################################

############### CLASSICAL METHODS #######################
############ Foward Euler #######################
def FE(f, y0, t, h):
    y = np.zeros((len(t),y0.shape[0]))
    y[0,:] = y0
    for i in range(len(t)-1):
        y[i+1,:] =  y[i,:] + h*f(y[i,:],t[i])
    return y

############ Backward Euler #######################
#g(y) 
def g_Euler(f,y,y0,t,h):
    g = (np.add(np.add(y, -y0),-h*f(y,t)))
    return g

#dg/dy
def dgdy_Euler(f,y,t,h):
    gdev = np.eye(y.shape[0]) - h*Jacobian(f,y,t)
    return gdev

#Newtons Method
def Newtons_Euler(f, y0, tnow, h, tol):
    ynow = y0
    diff = np.ones(y0.shape[0])
    iter = 0
    while np.linalg.norm(diff) > tol and iter < 100:
        diff = -np.linalg.solve(dgdy_Euler(f,ynow,tnow,h),g_Euler(f,ynow,y0,tnow,h))
        if np.linalg.norm(diff) > tol:
            ynow = np.add(ynow,diff)
            iter = iter + 1
    return ynow

def BE(f, y0, t, h):
    tol = 10.0**(-7.0)
    y = np.zeros((len(t),y0.shape[0]))
    y[0,:] = y0
    for i in range(len(t)-1):
        y[i+1,:] = Newtons_Euler(f, y[i,:], t[i+1], h, tol)
    return y

############ Midpoint #######################
#g(y) 
def g_Midpt(f,y,y0,t,h):
    g = y - y0 - h*f((y+y0)/2,(t-h/2))
    return g

#dg/dy
def dgdy_Midpt(f,y,y0,t,h):
    gdev = np.eye(y.shape[0]) - h/2*Jacobian(f,(y+y0)/2,t-h/2)
    return gdev

#Newtons Method
def Newtons_Midpt(f, y0, t, h, tol):
    ynow = y0
    diff = np.ones(y0.shape[0])
    iter = 0
    while np.linalg.norm(diff) > tol and iter < 100:
        diff = -np.inner(np.linalg.inv(dgdy_Midpt(f,ynow,y0,t,h)),g_Midpt(f,ynow,y0,t,h)).T
        if np.linalg.norm(diff) > tol:
            ynow = np.add(ynow,diff)
            iter = iter + 1
    return ynow

def Mid2(f, y0, t, h):
    tol = 10.0**(-7.0)
    y = np.zeros((len(t),y0.shape[0]))
    y[0,:] = y0
    for i in range(len(t)-1):
        y[i+1,:] = Newtons_Midpt(f, y[i,:], t[i+1], h, tol)
    return y

############ Trapezoidal #######################
#g(y) 
def g_Trap(f,y,y0,t,h):
    g = y - y0 - h/2*(f(y,t)+f(y0,t-h))
    return g

#dg/dy
def dgdy_Trap(f,y,y0,t,h):
    gdev = np.eye(y.shape[0]) - h/2*Jacobian(f,y,t)
    return gdev

#Newtons Method
def Newtons_Trap(f, y0, t, h, tol):
    ynow = y0
    diff = np.ones(y0.shape[0])
    iter = 0
    while np.linalg.norm(diff) > tol and iter < 100:
        diff = -np.inner(np.linalg.inv(dgdy_Trap(f,ynow,y0,t,h)),g_Trap(f,ynow,y0,t,h)).T
        if np.linalg.norm(diff) > tol:
            ynow = np.add(ynow,diff)
            iter = iter + 1
    return ynow

def Trap(f, y0, t, h):
    tol = 10.0**(-7.0)
    y = np.zeros((len(t),y0.shape[0]))
    y[0,:] = y0
    for i in range(len(t)-1):
        y[i+1,:] = Newtons_Trap(f, y[i,:], t[i+1], h, tol)
    return y

############### RUNGE-KUTTA METHODS #######################

########### EXPLICIT - RK4 ######################
####### TOO MUCH LOOPAGE --- NEED TO FIX ########
def RK_Exp(f, y0, t, h, A, b, c):
    s = c.shape[0]
    print(b)
    y = np.zeros((len(t),y0.shape[0]))
    y[0,:] = y0
    for w in range(len(t)-1):
        K = np.zeros((s,y0.shape[0]))
        for i in range(s): 
            sum = np.zeros(y0.shape[0])
            for j in range(s):
                sum += h*A[i,j]*K[j,:]
            K[i,:] = f(y[w] + h*sum,t[w]+c[i]*h)
        print(K)
        sum = np.zeros(y0.shape[0])
        for i in range(s):
            sum += b[i]*K[i,:]
        y[w+1,:] =  y[w,:] + h*sum
    return y


########### IMPLICIT - RK ######################
#g(y) 
def g_RK(f, Y_inter, y0, t, h, A, c):
    dim = y0.shape[0]
    s = A.shape[0] 

    g = np.zeros((dim*s))

    for i in range(0, s):
        g[i*dim:i*dim+dim] = Y_inter[i*dim:i*dim+dim] - y0
        for j in range(0, s):
            g[i*dim:i*dim+dim] += -h*(A[i, j]*(f(Y_inter[j*dim:j*dim+dim], t + h*c[j])))
    return g

def g_RK2(x, f, yprev, t_now, h, A, c):
    dim = yprev.shape[0]
    s = A.shape[0] 

    g = np.zeros((dim*s))

    for i in range(0, s):
        g[i*dim:i*dim+dim] = x[i*dim:i*dim+dim] - yprev
        for j in range(0, s):
            g[i*dim:i*dim+dim] += -h*(A[i, j]*(f(x[j*dim:j*dim+dim], t_now + h*c[j])))
    return g

#dg/dy
def dgdy_RK(f,y0,t,h,A):
    dim = y0.shape[0]
    s = A.shape[0] #ensure correct number of states
    gdev = np.eye(s*dim) - h*np.kron(A, Jacobian(f, y0, t))
    return gdev

#dg/dy
def dgdy_RK2(f,Y_inter,y0,t,h,A,c):
    dim = y0.shape[0]
    s = A.shape[0] #ensure correct number of states
    J = np.zeros((dim,dim,s))
    gdev = np.eye(s*dim)
    for j in range(s):
        J[:,:,j] = Jacobian(f, Y_inter[j*dim:j*dim+dim], t + h*c[j])
        for i in range(s):
            gdev[i*dim:i*dim+dim,j*dim:j*dim+dim] += - h*A[i,j]*J[:,:,j]
    return gdev

#Newtons Method
def Newtons_RK(f, y0, t_now, h, A, c, tol):
    s = A.shape[0]
    dim = y0.shape[0]

    Y_inter = np.zeros((dim*s))
    for i in range(0,s*dim, dim):
        Y_inter[i:i+dim] = y0

    diff = np.ones((s*dim, 1))
    iter = 0
    while np.linalg.norm(diff) > tol and iter < 100:
        diff = -np.linalg.solve(dgdy_RK2(f,Y_inter,y0,t_now,h,A,c),g_RK(f, Y_inter, y0, t_now, h, A, c))
        Y_inter = np.add(Y_inter, diff)
        iter = iter + 1
    return Y_inter

def RK_Imp(f, y0, t, h, A, b, c):
    tol = 10.0**(-5.0)
    s = c.shape[0]
    dim = y0.shape[0]

    y = np.zeros((dim, len(t)))
    y[:, 0] = y0
    Y_inter = np.ones((y0.shape[0]))

    for w in range(len(t)-1):
        #set initial guess for all values of Y_cap to previous iteration
        Y_inter = Newtons_RK(f, y[:,w], t[w], h, A, c, tol)
        sum = np.zeros(dim)
        for i in range(s):
            sum += b[i]*f(Y_inter[i*dim:i*dim + dim], t[w] + c[i]*h)
        y[:, w + 1] = y[:, w] + h*sum
    return y

def RK_Imp2(f, y0, t, h, A, b, c):
    tol = 10.0**(-7.0)
    s = c.shape[0]
    dim = y0.shape[0]

    y = np.zeros((dim, len(t)))
    y[:, 0] = y0
    Y_inter = np.zeros((s*dim, 1))
    for i in range(0,s*dim, dim):
        Y_inter[i:i+dim, 0] = y0

    for w in range(len(t)-1):
        #set initial guess for all values of Y_cap to previous iteration
        Y_inter = scipy.optimize.fsolve(g_RK2, Y_inter, (f , y[:, w],t[w],h,A,c))
        sum = 0
        for i in range(s):
            sum += b[i]*f(Y_inter[i*dim:i*dim + dim], t[w] + c[i]*h)
        y[:, w + 1] = y[:, w] + h*sum
        
    return y

