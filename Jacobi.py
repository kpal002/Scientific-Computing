import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Linear fit
def func(x, a, b):
    return  a*np.array(x) + b

# Implementation of the Jacobi method
def jacobi(A, b, x, n):
    log_err = [np.log(np.linalg.norm(x-solve(A,b)))]
    D = np.diag(A)
    R = A - np.diagflat(D)
    
    for i in range(n):
        x = (b - np.dot(R,x))/ D
        log_err.append(np.log(np.linalg.norm(x-solve(A,b))))
    return x,log_err

'''___Main___'''

A = np.array([[ 3.5, 2, -1],[1 , 2.5, 0],[ 1, 2, -3.5]])
b = [ 1.0, 2.0, 3.0]
x = [0.0, 0.0, 0.0]
n = 50

# Checking the convergence rate for the Jacobi method.
x1 = jacobi(A,b,x,n)[1][:n-1]
y1 = jacobi(A,b,x,n)[1][1:n]

popt, _ = curve_fit(func, x1, y1)

plt.plot(x1, y1, 'b-', label='data')
plt.plot(x1, func(x1, *popt), 'g--',label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
plt.xlabel('e (k)')
plt.ylabel('e (k+1)')
plt.title('Jacobi error')
plt.legend()
plt.show()