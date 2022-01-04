import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

np.random.seed(12)


def func(x, a, b):
    return  a*np.array(x) + b


def power(A, num_simulations,exact):
        
        b_k = np.random.rand(A.shape[0])
        
        log_err = [np.log(np.linalg.norm(b_k-exact))]
        for _ in range(num_simulations):
                
                b_k1 = np.dot(A, b_k)
                
                b_k1_norm = np.linalg.norm(b_k1) 
                b_k = b_k1 / b_k1_norm 
                log_err.append(np.log(np.linalg.norm(b_k-exact)))

        return b_k,log_err
n=50
A = np.random.rand(4,4)
A = A + A.T
[eigs, vecs] = np.linalg.eig(A)

x1 = power(A, n,vecs[:,0])[1][:n-1]
y1 = power(A, n,vecs[:,0])[1][1:n]

popt, _ = curve_fit(func, x1, y1)
print(popt)
plt.plot(x1, y1, 'b-', label='data')
plt.plot(x1, func(x1, *popt), 'g--',label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
plt.xlabel('e (k)')
plt.ylabel('e (k+1)')
plt.title('Jacobi error')
plt.legend()
plt.show()
