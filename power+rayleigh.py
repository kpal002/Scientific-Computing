import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

np.random.seed(12)


def func(x, a, b):
    return  a*np.array(x) + b

def rayleigh(A,v):
        Av = A.dot(v)
        return v.dot(Av)/v.dot(v)


def power(A, num_simulations,exact_vec,exact_val):
        
        b_k = np.random.rand(A.shape[0])

        log_err = [np.log(np.linalg.norm(b_k-exact_vec))]
        log_err2 = [np.abs(exact_val-rayleigh(A,b_k))]
        for _ in range(num_simulations):
                
                b_k1 = np.dot(A, b_k)
                
                b_k1_norm = np.linalg.norm(b_k1) 
                b_k = b_k1 / b_k1_norm 
                log_err.append(np.log(np.linalg.norm(b_k-exact_vec)))
                log_err2.append(np.abs(exact_val-rayleigh(A,b_k)))

        return b_k,rayleigh(A,b_k),log_err,log_err2
n=50
A = np.random.rand(4,4)
A = A + A.T
[eigs, vecs] = np.linalg.eig(A)

x1 = power(A, n,vecs[:,0],eigs[0])[2][:n-1]
y1 = power(A, n,vecs[:,0],eigs[0])[2][1:n]



x2 = power(A, n,vecs[:,0],eigs[0])[3][:4]
y2 = power(A, n,vecs[:,0],eigs[0])[3][1:5]

popt, _ = curve_fit(func, x1, y1)
popt2, _ = curve_fit(func, x2, y2)

plt.plot(x1, y1, 'b-', label='data')
plt.plot(x1, func(x1, *popt), 'g--',label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
plt.xlabel('e (k)')
plt.ylabel('e (k+1)')
plt.title('Power error')
plt.legend()
plt.show()




plt.plot(x2, y2, 'b-', label='data')
plt.plot(x2, func(x2, *popt2), 'g--',label='fit: a=%5.3f, b=%5.3f' % tuple(popt2))
plt.xlabel('e (k)')
plt.ylabel('e (k+1)')
plt.title('Rayleigh error')
plt.legend()
plt.show()
