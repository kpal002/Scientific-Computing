import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Quadratic fit function
def parabola(x, a, b, c):
	return a*x**2 + b*x + c


# Create a matrix with columns [1, x, x^2]
ones = np.ones(11)
xfeature = np.arange(-5,6)
squaredfeature = xfeature**2
b = 0.2*np.square(xfeature) + np.random.rand(len(xfeature))

features = np.concatenate((np.vstack(ones),np.vstack(xfeature),np.vstack(squaredfeature)), axis = 1)


# Least square function in numpy
lst_fit = np.linalg.lstsq(features, b, rcond= None)[0]


# Using the curve_fit function which also implements the least square method just to verify my result.
curve_fit, _ = curve_fit(parabola, xfeature, b)
plt.plot(xfeature,b,'b-', label='data')
u = np.linspace(-5,5,100)
plt.plot(u, u**2*lst_fit[2] + u*lst_fit[1] + lst_fit[0],'r-', label='fit using least squares: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(np.flip(lst_fit) ))
plt.plot(xfeature, parabola(xfeature, *curve_fit), 'g--', label='fit using curve_fit function: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(curve_fit))
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
