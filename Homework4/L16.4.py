import numpy as np


def gaussian_probability(x, mu, variance):
    d = x.shape[1]
    return 1/((2*np.pi*variance)**(d/2)) * np.exp(-(x-mu)**2/(2*variance))

mu = np.atleast_2d([-3., 2.]).T
variance = np.atleast_2d([4., 4.]).T
x = np.atleast_2d([0.2, -0.9, -1, 1.2, 1.8]).T
p = np.array([0.5, 0.5]).T

p_i_j1 = gaussian_probability(x, mu[0,:], variance[0,:])
p_i_j2 = gaussian_probability(x, mu[1,:], variance[1,:])
#print(p_i_j1)

p = np.atleast_2d([0.5, 0.5]).T
sum_p_i_j = p[0]*p_i_j1 + p[1]*p_i_j2
p_j1_i = p[0]*p_i_j1/sum_p_i_j
p_j2_i = p[1]*p_i_j2/sum_p_i_j
print(p_j1_i)

p[0] = np.sum(p_j1_i) / x.shape[0]
p[1] = np.sum(p_j2_i) / x.shape[0]
print(p)
mu[0,:] = np.sum(p_j1_i * x) / np.sum(p_j1_i)
mu[1,:] = np.sum(p_j2_i * x) / np.sum(p_j2_i)
print(mu)
variance[0,:] = np.sum(p_j1_i*(x-mu[0,:])**2) / np.sum(p_j1_i)
variance[1,:] = np.sum(p_j2_i*(x-mu[1,:])**2) / np.sum(p_j2_i)
print(variance)