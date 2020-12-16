import numpy as np
import copy

theta = np.float64(np.array([[0.5, 0.5], [6., 7.], [1., 4.]]))
x = np.atleast_2d([-1., 0., 4., 5., 6.]).T

def gaussian_likelihood(x, theta):
    n = x.shape[0]
    K = theta.shape[1]
    d = x.shape[1]
    p = theta[0,:]
    mu = theta[1,:]
    variance = theta[2,:]
    return p/((2*np.pi*variance)**(d/2)) * np.exp(-(np.ones([n,K])*x-mu)**2/(2*variance))

#E step
likelihood = gaussian_likelihood(x, theta)
loglikelihood = np.log(np.sum(likelihood, axis=1))
loglikelihood_D = np.sum(loglikelihood)
print(np.round(loglikelihood_D, 1))

assignment = np.argmax(likelihood, axis=1)
print(assignment)

#M step
p_j_i = theta[0,:]*likelihood/np.sum(theta[0,:]*likelihood, axis=1)[np.newaxis].T

theta[0,:] = np.sum(p_j_i, axis=0)/x.shape[0]
theta[1,:] = np.sum(x*p_j_i, axis=0)/np.sum(p_j_i, axis=0)
theta[2,:] = np.sum(p_j_i*(np.ones([x.shape[0],theta.shape[1]])*x-theta[1,:])**2, axis=0)/np.sum(p_j_i, axis=0)
print(theta)

#until convergence
theta_old = np.float64(np.zeros(np.shape(theta)))
while np.linalg.norm(theta_old - theta, 2) > 1e-6:
    likelihood = gaussian_likelihood(x, theta)

    p_j_i = theta[0, :] * likelihood / np.sum(theta[0, :] * likelihood, axis=1)[np.newaxis].T

    theta_old = theta.copy()
    theta[0, :] = np.sum(p_j_i, axis=0) / x.shape[0]
    theta[1, :] = np.sum(x * p_j_i, axis=0) / np.sum(p_j_i, axis=0)
    theta[2, :] = np.sum(p_j_i * (np.ones([x.shape[0], theta.shape[1]]) * x - theta[1, :]) ** 2, axis=0) / np.sum(p_j_i,
                                                                                                                  axis=0)

print(theta)