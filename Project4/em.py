"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture
import naive_em


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    rated = (X != 0).astype(np.float)
    d = np.sum(rated, axis=1)[:,None]

    f = np.log(np.ones(d.shape) * mixture.p + 1e-16) - \
        d * (np.log(2 * np.pi) + np.log(np.ones(d.shape) * mixture.var))/2 - \
        np.linalg.norm(X[:, None, :] - rated[:, None, :] * mixture.mu, axis=2) ** 2 / (2 * mixture.var[None, :])
    f_max = np.max(f, axis=1, keepdims=True)

    l = f - (f_max + logsumexp(f - f_max, axis=1, keepdims=True))

    post = np.exp(l)
    if (~np.isfinite(post)).any():
        hold=1
    ll = np.sum(post * (f_max + logsumexp(f - f_max, axis=1, keepdims=True)))

    return post, ll


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n = X.shape[0]
    d = X.shape[1]
    K = post.shape[1]
    rated = (X != 0).astype(np.float)[:, None, :]
    p = 1 / post.shape[1] * np.ones(post.shape[1])
    p_j_i = mixture.p * post / np.sum(mixture.p * post, axis=1)[:,None]

    p = np.sum(p_j_i, axis=0) / n
    #p = p/np.sum(p, axis=0)

    mu_mask = (np.sum(rated*p_j_i[:,:,None], axis=0) >= 1.)
    if ~(np.isfinite(np.sum(X[:,None,:] * rated*p_j_i[:,:,None], axis=0)).all()) or (np.sum(rated*p_j_i[:,:,None], axis=0) * mu_mask).any() == 0:
        hold=1
    delta = np.sum(rated, axis=2)
    mu_new = (np.sum(p_j_i[:,:,None] * delta[:,:,None] * X[:,None,:], axis=0) / np.sum(p_j_i*delta, axis=0)[:,None]) * mu_mask
    mu_old = mixture.mu * (mu_mask == 0)
    mu = mu_new + mu_old

    var = np.sum(p_j_i * (np.linalg.norm(X[:,None,:] - rated * mu, axis=2) ** 2), axis=0) / \
          np.sum(np.sum(rated*p_j_i[:,:,None], axis=0), axis=1)

    if (~np.isfinite(mu)).any():
        hold=1
    if (~np.isfinite(var)).any():
        hold=1
    if (~np.isfinite(p)).any():
        hold=1
    return GaussianMixture(mu, np.maximum(var, min_variance), p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    old_ll = None
    ll = None
    while old_ll is None or ll - old_ll > 1e-6*np.absolute(ll):
        old_ll = ll
        post, ll = estep(X, mixture)
        mixture = mstep(X, post, mixture)

    return mixture, post, ll


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    rated = (X != 0).astype(np.float)
    d = np.sum(rated, axis=1)[:,None]

    f = np.log(np.ones(d.shape) * mixture.p + 1e-16) - \
        d * (np.log(2 * np.pi) + np.log(np.ones(d.shape) * mixture.var))/2 - \
        np.linalg.norm(X[:, None, :] - rated[:, None, :] * mixture.mu, axis=2) ** 2 / (2 * mixture.var[None, :])
    f_max = np.max(f, axis=1, keepdims=True)

    l = f - (f_max + logsumexp(f - f_max, axis=1, keepdims=True))

    post = np.exp(l)
    d = X.shape[1]
    output = np.sum((1 / ((2 * np.pi * mixture.var) ** (d / 2)) * np.exp(-(X- mixture.mu) ** 2 / (2 * mixture.var)))[:,None,:] * post, axis=1)
    #output = post*
    #output += post*(X==0)
    return output