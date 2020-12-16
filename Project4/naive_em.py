"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    rated = (X != 0).astype(np.float)
    d = np.sum(rated, axis=1)[:,None]

    likelihood = np.ones(d.shape) * mixture.p / (
            (2 * np.pi * np.ones(d.shape) * mixture.var) ** (d / 2)) * \
           np.exp(-np.linalg.norm(X[:, None, :] - rated[:, None, :] * mixture.mu, axis=2) ** 2 / (
                   2 * mixture.var[None, :]))
    post = likelihood/np.sum(likelihood, axis=1)[:,None]
    ll = np.sum(np.log(np.sum(likelihood, axis=1)))

    return post, ll


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    rated = (X != 0).astype(np.float)[:, None, :]
    p = 1/post.shape[1] * np.ones(post.shape[1])

    p_j_i = p * post / np.sum(p * post, axis=1)[:,None]

    p = np.sum(p_j_i, axis=0) / X.shape[0]
    mu = np.sum(X[:,None,:] * rated*p_j_i[:,:,None], axis=0) / np.sum(rated*p_j_i[:,:,None], axis=0)
    var = np.sum(p_j_i * (np.linalg.norm(X[:,None,:] - rated * mu, axis=2) ** 2), axis=0) / \
          np.sum(np.sum(rated*p_j_i[:,:,None], axis=0), axis=1)

    return GaussianMixture(mu, var, p)


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
    while old_ll is None or ll - old_ll >= 1e-6*np.absolute(ll):
        old_ll = ll
        post, ll = estep(X, mixture)
        mixture = mstep(X, post)

    return mixture, post, ll
