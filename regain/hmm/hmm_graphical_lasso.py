# BSD 3-Clause License

# Copyright (c) 2017, Federico T.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
from __future__ import division

import warnings

import numpy as np
from regain.covariance.graphical_lasso_ import GraphicalLasso, graphical_lasso
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.covariance import empirical_covariance
from sklearn.utils.validation import check_array


def scaled_forward_backward(X, pis, probabilities, A):
    # TODO it could be parallelised
    N, _ = X.shape
    K = pis.shape[0]
    alphas = np.zeros((N, K))
    betas = np.zeros((N, K))
    alphas[0, :] = pis.ravel() * probabilities[0, :]
    betas[-1, :] = 1
    cs = np.zeros(N)

    for n in range(1, N):
        for k in range(K):
            alphas[n, k] = probabilities[n, k] * np.sum(
                alphas[n - 1, :] * A[:, k])
        cs[n] = np.sum(alphas[n, :])
        alphas[n, :] /= cs[n]
    for n in range(N - 2, -1, -1):
        for k in range(K):
            betas[n, k] = np.sum(
                betas[n + 1, :] * probabilities[n + 1, :]
                *  # non sono sicura che ci vada il k in probabilities
                A[k, :])
        betas[n, :] /= cs[n + 1]
    return alphas, betas, cs


def forward_backward(X, pis, probabilities, A):
    # TODO it could be parallelised
    N, _ = X.shape
    K = pis.shape[0]
    alphas = np.zeros((N, K))
    betas = np.zeros((N, K))
    alphas[0, :] = pis.ravel() * probabilities[0, :]
    betas[-1, :] = 1

    for n in range(1, N):
        for k in range(K):
            alphas[n, k] = probabilities[n, k] * np.sum(
                alphas[n - 1, :] * A[:, k])
    for n in range(N - 2, -1, -1):
        for k in range(K):
            betas[n, k] = np.sum(
                betas[n + 1, :] * probabilities[n + 1, :]
                *  # non sono sicura che ci vada il k in probabilities
                A[k, :])
    return alphas, betas


def compute_likelihood(gammas, pis, xi, A, probabilities):
    N, K = probabilities.shape
    pis = pis + 1e-200
    aux = np.sum(gammas[0, :] * np.log(pis))
    for n in range(N - 1):
        aux += np.sum(xi[n, :, :] * np.log(A))
    aux += np.sum(gammas * probabilities)
    return aux


class HMM_GraphicalLasso(GraphicalLasso, KMeans):
    """TODOOOO

    Parameters
    ----------
    alpha : positive float, default 0.01
        The regularization parameter: the higher alpha, the more
        regularization, the sparser the inverse covariance.

    n_clusters: integer, default 3
        The number of clusters to form as well as the number of centroids to
        generate.

    tol : positive float, default 1e-4
        Absolute tolerance to declare convergence.

    max_iter : integer, default 100
        The maximum number of iterations.

    verbose : boolean, default False
        If verbose is True, the objective function, rnorm and snorm are
        printed at each iteration.



    Attributes
    ----------
    covariance_ : array-like, shape (n_features, n_features)
        Estimated covariance matrix

    precision_ : array-like, shape (n_features, n_features)
        Estimated pseudo inverse matrix.

    n_iter_ : int
        Number of iterations run.

    """
    def __init__(self,
                 alpha=0.01,
                 n_clusters=3,
                 max_iter=100,
                 tol=1e-4,
                 verbose=False,
                 mode='scaled',
                 init='uniform'):
        GraphicalLasso.__init__(self, alpha=alpha, tol=tol, max_iter=max_iter)
        KMeans.__init__(
            self, n_clusters=n_clusters
        )  # TODO add other type of initialisation rather then the default
        self.verbose_ = verbose
        self.mode = mode

    def fit(self, X, y=None):
        """Fit the GraphicalLasso model to X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data matrix that represents a temporal sequence of data.
        """
        # Covariance does not make sense for a single feature
        X = check_array(X,
                        ensure_min_features=2,
                        ensure_min_samples=2,
                        estimator=self)
        # Initialization
        clusters = KMeans.fit(self, X).labels_
        # initialisation
        N, d = X.shape
        K = self.n_clusters
        means = np.zeros((K, d))
        thetas = []
        # TODO: it may be changed to sampling from a uniform distribution
        pis = np.ones((K, 1)) / K
        A = np.ones((K, K)) / K
        # A = np.random.uniform(0, 1, (K, K))
        # A = A / np.sum(A, axis=0)[:, np.newaxis]
        for i, l in enumerate(np.unique(clusters)):
            means[i, :] = np.mean(X[np.where(clusters == l)[0], :], axis=0)
            emp_cov = empirical_covariance(X - means[i, :],
                                           assume_centered=True)
            thetas.append(graphical_lasso(emp_cov, alpha=self.alpha)[0])

        covariances = [np.linalg.pinv(t) for t in thetas]
        probabilities = np.zeros((N, K))
        for n in range(N):
            for k in range(K):
                probabilities[n,
                              k] = multivariate_normal.pdf(X[n, :],
                                                           mean=means[k, :],
                                                           cov=covariances[k])
        likelihood_ = -np.inf

        for iter_ in range(self.max_iter):
            # E-step
            if self.mode == 'scaled':
                alphas, betas, cs = scaled_forward_backward(
                    X, pis, probabilities, A)
                gammas = alphas * betas
                xi = np.zeros((N - 1, K, K))
                for n in range(1, N):
                    for k in range(K):
                        xi[n-1, k, :] = cs[n]*alphas[n-1, k]*probabilities[n, k] * \
                                      betas[n, k]*A[k, :]
            else:
                alphas, betas = forward_backward(X, pis, probabilities, A)
                gammas = alphas * betas
                p_x = np.sum(alphas[-1, :])
                gammas = gammas / p_x  #np.sum(gammas, axis=1)[:, np.newaxis]
                #print(gammas)
                #print(gammas)
                xi = np.zeros((N - 1, K, K))
                for n in range(1, N):
                    for k in range(K):
                        xi[n-1, k, :] = alphas[n-1, k]*probabilities[n, k] * \
                                      betas[n, k]*A[k, :]

            # M-step
            pis = gammas[0, :] / np.sum(gammas[0, :])
            lambdas = self.alpha / np.sum(gammas,
                                          axis=0)  # should be of length K
            thetas = []
            for k in range(K):
                means[k, :] = np.sum(gammas[:, k][:, np.newaxis] * X,
                                     axis=0) / np.sum(gammas[:, k])
                # not sure it is like this
                S_k = (gammas[:, k][:, np.newaxis]*(X - means[k, :])).T.dot(X - means[k, :]) \
                    / np.sum(gammas[:, k])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    thetas.append(graphical_lasso(S_k, alpha=lambdas[k])[0])
                for j in range(K):
                    A[j, k] = np.sum(xi[:, j, k]) / np.sum(xi[:, j, :])
            covariances = [np.linalg.pinv(t) for t in thetas]
            probabilities = np.zeros((N, K))
            for n in range(N):
                for k in range(K):
                    probabilities[n, k] = multivariate_normal.pdf(
                        X[n, :], mean=means[k, :], cov=covariances[k])
            likelihood_old = likelihood_
            likelihood_ = compute_likelihood(gammas, pis, xi, A, probabilities)

            if self.verbose_:
                print(
                    'Iter: %d, likelihood: %.5f, difference: %.5f' %
                    (iter_, likelihood_, np.abs(likelihood_ - likelihood_old)))
            if np.abs(likelihood_ - likelihood_old) < self.tol:
                break

        else:
            warnings.warn('The optimisation did not converge.')

        self.precisions_ = thetas
        # self.states = NON CE li HO
        self.means_ = means
        self.state_change = A
        self.pis_ = pis
        self.labels_ = np.argmax(gammas, axis=1)
        self.gammas_ = gammas

        return self
