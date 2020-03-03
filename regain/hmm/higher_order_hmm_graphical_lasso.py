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
from joblib import Parallel, delayed
from regain.covariance.graphical_lasso_ import GraphicalLasso, graphical_lasso
from regain.hmm.hmm import (HMM_GraphicalLasso, _initialization,
                            compute_likelihood)
from regain.hmm.utils import probability_next_point, viterbi_path
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.covariance import empirical_covariance
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_array


def NormAlpha(n, K, alp, A, prob):
    s2 = 0
    for k in range(K):
        s1 = 0
        for j in range(K):
            s1 += alp[n, j] * A[j, k]
        s2 += s1 * prob[n + 1, k]
    return s2


def NormAlphaUpdate(K, alp, A, prob):
    s2 = 0
    for k in range(K):
        s1 = 0
        for j in range(K):
            s1 += alp[-1, j] * A[j, k]
        s2 += s1 * prob[-1, k]
    return s2


def scaled_forward_backward(X, pis, probabilities, A):
    # TODO it could be parallelised
    N, _ = X.shape
    K = pis.shape[0]
    alphas = np.zeros((N, K))
    betas = np.zeros((N, K))

    # Initial forward probabilities
    alphas[0, :] = pis.ravel() * probabilities[0, :] / np.sum(
        pis.ravel() * probabilities[0, :])
    # recursive forward algorithm
    for n in range(N - 1):
        Normn = NormAlpha(n, K, alphas, A, probabilities)
        for k in range(K):
            alphas[n + 1, k] = np.sum(
                alphas[n, :] * A[:, k]) * probabilities[n + 1, k] / Normn

    # Initial backward probabilities
    betas[-1, :] = 1

    # recursive backward algorithm
    for n in range(N - 2, -1, -1):
        Normn = NormAlpha(n, K, alphas, A, probabilities)
        for k in range(K):
            betas[n, k] = np.sum(
                A[k, :] * probabilities[n + 1, :] * betas[n + 1, :]) / Normn
    return alphas, betas


def forward_backward(X, pis, probabilities, A):
    # TODO it could be parallelised
    N, _ = X.shape
    K = pis.shape[0]
    alphas = np.zeros((N, K))
    betas = np.zeros((N, K))

    # Initial forward probabilities
    alphas[0, :] = pis.ravel() * probabilities[0, :]
    # recursive forward algorithm
    for n in range(N - 1):
        for k in range(K):
            alphas[n + 1, k] = probabilities[n + 1, k] * np.sum(
                alphas[n, :] * A[:, k])

    # Normalization term
    P_X = np.sum(alphas[-1, :])

    # Initial backward probabilities
    betas[-1, :] = 1

    # recursive backward algorithm
    for n in range(N - 2, -1, -1):
        for k in range(K):
            betas[n, k] = np.sum(A[k, :] * probabilities[n + 1, :] *
                                 betas[n + 1, :])

    return alphas, betas, P_X


def TransMatrixInit(method, K, nu):

    A = np.zeros((K**nu, K**nu))
    for i in range(K**nu):
        index_fill = np.zeros(K**nu, bool)
        for j in range(K**nu):
            index_fill[j] = np.floor((i) / K) == (j) - np.floor(
                (j) / (K**(nu - 1))) * K**(nu - 1)
        if method is None or str(method).lower() == 'uniform':
            A[i, index_fill] = 1 / np.sum(index_fill)
        elif str(method).lower() == 'random_uniform':
            draw = np.random.uniform(0, 1, np.sum(index_fill))
            draw /= np.sum(draw)
            A[i, index_fill] = draw
        elif str(method).lower() == 'random_dirichlet':
            A[i, index_fill] = np.random.dirichlet(np.ones(np.sum(index_fill)),
                                                   1)
        else:
            raise ValueError(
                'Unexpected value for probabilities initialisations. '
                'Options are uniform, random_uniform and '
                'random_dirichlet, found' + str(method))
    return A


def ImSet(K, i, nu, m):
    lower = np.floor(i / (K**(nu - m)) * K**(nu - m))
    upper = np.floor(i / (K**(nu - m)) + 1) * K**(nu - m)
    return list(range(int(lower), int(upper)))


def hhmm_graphical_lasso(X,
                         A,
                         pis,
                         means,
                         covariances,
                         alpha=0.1,
                         max_iter=100,
                         mode='scaled',
                         verbose=0,
                         warm_restart=False,
                         tol=5e-3,
                         m=1,
                         r=2):

    nu = max([m, r])
    K = int(pis.shape[0]**(1 / nu))
    N, d = X.shape
    probabilities = np.zeros((N, int(K**nu)))
    for n in range(N):
        for k in range(int(K**nu)):
            probabilities[n, k] = multivariate_normal.pdf(X[n, :],
                                                          mean=means[k, :],
                                                          cov=covariances[k])
    likelihood_ = -np.inf
    thetas = []
    for iter_ in range(max_iter):
        # E-step
        if mode == 'scaled':
            alphas, betas = scaled_forward_backward(X, pis, probabilities, A)
            gammas = alphas * betas
            for n in range(np.size(gammas, axis=0)):
                gammas[n, :] = gammas[n, :] / np.sum(
                    alphas[n, :] * betas[n, :])
            xi = np.zeros((N - 1, int(K**nu), int(K**nu)))
            for n in range(N - 1):
                Normn = NormAlpha(n, int(K**nu), alphas, A, probabilities)
                for k in range(int(K**nu)):
                    for j in range(int(K**nu)):
                        xi[n, k, j] = (gammas[n, k] * A[k, j] *
                                       probabilities[n + 1, j] *
                                       betas[n + 1, j]) / (Normn * betas[n, k])
        else:

            # forward-backward algorithm
            alphas, betas, p_x = forward_backward(X, pis, probabilities, A)

            # transition probability
            xi = np.zeros((N - 1, int(K**nu), int(K**nu)))
            gammas = np.zeros((N - 1, int(K**nu)))
            for n in range(N - 1):
                for k in range(int(K**nu)):
                    for j in range(int(K**nu)):
                        xi[n, k, j] = (alphas[n, k] * A[k, j] *
                                       probabilities[n + 1, j] *
                                       betas[n + 1, j]) / p_x
                    gammas[n, k] = np.sum(xi[n, k, :])

            gammas = np.vstack((gammas, alphas[-1, :] * betas[-1, :] /
                                np.sum(alphas[-1, :] * betas[-1, :])))

        # M-step
        pis = gammas[0, :] / np.sum(gammas[0, :])
        lambdas = np.zeros(int(K**nu))
        thetas_old = thetas
        thetas = []
        emp_cov = []
        for j in range(int(K**nu)):
            Ir = ImSet(K, j, nu, r)
            for k in range(int(K**nu)):
                xi_sum = np.zeros(N - 1)
                gammas_sum = np.zeros(N - 1)

                if np.floor((j) / K) == (k) - np.floor(
                    (k) / (K**(nu - 1))) * K**(nu - 1):
                    for kk in Ir:
                        kkto = np.floor((kk) / K) + np.floor(
                            (k) / (K**(nu - 1))) * K**(nu - 1)
                        xi_sum = xi_sum + xi[:, kk, int(kkto)]
                        gammas_sum = gammas_sum + gammas[:-1, kk]
                    A[j, k] = np.sum(xi_sum) / np.sum(gammas_sum)
                else:
                    A[j, k] = 0
            Im = ImSet(K, j, nu, m)
            gammas_sum_m = np.zeros(N)
            for km in Im:
                gammas_sum_m = gammas_sum_m + gammas[:, km]

            lambdas[j] = alpha / np.sum(gammas_sum_m)
            means[j, :] = np.sum(gammas_sum_m[:, np.newaxis] * X,
                                 axis=0) / np.sum(gammas_sum_m)

            S_k = (gammas_sum_m[:, np.newaxis] *
                   (X - means[j, :])).T.dot(X -
                                            means[j, :]) / np.sum(gammas_sum_m)
            emp_cov.append(S_k)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if warm_restart and iter_ > 0:
                    thetas.append(
                        graphical_lasso(S_k,
                                        alpha=lambdas[j],
                                        init=thetas_old[j])[0])
                else:
                    thetas.append(graphical_lasso(S_k, alpha=lambdas[j])[0])

        covariances = [np.linalg.pinv(t) for t in thetas]
        probabilities = np.zeros((N, int(K**nu)))
        for n in range(N):
            for k in range(int(K**nu)):
                probabilities[n,
                              k] = multivariate_normal.pdf(X[n, :],
                                                           mean=means[k, :],
                                                           cov=covariances[k])
        # print(A)
        likelihood_old = likelihood_
        likelihood_ = compute_likelihood(gammas, pis, xi, A, probabilities)
        if verbose:
            print('Iter: %d, likelihood: %.5f, difference: %.5f' %
                  (iter_, likelihood_, np.abs(likelihood_ - likelihood_old)))
        if np.abs(likelihood_ - likelihood_old) < tol:
            break

    else:
        warnings.warn('The optimisation did not converge.')

    return thetas, means, covariances, A, pis, gammas, probabilities, alphas, betas, xi, emp_cov, likelihood_


class HHMM_GraphicalLasso(HMM_GraphicalLasso):
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

    init_params: dict, default dict()
        For each initial parameter we can specify different initialisation
        values.
        dict=(clustering=[kmeans, gmm],
              probabilities=[uniform, random_uniform, random_dirichlet])

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
                 N_memory_trans=2,
                 N_memory_emis=1,
                 warm_restart=False,
                 init_params=dict(),
                 mode='scaled',
                 repetitions=1,
                 n_jobs=-1):
        super(HHMM_GraphicalLasso,
              alpha=alpha,
              tol=tol,
              max_iter=max_iter,
              mode=mode,
              verbose=verbose,
              n_clusters=n_clusters,
              init_params=init_params,
              warm_restart=warm_restart,
              repetitions=repetitions,
              n_jobs=n_jobs)
        self.N_memory_trans = N_memory_trans
        self.N_memory_emis = N_memory_emis
        self.nu = max([N_memory_trans, N_memory_emis])

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

        N, d = X.shape
        K = self.n_clusters

        means, covariances, A, pis = _initialization(
            X, K, self.init_params, self.alpha, self.nu
        )  # TODO - probabilmente ho cancellato initialization ma non dovevo
        thetas, means, covariances, A, pis, gammas, probabilities, alphas, \
            betas, xi, emp_cov, likelihood = hhmm_graphical_lasso(
                    X,
                    A,
                    pis,
                    means,
                    covariances,
                    alpha=self.alpha,
                    max_iter=self.max_iter,
                    verbose=self.verbose,
                    mode=self.mode,
                    warm_restart=self.warm_restart,
                    tol=self.tol,
                    r=self.N_memory_trans,
                    m=self.N_memory_emis)

        self.likelihood_ = likelihood
        self.precisions_ = thetas
        self.means_ = means
        self.state_change = A
        self.pis_ = pis
        self.gammas_ = gammas
        self.probabilities_ = probabilities
        self.alphas_ = alphas
        self.labels_ = np.argmax(self.gammas_, axis=1)

        return self
