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
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.covariance import empirical_covariance
from sklearn.mixture import GaussianMixture
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
            betas[n, k] = np.sum(betas[n + 1, :] * probabilities[n + 1, :] *
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
            betas[n, k] = np.sum(betas[n + 1, :] * probabilities[n + 1, :] *
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


def _initialization(X, K, init_params, alpha):
    N, d = X.shape
    means = np.zeros((K, d))
    thetas = []

    # Initialization
    init_type = init_params.get('clustering', None)
    if init_type is None or str(init_type).lower() == 'kmeans':
        clusters = KMeans(n_clusters=K).fit(X).labels_
        for i, l in enumerate(np.unique(clusters)):
            means[i, :] = np.mean(X[np.where(clusters == l)[0], :], axis=0)
            emp_cov = empirical_covariance(X - means[i, :],
                                           assume_centered=True)
            thetas.append(graphical_lasso(emp_cov, alpha=alpha)[0])
            covariances = [np.linalg.pinv(t) for t in thetas]
    elif str(init_type).lower() == 'gmm':
        gmm = GaussianMixture(n_components=K).fit(X)
        means = gmm.means_
        covariances = []
        for i in range(K):
            covariances.append(gmm.covariances_[i])
    else:
        raise ValueError('Unexpected value for clusters initialisations. '
                         'Options are kmeans and ggm, found' + str(init_type))

    init_type = init_params.get('probabilities', None)
    if init_type is None or str(init_type).lower() == 'uniform':
        pis = np.ones((K, 1)) / K
        A = np.ones((K, K)) / K
    elif str(init_type).lower() == 'random_uniform':
        pis = np.random.uniform(0, 1, K)
        pis /= np.sum(pis)
        A = np.random.uniform(0, 1, shape=(K, K))
        A /= np.sum(A, axis=1)[:, np.newaxis]
    elif str(init_type).lower() == 'random_dirichlet':
        pis = np.random.dirichlet(np.ones(K), 1)
        A = np.random.dirichlet(np.ones(K), K)
    else:
        raise ValueError('Unexpected value for probabilities initialisations. '
                         'Options are uniform, random_uniform and '
                         'random_dirichlet, found' + str(init_type))
    return means, covariances, A, pis


def hmm_graphical_lasso(X,
                        A,
                        pis,
                        means,
                        covariances,
                        alpha=0.1,
                        max_iter=100,
                        mode='scaled',
                        verbose=0,
                        warm_restart=False,
                        tol=1e-3):
    K = pis.shape[0]
    N, d = X.shape
    probabilities = np.zeros((N, K))
    for n in range(N):
        for k in range(K):
            probabilities[n, k] = multivariate_normal.pdf(X[n, :],
                                                          mean=means[k, :],
                                                          cov=covariances[k])
    likelihood_ = -np.inf
    thetas = []
    for iter_ in range(max_iter):
        # E-step
        if mode == 'scaled':
            alphas, betas, cs = scaled_forward_backward(
                X, pis, probabilities, A)
            gammas = alphas * betas
            xi = np.zeros((N - 1, K, K))
            for n in range(1, N):
                for k in range(K):
                    xi[n - 1,
                       k, :] = (cs[n] * alphas[n - 1, k] *
                                probabilities[n, k] * betas[n, k] * A[k, :])
        else:
            alphas, betas = forward_backward(X, pis, probabilities, A)
            gammas = alphas * betas
            p_x = np.sum(alphas[-1, :])
            gammas = gammas / p_x
            xi = np.zeros((N - 1, K, K))
            for n in range(1, N):
                for k in range(K):
                    xi[n-1, k, :] = alphas[n-1, k]*probabilities[n, k] * \
                                  betas[n, k]*A[k, :]

        # M-step
        pis = gammas[0, :] / np.sum(gammas[0, :])
        lambdas = alpha / np.sum(gammas, axis=0)  # should be of length K
        thetas_old = thetas
        thetas = []
        for k in range(K):
            means[k, :] = np.sum(gammas[:, k][:, np.newaxis] * X,
                                 axis=0) / np.sum(gammas[:, k])
            # not sure it is like this
            S_k = (gammas[:, k][:, np.newaxis] *
                   (X - means[k, :])).T.dot(X - means[k, :]) / np.sum(
                       gammas[:, k])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if warm_restart and iter_ > 0:
                    thetas.append(
                        graphical_lasso(S_k,
                                        alpha=lambdas[k],
                                        init=thetas_old[k])[0])
                else:
                    thetas.append(graphical_lasso(S_k, alpha=lambdas[k])[0])
            for j in range(K):
                A[j, k] = np.sum(xi[:, j, k]) / np.sum(xi[:, j, :])
        covariances = [np.linalg.pinv(t) for t in thetas]
        probabilities = np.zeros((N, K))
        for n in range(N):
            for k in range(K):
                probabilities[n,
                              k] = multivariate_normal.pdf(X[n, :],
                                                           mean=means[k, :],
                                                           cov=covariances[k])
        likelihood_old = likelihood_
        likelihood_ = compute_likelihood(gammas, pis, xi, A, probabilities)

        if verbose:
            print('Iter: %d, likelihood: %.5f, difference: %.5f' %
                  (iter_, likelihood_, np.abs(likelihood_ - likelihood_old)))
        if np.abs(likelihood_ - likelihood_old) < tol:
            break

    else:
        warnings.warn('The optimisation did not converge.')

    return thetas, means, A, pis, gammas, probabilities, likelihood_


class HMM_GraphicalLasso(GraphicalLasso):
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
                 mode='scaled',
                 warm_restart=False,
                 init_params=dict(),
                 repetitions=1,
                 n_jobs=-1):
        GraphicalLasso.__init__(self, alpha=alpha, tol=tol, max_iter=max_iter)
        self.mode = mode
        self.verbose = verbose
        self.n_clusters = n_clusters
        self.init_params = init_params
        self.warm_restart = warm_restart
        self.repetitions = repetitions
        self.n_jobs = n_jobs

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

        # A = np.random.uniform(0, 1, (K, K))
        # A = A / np.sum(A, axis=0)[:, np.newaxis]
        N, d = X.shape
        K = self.n_clusters

        def _to_parallelize(X, K, init_params, alpha, max_iter, mode, verbose,
                            warm_restart, tol):
            means, covariances, A, pis = _initialization(
                X, K, init_params, alpha)
            thetas, means, A, pis, gammas, probabilities, likelihood = hmm_graphical_lasso(
                X,
                A,
                pis,
                means,
                covariances,
                alpha=alpha,
                max_iter=max_iter,
                mode=mode,
                verbose=verbose,
                warm_restart=warm_restart,
                tol=tol)
            return thetas, means, A, pis, gammas, probabilities, likelihood

        if self.repetitions == 1:
            out = [
                _to_parallelize(X, K, self.init_params, self.alpha,
                                self.max_iter, self.mode, self.verbose,
                                self.warm_restart, self.tol)
            ]
        else:
            parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)
            with parallel:
                out = parallel(
                    delayed(_to_parallelize)
                    (X, K, self.init_params, self.alpha, self.max_iter,
                     self.mode, self.verbose, self.warm_restart, self.tol)
                    for i in range(self.repetitions))

        best_repetition = np.argmax([o[-1] for o in out])
        self.all_results = out
        self.likelihood_ = out[best_repetition][-1]
        self.precisions_ = out[best_repetition][0]
        self.means_ = out[best_repetition][1]
        self.state_change = out[best_repetition][2]
        self.pis_ = out[best_repetition][3]
        self.gammas_ = out[best_repetition][4]
        self.probabilities_ = out[best_repetition][5]
        self.labels_ = np.argmax(self.gammas_, axis=1)

        return self
