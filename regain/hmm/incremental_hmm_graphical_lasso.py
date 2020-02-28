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
from regain.hmm.hmm_graphical_lasso import HMM_GraphicalLasso
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.covariance import empirical_covariance
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_array


def NormAlphaUpdate(K, alp, A, prob):
    #alp[-1, :].dot(A).dot(prob[-1,:].T)
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
        Normn = alphas[n, :].dot(A).dot(probabilities[n + 1, :].T)
        for k in range(K):
            alphas[n + 1, k] = np.sum(
                alphas[n, :] * A[:, k]) * probabilities[n + 1, k] / Normn

    # Initial backward probabilities
    betas[-1, :] = 1

    # recursive backward algorithm
    for n in range(N - 2, -1, -1):
        Normn = alphas[n, :].dot(A).dot(probabilities[n + 1, :].T)
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


def compute_likelihood(gammas, pis, xi, A, probabilities):
    N, K = probabilities.shape
    if np.any(pis == 0):
        pis = pis + 1e-10
    if np.any(A == 0):
        A = A + 1e-10
    aux = np.sum(gammas[0, :] * np.log(pis))
    for n in range(N - 1):
        aux += np.sum(xi[n, :, :] * np.log(A))

    if np.any(probabilities == 0):
        probabilities = probabilities + np.min(
            probabilities[probabilities != 0])

    aux += np.sum(gammas * np.log(probabilities))
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
        A = np.random.uniform(0, 1, (K, K))
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
                        tol=5e-3):
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
            alphas, betas = scaled_forward_backward(X, pis, probabilities, A)
            gammas = alphas * betas
            for n in range(np.size(gammas, axis=0)):
                gammas[n, :] = gammas[n, :] / np.sum(
                    alphas[n, :] * betas[n, :])
            xi = np.zeros((N - 1, K, K))
            for n in range(N - 1):
                Normn = alphas[n, :].dot(A).dot(probabilities[n + 1, :].T)
                for k in range(K):
                    for j in range(K):
                        xi[n, k, j] = (gammas[n, k] * A[k, j] *
                                       probabilities[n + 1, j] *
                                       betas[n + 1, j]) / (Normn * betas[n, k])
        else:

            # forward-backward algorithm
            alphas, betas, p_x = forward_backward(X, pis, probabilities, A)

            # transition probability
            xi = np.zeros((N - 1, K, K))
            gammas = np.zeros((N - 1, K))
            for n in range(N - 1):
                for k in range(K):
                    for j in range(K):
                        xi[n, k, j] = (alphas[n, k] * A[k, j] *
                                       probabilities[n + 1, j] *
                                       betas[n + 1, j]) / p_x
                    gammas[n, k] = np.sum(xi[n, k, :])

            gammas = np.vstack((gammas, alphas[-1, :] * betas[-1, :] /
                                np.sum(alphas[-1, :] * betas[-1, :])))

        # M-step
        pis = gammas[0, :] / np.sum(gammas[0, :])
        lambdas = alpha / np.sum(gammas, axis=0)  # should be of length K
        thetas_old = thetas
        thetas = []
        emp_cov = []
        for k in range(K):
            for j in range(K):
                A[j, k] = np.sum(xi[:, j, k]) / np.sum(gammas[:-1, j])
            means[k, :] = np.sum(gammas[:, k][:, np.newaxis] * X,
                                 axis=0) / np.sum(gammas[:, k])
            S_k = (gammas[:, k][:, np.newaxis] *
                   (X - means[k, :])).T.dot(X - means[k, :]) / np.sum(
                       gammas[:, k])
            emp_cov.append(S_k)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if warm_restart and iter_ > 0:
                    thetas.append(
                        graphical_lasso(S_k,
                                        alpha=lambdas[k],
                                        init=thetas_old[k])[0])
                else:
                    thetas.append(graphical_lasso(S_k, alpha=lambdas[k])[0])

        covariances = [np.linalg.pinv(t) for t in thetas]
        probabilities = np.zeros((N, K))
        for n in range(N):
            for k in range(K):
                probabilities[n,
                              k] = multivariate_normal.pdf(X[n, :],
                                                           mean=means[k, :],
                                                           cov=covariances[k])
        #print(A)
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


def _incremental_hmm_graphical_lasso(
    X,
    n_for_init,
):
    N, _ = X.shape
    for n in range(self.n_for_init, N):
        thetas, means, covariances, emp_cov, A, pis, gammas, probabilities, alphas, betas, xi, likelihood = Param_Update(
            X[:n, :], X[n, :], thetas, self.mode, means, covariances, emp_cov,
            A, gammas, probabilities, alphas, betas, xi, self.alpha,
            self.Sliding_window, self.N_init_HMM)

    return likelihood, thetas, means, A, pis, gammas, probabilities, alphas


def Param_Update(X_old, X_new, thetas, mode, means, covariances, emp_cov, A,
                 gammas, probabilities, alphas, betas, xi, alpha,
                 Sliding_window, N_Sliding_window):

    X = np.vstack((X_old, X_new))
    K = np.size(means, axis=0)
    probability = np.zeros(K)
    for k in range(K):
        probability[k] = multivariate_normal.pdf(X_new,
                                                 mean=means[k, :],
                                                 cov=covariances[k])

    probabilities = np.vstack((probabilities, probability))

    alphas_t1 = np.zeros(K)
    betas_t1 = np.zeros(K)
    xi_t1 = np.zeros((K, K))
    xi_update = np.zeros((X.shape[0] - 1, K, K))
    for n in range(xi.shape[0]):
        xi_update[n, :, :] = xi[n, :, :]

    if mode == 'scaled':
        for k in range(K):
            alphas_t1[k] = np.sum(alphas[-1, :] * A[:, k]) * probabilities[
                -1, k] / alphas[-1, :].dot(A).dot(probabilities[-1, :].T)
            betas_t1[k] = (betas[-1, k] * NormAlphaUpdate(
                K, alphas, A, probabilities)) / np.sum(
                    A[k, :] * probabilities[-1, :])

        gamma_t1 = alphas_t1 * betas_t1 / (np.sum(alphas_t1 * betas_t1))
        for k in range(K):
            for j in range(K):
                xi_t1[k, j] = (gammas[-1, k] * A[k, j] * probabilities[-1, j] *
                               betas_t1[k]) / (alphas[-1, :].dot(A).dot(
                                   probabilities[-1, :].T) * betas[-1, k])
        alphas_update = np.vstack((alphas, alphas_t1))
        betas_update = np.vstack((betas, betas_t1))
        gammas_update = np.vstack((gammas, gamma_t1))
        xi_update[-1, :, :] = xi_t1
    else:
        for k in range(K):
            alphas_t1[k] = probabilities[-1, k] * np.sum(
                alphas[-1, :] * A[:, k])
            betas_t1[k] = betas[-1, k] / np.sum(A[k, :] * probabilities[-1, :])

        alphas_update = np.vstack((alphas, alphas_t1))
        betas_update = np.vstack((betas, betas_t1))

        for k in range(K):
            for j in range(K):
                xi_t1[k, j] = (alphas[-1, k] * A[k, j] * probabilities[-1, j] *
                               betas_update[-1, j]) / (np.sum(
                                   alphas_t1 * betas_t1))
            gammas[-1, k] = np.sum(xi_t1[k, :])
        gamma_t1 = alphas_t1 * betas_t1 / (np.sum(alphas_t1 * betas_t1))
        gammas_update = np.vstack((gammas, gamma_t1))
        xi_update[-1, :, :] = xi_t1

    # M-step

    pis_update = gammas_update[0, :] / np.sum(gammas_update[0, :])

    if Sliding_window:
        lambdas = alpha / np.sum(gammas_update[-N_Sliding_window:],
                                 axis=0)  # should be of length K
    else:
        lambdas = alpha / np.sum(gammas_update,
                                 axis=0)  # should be of length K
    thetas_update = []
    means_update = np.zeros((np.size(means, axis=0), np.size(means, axis=1)))
    emp_cov_update = []
    A_update = np.zeros((np.size(A, axis=0), np.size(A, axis=1)))
    for k in range(K):
        if Sliding_window:
            means_update[k, :] = (np.sum(gammas_update[-N_Sliding_window:, k] *
                                         X[-N_Sliding_window:, k]) /
                                  np.sum(gammas_update[-N_Sliding_window:, k]))
        else:
            means_update[k, :] = (np.sum(gammas_update[:-1, k]) / np.sum(gammas_update[:, k]) * means[k, :]) + \
                                 (gammas_update[-1, k] / np.sum(gammas_update[:, k]) * X_new)
        if Sliding_window:
            S_k_update = (gammas_update[-N_Sliding_window:, k][:, np.newaxis] * ( X[-N_Sliding_window:, :]- means_update[k, :])).T.dot(X[-N_Sliding_window:, :] - means_update[k, :]) / \
                         np.sum(gammas_update[-N_Sliding_window:, k])
        else:
            S_k_update = np.sum(gammas_update[:-1, k]) / np.sum(
                gammas_update[:, k]) * emp_cov[k] + gammas_update[
                    -1, k] / np.sum(gammas_update[:, k]) * (
                        X_new - means_update[k, :]).T.dot(X_new -
                                                          means_update[k, :])
        emp_cov_update.append(S_k_update)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            thetas_update.append(
                graphical_lasso(S_k_update, alpha=lambdas[k],
                                init=thetas[k])[0])
        for j in range(K):
            if Sliding_window:
                A_update[j, k] = np.sum(
                    xi_update[-(N_Sliding_window - 1):, j, k]) / np.sum(
                        gammas_update[-N_Sliding_window:-1, j])
            else:
                A_update[j, k] = (np.sum(gammas_update[:-2, j]) * A[j, k]) / (
                    np.sum(gammas_update[:-1, j])) + (
                        xi_update[-1, j, k] / np.sum(gammas_update[:-1, j]))
    covariances_update = [np.linalg.pinv(t) for t in thetas_update]
    probabilities_update = np.zeros(
        (np.size(probabilities, axis=0), np.size(probabilities, axis=1)))
    for n in range(np.size(probabilities, axis=0)):
        for k in range(np.size(probabilities, axis=1)):
            probabilities_update[n, k] = multivariate_normal.pdf(
                X[n, :], mean=means_update[k, :], cov=covariances_update[k])
    likelihood_ = compute_likelihood(gammas_update, pis_update, xi_update,
                                     A_update, probabilities_update)

    return thetas_update, means_update, covariances_update, emp_cov_update, A_update, pis_update, gammas_update, probabilities_update, alphas_update, betas_update, xi_update, likelihood_


class Incremental_HMM_GraphicalLasso(HMM_GraphicalLasso):
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
                 n_for_init=50,
                 window=False,
                 warm_restart=False,
                 init_params=dict(),
                 mode='scaled',
                 repetitions=1,
                 n_jobs=-1):
        super(Incremental_HMM_GraphicalLasso,
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
        self.n_for_init = n_for_init
        self.window = window

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
        incremental = False
        if N > self.n_for_init:
            X_ = X[:self.n_for_init, :]
            incremental = True
        else:
            X_ = X

        super.fit(Incremental_HMM_GraphicalLasso, X_)

        if incremental:
            out = _incremental_hmm_graphical_lasso(
                X, self.n_for_init, self.precisions_, self.mode, self.means_,
                self.covariances_, self.state_change, self.gammas_,
                self.probabilities_, self.alphas_, self.betas_, self.xi_,
                self.alpha, self.window)
            self.likelihood_ = out[0]
            self.precisions_ = out[1]
            self.means_ = out[2]
            self.state_change = out[3]
            self.pis_ = out[4]
            self.gammas_ = out[5]
            self.probabilities_ = out[6]
            self.alphas_ = out[7]
            self.labels_ = np.argmax(self.gammas_, axis=1)

        return self
