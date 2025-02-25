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
from regain.covariance.graphical_lasso_ import graphical_lasso
from regain.hmm.hmm_graphical_lasso import (HMM_GraphicalLasso,
                                            compute_likelihood)
from scipy.stats import multivariate_normal
from sklearn.utils.validation import check_array


#def NormAlphaUpdate(K, alp, A, prob):
    #alp[-1, :].dot(A).dot(prob[-1,:].T)
#    s2 = 0
#   for k in range(K):
#        s1 = 0
#        for j in range(K):
#            s1 += alp[-1, j] * A[j, k]
#       s2 += s1 * prob[-1, k]
#    return s2


def _incremental_hmm_graphical_lasso(X, n_for_init, thetas, mode, means,
                                     covariances, emp_cov, A, gammas,
                                     probabilities, alphas, betas, xi, alpha,
                                     window):
    N, _ = X.shape
    for n in range(n_for_init, N):
        K = np.size(means, axis=0)
        probability = np.zeros(K)
        for k in range(K):
            probability[k] = multivariate_normal.pdf(X[n,:],
                                                     mean=means[k, :],
                                                     cov=covariances[k])

        probabilities = np.vstack((probabilities, probability))

        # E step
        alphas_t1 = np.zeros(K)
        betas_t1 = np.zeros(K)
        xi_t1 = np.zeros((K, K))

        xi_update = np.zeros((n, K, K))
        for nn in range(n-1):
            xi_update[nn, :, :] = xi[nn, :, :]
        xi = xi_update

        if mode == 'scaled':
            for k in range(K):
                alphas_t1[k] = np.sum(alphas[-1, :] * A[:, k]) * probabilities[
                    -1, k] / alphas[-1, :].dot(A).dot(probabilities[-1, :].T)
                betas_t1[k] = (betas[-1, k] * alphas[-1, :].dot(A).dot(probabilities[-1, :].T) )/ np.sum(
                        A[k, :] * probabilities[-1, :])

            gamma_t1 = alphas_t1 * betas_t1 / (np.sum(alphas_t1 * betas_t1))
            for k in range(K):
                for j in range(K):
                    xi_t1[k,
                          j] = (gammas[-1, k] * A[k, j] * probabilities[-1, j]
                                * betas_t1[k]) / (alphas[-1, :].dot(A).dot(
                                    probabilities[-1, :].T) * betas[-1, k])
            alphas = np.vstack((alphas, alphas_t1))
            betas = np.vstack((betas, betas_t1))
            gammas = np.vstack((gammas, gamma_t1))
            xi[-1, :, :] = xi_t1
        else:
            for k in range(K):
                alphas_t1[k] = probabilities[-1, k] * np.sum(
                    alphas[-1, :] * A[:, k])
                betas_t1[k] = betas[-1, k] / np.sum(
                    A[k, :] * probabilities[-1, :])

            alphas = np.vstack((alphas, alphas_t1))
            betas = np.vstack((betas, betas_t1))

            for k in range(K):
                for j in range(K):
                    xi_t1[k,
                          j] = (alphas[-1, k] * A[k, j] *
                                probabilities[-1, j] * betas[-1, j]) / (np.sum(
                                    alphas_t1 * betas_t1))
                gammas[-1, k] = np.sum(xi_t1[k, :])
            gamma_t1 = alphas_t1 * betas_t1 / (np.sum(alphas_t1 * betas_t1))
            gammas = np.vstack((gammas, gamma_t1))
            xi[-1, :, :] = xi_t1

        # M-step
        pis = gammas[0, :] / np.sum(gammas[0, :])

        if window:
            lambdas = alpha / np.sum(gammas[-n_for_init:], axis=0)
        else:
            lambdas = alpha / np.sum(gammas, axis=0)
        for k in range(K):
            if window:
                means[k, :] = (
                    np.sum(gammas[-n_for_init:, k] * X[-n_for_init:, k]) /
                    np.sum(gammas[-n_for_init:, k]))
                S_k = (gammas[-n_for_init:, k][:, np.newaxis] *
                              (X[-n_for_init:, :] - means[k, :])).T.dot(
                                    X[-n:, :] - means[k, :]) / \
                              np.sum(gammas[-n_for_init:, k])
                for j in range(K):
                    A[j, k] = np.sum(xi[-(n_for_init - 1):, j,
                                               k]) / np.sum(
                                                   gammas[-n_for_init:-1, j])
            else:
                means[k, :] = (np.sum(gammas[:-1, k]) / np.sum(gammas[:, k]) *
                                means[k, :]) +(gammas[-1, k] / np.sum(gammas[:, k]) * X[n,:])
                S_k = np.sum(gammas[:-1, k]) / np.sum(gammas[:, k]) * emp_cov[k] + gammas[-1, k] / np.sum(
                        gammas[:, k]) * (X[n,:] - means[k, :]).T.dot(X[n,:] - means[k, :])
                emp_cov[k] = S_k
                for j in range(K):
                    A[j, k] = (np.sum(gammas[:-2, j]) * A[j, k]) / (np.sum(
                                 gammas[:-1, j])) + (xi[-1, j, k] /
                                                     np.sum(gammas[:-1, j]))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                thetas[k] = graphical_lasso(S_k, alpha=lambdas[k], init=thetas[k])[0]

        covariances = [np.linalg.pinv(t) for t in thetas]
        for n in range(np.size(probabilities, axis=0)):
            for k in range(np.size(probabilities, axis=1)):
                probabilities[n, k] = multivariate_normal.pdf(
                    X[n, :], mean=means[k, :], cov=covariances[k])

        likelihood_ = compute_likelihood(gammas, pis, xi, A,probabilities)

    res = [
            thetas, means, covariances, emp_cov, A, pis, gammas,
            probabilities, alphas, betas, xi, lambdas, likelihood_
        ]
    return res




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
                 n_for_init=200,
                 window=False,
                 warm_restart=False,
                 init_params=dict(),
                 mode='scaled',
                 repetitions=1,
                 n_jobs=-1):
        super().__init__(alpha=alpha,
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

        super().fit(X_)

        if incremental:
            out = _incremental_hmm_graphical_lasso(
                X, self.n_for_init, self.precisions_, self.mode, self.means_,
                self.covariances_,self.emp_cov_, self.state_change, self.gammas_,
                self.probabilities_, self.alphas_, self.betas_, self.xi_,
                self.alpha, self.window)

            self.precisions_ = out[0]
            self.means_ = out[1]
            self.covariances_ = out[2]
            self.emp_cov_ = out[3]
            self.state_change = out[4]
            self.pis_ = out[5]
            self.gammas_ = out[6]
            self.probabilities_ = out[7]
            self.alphas_ = out[8]
            self.betas_ = out[9]
            self.xi_ = out[10]
            self.lambdas = out[11]
            self.likelihood_ = out[12]
            self.labels_ = np.argmax(self.gammas_, axis=1)

        return self
