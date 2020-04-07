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
from regain.hmm.hmm_graphical_lasso import (HMM_GraphicalLasso,
                                            compute_likelihood,
                                            scaled_forward_backward,
                                            forward_backward)
from regain.hmm.utils import probability_next_point, viterbi_path
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.covariance import empirical_covariance
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_array
from regain.datasets.hmm import generate_hmm


def TransMatrixInit(method,K, nu):

    A = np.zeros((K**nu,K**nu))
    for i in range(K**nu):
        index_fill = np.zeros(K**nu, bool)
        for j in range(K ** nu):
            index_fill[j] = np.floor((i)/K)==(j)-np.floor((j)/(K**(nu-1)))*K**(nu-1)
        if method is None or str(method).lower() == 'uniform':
            A[i,index_fill] = 1/np.sum(index_fill)
        elif str(method).lower() == 'random_uniform':
            draw = np.random.uniform(0, 1, np.sum(index_fill))
            draw /= np.sum(draw)
            A[i, index_fill] = draw
        elif str(method).lower() == 'random_dirichlet':
            A[i, index_fill] = np.random.dirichlet(np.ones(np.sum(index_fill)), 1)
        else:
            raise ValueError('Unexpected value for probabilities initialisations. '
                             'Options are uniform, random_uniform and '
                             'random_dirichlet, found' + str(method))
    return A

def ImSet(K,i,nu,m):
    lower = np.floor(i / (K ** (nu - m))) * K ** (nu - m)
    upper = (np.floor(i / (K ** (nu - m))) + 1) * K ** (nu - m)
    return list(range(int(lower), int(upper)))


def _initialization(X, K, init_params, alpha,nu ):
    N, d = X.shape
    means = np.zeros((K**nu, d))
    thetas = []

    # Initialization
    init_type = init_params.get('clustering', None)
    if init_type is None or str(init_type).lower() == 'kmeans':
        clusters = KMeans(n_clusters=K**nu).fit(X).labels_
        for i, l in enumerate(np.unique(clusters)):
            means[i, :] = np.mean(X[np.where(clusters == l)[0], :], axis=0)
            emp_cov = empirical_covariance(X - means[i, :],
                                           assume_centered=True)
            thetas.append(graphical_lasso(emp_cov, alpha=alpha)[0])
            covariances = [np.linalg.pinv(t) for t in thetas]
    elif str(init_type).lower() == 'gmm':
        gmm = GaussianMixture(n_components=K**nu).fit(X)
        means = gmm.means_
        covariances = []
        for i in range(K**nu):
            covariances.append(gmm.covariances_[i])
    else:
        raise ValueError('Unexpected value for clusters initialisations. '
                         'Options are kmeans and ggm, found' + str(init_type))

    init_type = init_params.get('probabilities', None)
    A = TransMatrixInit(init_type, K, nu)
    if init_type is None or str(init_type).lower() == 'uniform':
        pis = np.ones((K**nu, 1)) / K**nu
    elif str(init_type).lower() == 'random_uniform':
        pis = np.random.uniform(0, 1, K**nu)
        pis /= np.sum(pis)
    elif str(init_type).lower() == 'random_dirichlet':
        pis = np.random.dirichlet(np.ones(K**nu), 1)
    else:
        raise ValueError('Unexpected value for probabilities initialisations. '
                         'Options are uniform, random_uniform and '
                         'random_dirichlet, found' + str(init_type))
    return means, covariances, A, pis


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

    nu=max([m,r])
    K = int(pis.shape[0]**(1/nu))
    N, d = X.shape
    probabilities = np.zeros((N, int(K**nu)))
    for n in range(N):
        for k in range(int(K**nu)):
            try:
                probabilities[n, k] = multivariate_normal.pdf(X[n, :], mean=means[k, :], cov=covariances[k])
            except:
                out = np.repeat(np.nan, 13)
                return out

    likelihood_ = -np.inf
    thetas = []
    for iter_ in range(max_iter):
        # E-step
        if mode == 'scaled':
            alphas, betas,cs = scaled_forward_backward(X, pis, probabilities, A)
            gammas = alphas * betas
            for n in range(np.size(gammas, axis=0)):
                gammas[n, :] = gammas[n, :] / np.sum(alphas[n, :] * betas[n, :])
            xi = np.zeros((N - 1, int(K**nu), int(K**nu)))
            for n in range(N - 1):
                Normn = alphas[n, :].dot(A).dot(probabilities[n+1, :].T)
                for k in range(int(K**nu)):
                    for j in range(int(K**nu)):
                        xi[n, k, j] = (gammas[n, k] * A[k, j] * probabilities[n + 1, j] * betas[n + 1, j]) / (
                                    Normn * betas[n, k])
        else:

            # forward-backward algorithm
            alphas, betas, p_x = forward_backward(X, pis, probabilities, A)

            # transition probability
            xi = np.zeros((N - 1, int(K**nu), int(K**nu)))
            gammas = np.zeros((N - 1, int(K**nu)))
            for n in range(N - 1):
                for k in range(int(K**nu)):
                    for j in range(int(K**nu)):
                        xi[n, k, j] = (alphas[n, k] * A[k, j] * probabilities[n + 1, j] * betas[n + 1, j]) / p_x
                    gammas[n, k] = np.sum(xi[n, k, :])

            gammas = np.vstack((gammas, alphas[-1, :] * betas[-1, :] / np.sum(alphas[-1, :] * betas[-1, :])))

        # M-step
        pis = gammas[0, :] / np.sum(gammas[0, :])
        lambdas = np.zeros(int(K**nu))
        thetas_old = thetas
        thetas = []
        emp_cov = []
        for j in range(int(K**nu)):
            Ir = ImSet(K,j,nu,r)
            for k in range(int(K**nu)):
                xi_sum = np.zeros(N-1)
                gammas_sum =  np.zeros(N-1)

                if np.floor((j)/K)==(k)-np.floor((k)/(K**(nu-1)))*K**(nu-1):
                    for kk in Ir:
                        kkto = np.floor((kk)/K)+np.floor((k)/(K**(nu-1)))*K**(nu-1)
                        xi_sum = xi_sum + xi[:, kk, int(kkto)]
                        gammas_sum = gammas_sum + gammas[:-1,kk]
                    A[j, k] = np.sum(xi_sum) / np.sum(gammas_sum)
                else:
                    A[j, k] = 0
            Im = ImSet(K, j, nu, m)
            gammas_sum_m = np.zeros(N)
            for km in Im:
                gammas_sum_m = gammas_sum_m + gammas[:, km]

            lambdas[j] = alpha/ np.sum(gammas_sum_m)
            means[j, :] = np.sum(gammas_sum_m[:, np.newaxis] * X, axis=0) / np.sum(gammas_sum_m)

            S_k = (gammas_sum_m[:, np.newaxis]* (X - means[j, :])).T.dot(X - means[j, :]) / np.sum(gammas_sum_m)
            emp_cov.append(S_k)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if warm_restart and iter_ > 0:

                    try:
                        thetas.append(
                        graphical_lasso(S_k,
                                        alpha=lambdas[j],
                                        init=thetas_old[j])[0])
                    except:
                        out = np.repeat(np.nan,13)
                        return out

                else:

                    try:
                        thetas.append(graphical_lasso(S_k, alpha=lambdas[j])[0])
                    except:
                        out = np.repeat(np.nan,13)
                        return out


        covariances = [np.linalg.pinv(t) for t in thetas]
        #print('iter',iter_,'cov',covariances)
        probabilities = np.zeros((N, int(K**nu)))
        for n in range(N):
            for k in range(int(K**nu)):
                try:
                    probabilities[n, k] = multivariate_normal.pdf(X[n, :], mean=means[k, :], cov=covariances[k])
                except:
                    out = np.repeat(np.nan, 13)
                    return out

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

    out = [
        thetas, means, covariances, A, pis, gammas, probabilities, alphas, betas, xi, emp_cov, lambdas,likelihood_
    ]

    return out



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
                 N_memory_trans = 2,
                 N_memory_emis=1,
                 warm_restart=False,
                 init_params=dict(),
                 mode='scaled',
                 repetitions=1,
                 n_jobs=-1):
        GraphicalLasso.__init__(self, alpha=alpha, tol=tol, max_iter=max_iter)
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
        self.N_memory_trans = N_memory_trans
        self.N_memory_emis = N_memory_emis
        self.nu = max([N_memory_trans,N_memory_emis])

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

        def _to_parallelize(X, K, init_params, alpha, max_iter, mode, verbose,
                            warm_restart, tol,nu,N_memory_trans,N_memory_emis):
            means, covariances, A, pis = _initialization(X, K, init_params, alpha, nu)

            thetas, means, covariances, A, pis, gammas, probabilities, alphas,\
            betas, xi, emp_cov, lambdas,likelihood_ = hhmm_graphical_lasso(
                                                                          X,
                                                                          A,
                                                                          pis,
                                                                          means,
                                                                          covariances,
                                                                          alpha=alpha,
                                                                          max_iter=max_iter,
                                                                          verbose=verbose,
                                                                          mode=mode,
                                                                          warm_restart=warm_restart,
                                                                          tol=tol,
                                                                          r=N_memory_trans,
                                                                          m=N_memory_emis
                                                                          )
            return thetas, means, covariances, A, pis, gammas, probabilities,\
                   alphas, betas, xi, emp_cov, lambdas,likelihood_

        if self.repetitions == 1:
            out = [_to_parallelize(X, K, self.init_params, self.alpha, self.max_iter,
                                self.mode, self.verbose, self.warm_restart, self.tol,
                                self.nu,self.N_memory_trans,self.N_memory_emis)]
        else:
            parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)
            with parallel:
                out = parallel(
                    delayed(_to_parallelize)
                    (X, K, self.init_params, self.alpha, self.max_iter,
                     self.mode, self.verbose, self.warm_restart, self.tol,self.nu,self.N_memory_trans,self.N_memory_emis)
                    for i in range(self.repetitions))

        best_repetition = np.nanargmax([o[-1] for o in out])
        self.all_results = out
        self.likelihood_ = out[best_repetition][-1]
        self.precisions_ = out[best_repetition][0]
        self.means_ = out[best_repetition][1]
        self.covariances_ = out[best_repetition][2]
        self.state_change = out[best_repetition][3]
        self.pis_ = out[best_repetition][4]
        self.gammas_ = out[best_repetition][5]
        self.probabilities_ = out[best_repetition][6]
        self.alphas_ = out[best_repetition][7]
        self.betas_ = out[best_repetition][8]
        self.xi_ = out[best_repetition][9]
        self.emp_cov_ = out[best_repetition][10]
        self.lambdas = out[best_repetition][11]
        self.labels_ = np.argmax(self.gammas_, axis=1)

        return self

    def predict(self, X, method='viterbi',conf_inter = False):

        if method == 'viterbi':
            results = viterbi_path(self.pis_, self.probabilities_,
                                   self.state_change, self.mode)
            state = np.random.choice(np.arange(self.n_clusters),
                                     replace=True,
                                     p=self.state_change[int(results[-1]), :])
            sample = np.random.multivariate_normal(self.means_[state],
                                                   self.covariances_[state], 1)

            if conf_inter:
                prob = probability_next_point(self.means_,
                                              self.covariances_,
                                              self.alphas_,
                                              self.state_change,
                                              self.mode,
                                              state=state)
                prediction = dict(pred=sample,
                              means=self.means_[state],
                              stds=np.sqrt(
                                  self.covariances_[state].diagonal()),
                              prob_sample=prob)
            else:
                prediction = dict(pred=sample,
                              means=self.means_[state],
                              stds=np.sqrt(
                                  self.covariances_[state].diagonal()))


        elif method == 'hassan':
            pXn = np.sum(self.probabilities_ * self.gammas_, axis=1)
            delpX_n = np.abs(pXn - pXn[-1])
            n_sim = np.argmin(delpX_n[:-1])
            state = np.argmax(self.gammas_[n_sim, :])
            delta = X[n_sim + 1, :] - X[n_sim, :]
            sample = X[-1, :] + delta

            if conf_inter:
                prob = probability_next_point(self.means_, self.covariances_,
                                          self.alphas_, self.state_change,
                                           self.mode, state)
                prediction = dict(pred=sample,
                              means=self.means_[state],
                              stds=np.sqrt(
                                  self.covariances_[state].diagonal()),
                              prob_sample=prob)
            else:
                prediction = dict(pred=sample,
                              means=self.means_[state],
                              stds=np.sqrt(
                                  self.covariances_[state].diagonal()))
        return prediction
