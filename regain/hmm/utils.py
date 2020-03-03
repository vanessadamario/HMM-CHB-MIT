import numpy as np
from regain.utils import structure_error
from scipy import integrate, stats
from scipy.stats import multivariate_normal
from sklearn.base import clone
from sklearn.metrics.cluster import (adjusted_mutual_info_score,
                                     contingency_matrix,
                                     homogeneity_completeness_v_measure)
from tqdm import tqdm


def alpha_heuristic(emp_cov, n_samples, gamma=0.1):
    if n_samples < 3:
        return 0
    else:
        d = np.size(emp_cov, axis=0)
        emp_var = np.diagonal(emp_cov)
        m = np.max([
            emp_var[i] * emp_var[j] for i in range(d) for j in range(i + 1, d)
        ])
        t = stats.t.pdf(gamma / (2 * d**2), n_samples - 2)
        num = t * m
        den = np.sqrt(n_samples - 2 + t**2)
        return num / den


def viterbi_path(pis, probs, A, mode):

    N, K = probs.shape

    if np.any(pis == 0):
        pis = pis + 1e-10
    if np.any(A == 0):
        A = A + 1e-10
    if np.any(probs == 0):
        probs = probs + np.min(probs[probs != 0])

    deltas = np.zeros((N, K))
    psis = np.zeros((N, K))
    S = np.zeros(N)

    if mode == 'scaled':
        deltas[0, :] = np.log(pis * probs[0, :])
    else:
        deltas[0, :] = pis * probs[0, :]

    psis[0, :] = 0

    for n in range(1, N):
        for j in range(K):
            if mode == 'scaled':
                deltas[n, j] = np.max(deltas[n - 1, :] +
                                      np.log(A[:, j])) + np.log(probs[n, j])
                psis[n, j] = np.argmax(deltas[n - 1, :] + np.log(A[:, j]))
            else:
                deltas[n, j] = np.max(deltas[n - 1, :] * A[:, j]) * probs[n, j]
                psis[n, j] = np.argmax(deltas[n - 1, :] * A[:, j])
    Pstar = np.max(deltas[N - 1, :])
    S[N - 1] = np.argmax(deltas[N - 1, :])

    for n in range(N - 2, -1, -1):
        S[n] = np.max(psis[n + 1, :])
    return S


def probability_next_point(means,
                           covariances,
                           alphas,
                           A,
                           mode,
                           state=None,
                           i=None,
                           interval=None,
                           expectations=None):
    if not mode == 'scaled':
        pX = np.sum(alphas[-1, :])
    else:
        pX = 1
    K, D = means.shape

    # interval to integrate
    if interval is None:
        interval = []
        for d in range(D):
            interval.append([
                means[state][d] - covariances[state][d][d],
                means[state][d] + covariances[state][d][d]
            ])
    prob = 0
    for k in range(K):

        # def _to_integrate(x):
        if i is None and expectations is None:
            _to_integrate = lambda *x: 1 / pX * multivariate_normal.pdf(
                x, mean=means[k], cov=covariances[k]) * np.sum(A[:, k] *
                                                               alphas[-1, :])
        #         return 1 / pX * multivariate_normal.pdf(
        #             x, mean=means[k], cov=covariances[k]) * np.sum(
        #                 A[:, k] * alphas[-1, :])
        elif expectations is None:
            _to_integrate = lambda *x: 1 / pX * x[i] * multivariate_normal.pdf(
                x, mean=means[k], cov=covariances[k]) * np.sum(A[:, k] *
                                                               alphas[-1, :])
        else:
            _to_integrate = lambda *x: 1 / pX * (x[i] - expectations[i])**2 * \
                    multivariate_normal.pdf(
                        x, mean=means[k], cov=covariances[k]) * np.sum(
                            A[:, k] * alphas[-1, :])

    # _to_integrate = lambda *x: 1 / pX * multivariate_normal.pdf(
    #      x, mean=means[k], cov=covariances[k]) * np.sum(A[:, k] * alphas[
    #          -1, :])
        res, err = integrate.nquad(_to_integrate, interval)
        prob += res

    return prob


def cross_validation(estimator, X, params=None, mode=None, n_repetitions=10):
    """
    params: dict, optional default None.
    The parameters to try, keys of the dictionaries are 'alpha' and 'n_clusters'.
    If no interval is provided default values will be cross-validate. In
    particular: alpha = np.logspace(-3,3, 10) and n_clusters=np.arange(2, 12)
    Alpha could also be passed as 'auto' in which case it is automatically
    computed with an heuristic.

    mode: string, optional default=None
    Options are:
    - 'bic' for model selection based on Bayesian Information Criterion.
    - 'stability' for model selection based on stability of the states detection
    If None both criteria are used.

    resampling_size= int or float, optional default=0.8
    The size of the subset obtained through Monte Carlo subsampling. If a float
    is provided it is used as the percentage of the total data to subsample. If
    an int is provided that amount of samples is taken each time.

    n_repetitions: int, optional default=10
    Number of times we repeate the procedure to compute mean bic or stability
    scores.
    """

    if params is None:
        alphas = np.logspace(-3, 3, 10)
        n_clusters = np.arange(2, 12)
    else:
        alphas = params.get('alpha', np.logspace(-3, 3, 10))
        n_clusters = params.get('n_clusters', np.arange(2, 12))

    N, D = X.shape
    results = {}
    for a in tqdm(alphas):
        for c in tqdm(n_clusters):
            est = clone(estimator)
            est.alpha = a
            est.n_clusters = c
            bics = []
            estimators = []
            connectivity_matrix = np.zeros((N, N))
            for i in range(n_repetitions):
                est.fit(X)
                dof = np.sum([np.count_nonzero(p) for p in est.precisions_])
                bics.append(
                    np.log(N) * ((c + 1) * (c - 1) + D * c + dof) -
                    2 * est.likelihood_)
                C = np.zeros_like(connectivity_matrix)
                for r, i in enumerate(est.labels_):
                    C[r, i] = 1
                connectivity_matrix += C.dot(C.T)
                estimators.append(est)
            connectivity_matrix /= n_repetitions

            non_diag = (np.ones(shape=(N, N)) - np.identity(N)).astype(bool)
            ravelled = connectivity_matrix[np.where(non_diag)]
            eta = np.var(ravelled)
            eta /= ((N / c - 1) / (N - 1) - ((N / c - 1) / (N - 1))**2)

            results[(a, c)] = {
                'bics': bics,
                'mean_bic': np.mean(bics),
                'std_bic': np.std(bics),
                'connectivity_matrix': connectivity_matrix,
                'dispersion_coefficient': eta,
                'estimators': estimators
            }

    scores = []
    params = []
    for k, v in results.items():
        if mode is None:
            s = v['mean_bic'] + v['dispersion_coefficient']
            results[k]['combined_score'] = s
            scores.append(s)
        elif mode == 'bic':
            scores.append(results[k]['mean_bic'])
        elif mode == 'stability':
            scores.append(results[k]['dispersion_coefficient'])
        params.append(k)
    best_params = params[np.argmin(scores)]
    return best_params, results


def results_recap(labels_true,
                  labels_pred,
                  thetas_true=None,
                  thetas_pred=None,
                  gammas_true=None,
                  gammas_pred=None):
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
        labels_true, labels_pred)
    mutual_info = adjusted_mutual_info_score(labels_true,
                                             labels_pred,
                                             average_method='arithmetic')

    results = {
        'homogeneity [0, 1]': homogeneity,
        'completeness [0, 1]': completeness,
        'v_measure [0, 1]': v_measure,
        'adjusted_mutual_info [0, 1]': mutual_info
    }

    if thetas_true is not None:
        c = contingency_matrix(labels_true, labels_pred)
        c = c / np.sum(c, axis=0)[np.newaxis, :]
        mcc = np.zeros((len(thetas_true), len(thetas_pred)))
        f1_score = np.zeros((len(thetas_true), len(thetas_pred)))
        for i, t_t in enumerate(thetas_true):
            for j, t_p in enumerate(thetas_pred):
                ss = structure_error(t_t, t_p, no_diagonal=True)
                mcc[i, j] = ss['mcc']
                f1_score[i, j] = ss['f1']

        couples = []
        aux = c.copy()
        res = []
        for i in range(len(thetas_pred)):
            couple = np.unravel_index(aux.argmax(), aux.shape)
            aux[:, couple[1]] = -np.inf
            couples.append(couple)
            res.append('Couple: ' + str(couple) + ', Probability: ' +
                       str(c[couple]) + ', MCC: ' + str(mcc[couple]) +
                       ', F1_score: ' + str(f1_score[couple]))

        results['weighted_mean_mcc [-1, 1]'] = np.sum(mcc * c) / np.sum(c),
        results['max_cluster_mean_mcc[-1,1]'] = np.sum(
            [mcc[c] for c in couples]) / len(thetas_pred),
        results['weighted_mean_f1 [0, 1]'] = np.sum(f1_score * c) / np.sum(c),
        results['max_cluster_mean_f1[0,1]'] = np.sum(
            [f1_score[c] for c in couples]) / len(thetas_pred),
        results['probabilities_clusters'] = c,
        results['max_probabilities_couples'] = res
    return results


def prepare_data_to_predict(X, p):
    N, d = X.shape
    if N <= p:
        raise ValueError('Not enough observation for ' + str(p) + 'memory')
    dataX = np.zeros((np.size(X, axis=0) - p, p * d))
    dataY = np.zeros((np.size(X, axis=0) - p, d))
    for i in range(p, np.size(X, axis=0)):
        temp = X[i - p:i, :]
        dataX[i - p, :] = X[i - p:i, :].reshape((1, np.size(temp)))
        dataY[i - p, :] = X[i, :]
    return dataX, dataY
