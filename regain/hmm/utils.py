import numpy as np
import pandas as pd
from regain.utils import structure_error
from scipy import integrate, stats
from scipy.stats import multivariate_normal
from sklearn.base import clone
from sklearn.metrics.cluster import (adjusted_mutual_info_score,
                                     contingency_matrix,
                                     homogeneity_completeness_v_measure)
from tqdm import tqdm
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import randomcolor
from scipy.integrate import quad
import scipy.stats
import math
from regain.datasets.hmm import generate_nhhmm
import random


def random_subset( iterator, K ):
    result = []
    N = 0
    for item in iterator:
        N += 1
        if len( result ) < K:
            result.append( item )
        else:
            s = int(random.random() * N)
            if s == N:
                s=N-1
            if s < K:
                result[ s ] = item
    return np.sort(result)

def squared_l2_norm(x):
    return np.linalg.norm(x)**2

def Norm_inter (X,a=0,b=1):

    if X.ndim == 1:
        N = X.shape
        d = 1
    else:
        N,d = X.shape

    if d ==1:
        return (b-a)*(X-np.min(X))/(np.max(X)-np.min(X))+a
    else:
        X_norm = np.zeros((N,d))

        for j in range(d):
            temp = X[:,j]
            X_norm[:,j] = (b-a)*(temp-np.min(temp))/(np.max(temp)-np.min(temp))+a

        return X_norm

def Scaled_Mult_Epanechnikov_ker(u,utmeno1,d,h):

    cd = math.pi**(d/2)/math.gamma(d/2 + 1)
    x = (u-utmeno1)/h
    return (d+2)/(2*cd)*np.max([0,1-squared_l2_norm(x)])/h


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


def cross_validation(estimator, X, params=None, n_repetitions=10):
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
        alphas = np.linspace(1, 20, 10)
        n_clusters = np.arange(2, 12)
    else:
        alphas = params.get('alpha', np.linspace(1, 20, 10))
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

    all_alpha = []
    all_k = []
    for k, v in results.items():
        all_alpha.append(k[0])
        all_k.append(k[1])
    alpha_uni = list(set(all_alpha))
    k_uni = list(set(all_k))

    scores_k = []
    params_k = []

    for k in k_uni:

        scores_al = []
        params_al = []

        for al in alpha_uni:
            scores_al.append(results[(al,k)]['dispersion_coefficient'])
            params_al.append((al,k))
        scores_k.append(results[params_al[np.nanargmin(scores_al)]]['mean_bic'])
        params_k.append(params_al[np.nanargmin(scores_al)])

    best_params = params_k[np.nanargmin(scores_k)]

    return best_params, results



def cross_validation_higher_order(estimator, X, params=None, n_repetitions=10):
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
        alphas = np.linspace(1, 20, 10)
        n_clusters = np.arange(2, 12)
        n_memory = np.arange(1, 4)
    else:
        alphas = params.get('alpha', np.linspace(1, 20, 10))
        n_clusters = params.get('n_clusters', np.arange(2, 12))
        n_memory = params.get('N_memory_trans', np.arange(1, 4))

    N, D = X.shape
    results = {}
    for a in tqdm(alphas):
        for c in tqdm(n_clusters):
            for m in tqdm(n_memory):

                est = clone(estimator)
                est.alpha = a
                est.n_clusters = c
                est.N_memory_trans = m
                bics = []
                estimators = []
                connectivity_matrix = np.zeros((N, N))
                for i in range(n_repetitions):
                    est.fit(X)
                    dof = np.sum([np.count_nonzero(p) for p in est.precisions_])
                    bics.append(
                        np.log(N) * (c**m*(c+D)-1 + dof) -
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
                eta /= ((N / c**m - 1) / (N - 1) - ((N / c**m - 1) / (N - 1))**2) # da controllare questa formula in questo caso

                results[(a, c,m)] = {
                    'bics': bics,
                    'mean_bic': np.mean(bics),
                    'std_bic': np.std(bics),
                    'connectivity_matrix': connectivity_matrix,
                    'dispersion_coefficient': eta,
                    'estimators': estimators
                }

    all_alpha = []
    all_k = []
    all_n = []
    for k, v in results.items():
        all_alpha.append(k[0])
        all_k.append(k[1])
        all_n.append(k[2])
    alpha_uni = list(set(all_alpha))
    k_uni = list(set(all_k))
    nu_uni = list(set(all_n))

    scores_nu = []
    params_nu = []

    for nu in nu_uni:

        scores_k = []
        params_k = []

        for k in k_uni:

            scores_al = []
            params_al = []

            for al in alpha_uni:
                scores_al.append(results[(al,k,nu)]['dispersion_coefficient'])
                params_al.append((al,k,nu))

            scores_k.append(results[params_al[np.nanargmin(scores_al)]]['mean_bic'])
            params_k.append(params_al[np.nanargmin(scores_al)])

        scores_nu.append(results[params_k[np.nanargmin(scores_k)]]['mean_bic'])
        params_nu.append(params_k[np.nanargmin(scores_k)])

    best_params = params_nu[np.nanargmin(scores_nu)]

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
        results['max_cluster_mean_mcc[-1,1]'] = np.sum([mcc[c] for c in couples]) / len(thetas_pred),
        results['weighted_mean_f1 [0, 1]'] = np.sum(f1_score * c) / np.sum(c),
        results['max_cluster_mean_f1[0,1]'] = np.sum([f1_score[c] for c in couples]) / len(thetas_pred),
        results['probabilities_clusters'] = c,
        results['max_probabilities_couples'] = res
    return results




def names(var_names,couple):
    if len(var_names) == 0:
        return 'var'+str(couple[0])+'- var'+str(couple[1])
    else:
        return str(var_names[couple[0]])+'-'+str(var_names[couple[1]])



def cov2corr(cov, return_std=False):
    cov = np.asanyarray(cov)
    std_ = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std_, std_)
    if return_std:
        return corr, std_
    else:
        return corr


def corr_plot(corrs,sizeplotx=20,sizeploty=10,N_st = 2,mem = 2, labels=None, covcorr=False, numbers=True):
    if covcorr:
        correlations = []
        for k in range(len(corrs)):
            correlations.append(cov2corr(corrs[k]))
        corrs = correlations

    N_plots = len(corrs)

    N_per_rows = 2
    N_rows = int(np.ceil(N_plots / N_per_rows))
    f, axes = plt.subplots(N_rows, N_per_rows, figsize=(sizeplotx, sizeploty))

    count = 0


    if labels is None:
        for i in range(N_rows):
            for j in range(N_per_rows):
                if N_rows == 1:
                    sns.heatmap(corrs[count],
                                annot=numbers,
                                ax=axes[j])
                    axes[j].set_title('Correlation matrix for cluster '+str(count))
                else:
                    sns.heatmap(corrs[count], annot=numbers,
                                ax=axes[i, j])

                    axes[i, j].set_title('Correlation matrix for cluster ' + str(count))


                count += 1
                if count == N_plots:
                    break

    else:
        for i in range(N_rows):
            for j in range(N_per_rows):
                if N_rows == 1:
                    sns.heatmap(corrs[count],
                                annot=numbers,
                                xticklabels=labels,
                                yticklabels=labels,
                                ax=axes[j])
                    axes[j].set_title('Correlation matrix for cluster '+str(count))
                else:
                    sns.heatmap(corrs[count], annot=numbers,
                                xticklabels=labels,
                                yticklabels=labels,
                                ax=axes[i, j])
                    axes[i, j].set_title('Correlation matrix for cluster ' + str(count))

                count += 1
                if count == N_plots:
                    break
    plt.show()



def plot_results_cluster(Data, clusters,N_st = 2,mem = 2, Dates = None, ts_labels = None):


    rand_color = randomcolor.RandomColor()
    fig, ax = plt.subplots(figsize=(15, 10))

    # Draw data

    for i in range(Data.shape[1]):
        if Dates is None:

            if ts_labels is None:
                ax.plot(Data[1:, i], label='ts ' + str(i))
            else:
                ax.plot(Data[1:, i], label=ts_labels[i])

        else:

            if ts_labels is None:
                ax.plot(Dates, Data[1:, i], label='ts '+str(i))
            else:
                ax.plot(Dates, Data[1:, i], label=ts_labels[i])

    # Draw shaded regions to highlight clusters
    N_clusters = np.size(np.unique(clusters))

    for k in range(N_clusters):
        if Dates is None:

            ax.fill_between(np.arange(np.size(Data[1:, :],axis=0)),0, 1, where=clusters == k,
                            color=rand_color.generate(), alpha=0.5, transform=ax.get_xaxis_transform(),
                            label='cluster ' + str(k))
        else:

            ax.fill_between(Dates, 0, 1, where=clusters == k,
                                color=rand_color.generate(), alpha=0.5, transform=ax.get_xaxis_transform(),
                                label='cluster ' + str(k))


    plt.legend(ncol=3)
    plt.show()



def cluster_returns_recap(means, covariances, labels=None):

    N_ts = np.size(means[0])
    mean_std = []
    for k in range(len(covariances)):

        temp = np.sqrt(covariances[k].diagonal())
        for n in range(N_ts):
            mean_std_row = []
            # cluster
            mean_std_row.append(str(k))
            # time series
            if labels is None:
                mean_std_row.append('ts '+str(n))
            else:
                mean_std_row.append(labels[n])
            # mean
            mean_std_row.append(means[k][n])
            # std
            mean_std_row.append(temp[n])
            # prob positive trend
            if means[k][n] + 3*temp[n]< 0:
                mean_std_row.append(0)
            else:
                f = lambda x: scipy.stats.norm.pdf(x, means[k][n], temp[n])
                mean_std_row.append(quad(f, 0, means[k][n] + 3*temp[n])[0]*100)

            # prob negative trend
            if means[k][n] - 3*temp[n]> 0:
                mean_std_row.append(0)
            else:
                f = lambda x: scipy.stats.norm.pdf(x, means[k][n], temp[n])
                mean_std_row.append(quad(f, means[k][n] - 3*temp[n],0)[0]*100)

            mean_std.append(mean_std_row)



    df_recap = pd.DataFrame(mean_std, columns=['Cluster','TS', 'mean', 'std', 'Prob positive return %', 'Prob negative return %' ])

    return df_recap


def conversion_hhmm_results(labels, gammas, thetas, n_states, order_hmm):

    N_samples,_ = gammas.shape
    N_rep = int(n_states ** order_hmm / n_states)
    stateseq = np.repeat(np.arange(n_states), N_rep)

    conv_label = np.zeros(N_samples)
    for n in range(N_samples):
        conv_label[n] = stateseq[labels[n]]

    conversion_prec = []
    for k in range(n_states):
        conversion_prec.append(thetas[np.where(stateseq == k)[0][0]])

    conv_gamma = np.zeros((N_samples, n_states))

    for n in range(N_samples):
        for k in range(n_states):
            indk = np.where(stateseq == k)[0]
            tempsum = 0
            for kk in indk:
                tempsum += gammas[n, kk]

            conv_gamma[n, k] = tempsum

    return conv_label, conversion_prec, conv_gamma

def cross_validation_nhhmm(estimator, X,U, params=None, n_repetitions=10,M=20,Nd = 0.05):
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
        alphas = np.linspace(1, 20, 10)
        n_clusters = np.arange(2, 12)
        bandwidths = np.linspace(0, 10, 30)
    else:
        alphas = params.get('alpha', np.linspace(1, 20, 10))
        n_clusters = params.get('n_clusters', np.arange(2, 12))
        bandwidths = params.get('bandwidths', np.linspace(0, 10, 30))

    N, D = X.shape
    results = {}
    for a in tqdm(alphas):
        for c in tqdm(n_clusters):
            for h in tqdm(bandwidths):
                est = clone(estimator)
                est.alpha = a
                est.n_clusters = c
                est.bandwidth = h
                bics = []
                estimators = []
                connectivity_matrix = np.zeros((N, N))

                lik_M_mean = 0
                for m_seq in range(M):
                    est.fit(X,U,Cel = True, N_del = Nd)
                    lik_M_mean += est.likelihood_/M

                for i in range(n_repetitions):
                    est.fit(X,U)
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

                results[(a, c,h)] = {
                    'bics': bics,
                    'mean_bic': np.mean(bics),
                    'std_bic': np.std(bics),
                    'connectivity_matrix': connectivity_matrix,
                    'dispersion_coefficient': eta,
                    'estimators': estimators,
                    'Celeux_method': lik_M_mean
                }

    all_alpha = []
    all_k = []
    all_h = []
    for k, v in results.items():
        all_alpha.append(k[0])
        all_k.append(k[1])
        all_h.append(k[2])
    alpha_uni = list(set(all_alpha))
    k_uni = list(set(all_k))
    h_uni = list(set(all_h))

    scores_k = []
    params_k = []

    for k in k_uni:

        scores_al = []
        params_al = []

        for al in alpha_uni:

            scores_h = []
            params_h = []

            for hh in h_uni:
                scores_h.append(results[(al, k,hh)]['Celeux_method'])
                params_h.append((al, k,hh))

            scores_al.append(results[params_h[np.nanargmax(scores_h)]]['dispersion_coefficient'])
            params_al.append(params_h[np.nanargmax(scores_h)])

        scores_k.append(results[params_al[np.nanargmin(scores_al)]]['mean_bic'])
        params_k.append(params_al[np.nanargmin(scores_al)])

    best_params = params_k[np.nanargmin(scores_k)]

    return best_params, results
