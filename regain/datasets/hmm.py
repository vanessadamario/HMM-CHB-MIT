import operator as op
from functools import reduce

import numpy as np
from regain.datasets.gaussian import make_starting
from scipy import linalg
from scipy.stats import bernoulli
from sklearn.datasets import make_spd_matrix


def isPSD(A, tol=1e-8):
    E = np.linalg.eigvalsh(A)
    return np.all(E > -tol)


def ncr(n, r):
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer / denom


def generate_precision_matrix(dim):
    Theta = np.identity(dim)
    for i in range(dim):
        for j in range(i + 1, dim):
            Theta[i, j] = bernoulli.rvs(0.5, size=1)
            Theta[j, i] = Theta[i, j]
    return Theta


def generate_complementary_precisions_matrix(dim, N_states):

    # Initial checks
    if dim <= 2:
        raise ValueError('With only 2 observed variables it is not possible '
                         'to create different complementary states.')
    # possible combinations
    N_tot_comb = 0
    for i in range(1, dim + 1):
        N_tot_comb += ncr(dim, i)
    if dim <= 2 or N_states > N_tot_comb:
        raise ValueError(
            'The number of states is too big, reduce the value of N_states')

    Theta = generate_precision_matrix(dim)
    precisions = [Theta]
    covariances = [linalg.pinv(Theta)]
    counts = 1
    while counts < N_states:
        Theta = generate_precision_matrix(dim)
        N_equal = []
        for j in range(len(precisions)):
            N_equal.append((Theta == precisions[j]).all())
        if np.sum(N_equal) == 0:
            precisions.append(Theta)
            covariances.append(linalg.pinv(Theta))
            counts += 1
    return covariances, precisions


def generate_hmm(n_samples=100,
                 n_states=5,
                 n_dim_obs=10,
                 mode_precisions='complementary',
                 mode_mean='Normal',
                 transition_type='fixed',
                 smooth_transition=10,
                 min_transition_window=1,
                 max_transition_window=20,
                 sigma=0.7,
                 stability_factor=12,
                 order_hmm = 1,
                 **kwargs):
    """
    transition_type: string, optional default='sudden'
    Possible values are 'fixed', 'fixed_smooth', 'random_smooth', 'random'
    smooth_transition: int, optional default=10
    Window for smooth transition in case of fixed_smooth type of transition.
    Ignored in the other cases.
    mode_precisions: string, optional default='complementary'
    Possible values are complementary to generate complementary precision matrices,
    regain to use the funcion make_starting of the library, sklearn to use the function make_spd_matrix
    """
    precisions, covariances, means = [], [], []
    if mode_precisions == 'complementary':
        covariances, precisions = generate_complementary_precisions_matrix(
            n_dim_obs, n_states ** order_hmm)
    else:
        for k in range(n_states ** order_hmm):
            if mode_precisions == 'regain':
                precisions.append(
                    make_starting(n_dim_obs=n_dim_obs, n_dim_lat=0,
                                  **kwargs)[0])
                covariances.append(linalg.pinv(precisions[k]))
            else:
                covariances.append(make_spd_matrix(n_dim_obs))
                precisions.append(linalg.pinv(covariances[k]))

    if mode_mean == 'Normal':
        means = [
            np.random.normal(0, sigma, n_dim_obs) for k in range(n_states ** order_hmm)
        ]
    elif mode_mean == 'Uniform':
        min_max = np.random.uniform(-50, 50, 2)
        means = [
            np.random.uniform(min(min_max), max(min_max), n_dim_obs)
            for k in range(n_states ** order_hmm)
        ]
    else:
        raise ValueError('Unknown mode_mean type')

    # Generate a transition matrix
    A = np.zeros((n_states ** order_hmm, n_states ** order_hmm))
    if order_hmm==1:
        for i in range(n_states):
            alphas = np.ones(n_states)
            alphas[i] = stability_factor * alphas[i]
            A[i, :] = np.random.dirichlet(alphas, 1)
    else:
        for i in range(n_states ** order_hmm):
            index_fill = np.zeros(n_states ** order_hmm, bool)
            for j in range(n_states ** order_hmm):
                index_fill[j] = np.floor((i) / n_states) == (j) - np.floor((j) / (n_states ** (order_hmm - 1))) * n_states ** (order_hmm - 1)
            A[i, index_fill] = np.random.dirichlet(np.ones(np.sum(index_fill)), 1)
    state = 0
    states = []
    data = np.zeros((n_samples, n_dim_obs))
    gammas = np.zeros((n_samples, n_states ** order_hmm))
    if transition_type == 'fixed':
        for i in range(n_samples):
            states.append(state)
            gammas[i, state] = 1
            data[i, :] = np.random.multivariate_normal(means[state],
                                                       covariances[state], 1)
            state = np.random.choice(np.arange(n_states ** order_hmm),
                                     replace=True,
                                     p=A[state, :])
    elif (transition_type == 'fixed_smooth'
          or transition_type == 'random_smooth'
          or transition_type == 'random'):
        States_count = np.zeros((n_samples, n_states))
        S = smooth_transition
        for i in range(n_samples):
            states.append(state)
            States_count[i, state] = 1

            if transition_type == 'random_smooth':
                if i > 0 and states[-1] != states[-2]:
                    S = int(
                        np.floor(
                            np.random.uniform(min_transition_window,
                                              max_transition_window, 1)))

            frac_count_vec = np.sum(States_count[max(0, i - S + 1):(i + 1), :],
                                    axis=0) / min(S, i + 1)
            if transition_type == 'random':
                frac_count_vec = (frac_count_vec * min(S, i + 1)) + 1
                frac_count_vec = np.random.dirichlet(
                    frac_count_vec * np.ones(n_states)[:, np.newaxis],
                    1).ravel()

            gammas[i, :] = frac_count_vec
            sample = np.zeros((1, n_dim_obs))
            for k in range(n_states):
                sample += frac_count_vec[k] * np.random.multivariate_normal(
                    means[k], covariances[k], 1)
            data[i, :] = sample
            state = np.random.choice(np.arange(n_states),
                                     replace=True,
                                     p=A[state, :])
    else:
        raise ValueError('Unknown type of transition. Possible choices are:'
                         'fixed, fixed_smooth, random_smooth, random')

    variations_sum = []
    p_vec_0 = np.random.uniform(0, 100, n_dim_obs)
    for j in range(n_dim_obs):
        variations_sum.append(p_vec_0[j] + np.cumsum(data[:, j]))
    variations_sum = np.array(variations_sum)

    states = np.argmax(gammas, axis=1)
    res = dict(data=data,
               thetas=precisions,
               covariances=covariances,
               means=means,
               transition_state=A,
               cumulatives=variations_sum,
               states=states,
               gammas=gammas)
    return res


