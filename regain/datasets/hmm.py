import numpy as np
from regain.datasets.gaussian import make_starting
from scipy import linalg
from scipy.stats import bernoulli
from sklearn.datasets import make_spd_matrix

# def ncr(n, r):
#     r = min(r, n-r)
#     numer = reduce(op.mul, range(n, n-r, -1), 1)
#     denom = reduce(op.mul, range(1, r+1), 1)
#     return numer / denom


def generate_precision_matrix(n_dim_obs=100, previouses=[]):
    if n_dim_obs <= 2:
        raise ValueError('With only 2 observed variables it is not possible '
                         'to create different complementary states.')

    theta = np.identity(n_dim_obs)

    not_available_positions = []

    for p in previouses:
        print(p)
        for i, j in zip(np.where(p != 0)[0], np.where(p != 0)[1]):
            not_available_positions.append((i, j))
    not_available_positions = list(set(not_available_positions))
    print(not_available_positions)
    for i in range(n_dim_obs):
        for j in range(i + 1, n_dim_obs):
            if (i, j) not in not_available_positions:
                theta[i, j] = bernoulli.rvs(0.5, size=1)
                theta[j, i] = theta[i, j]
    return theta


def generate_complementary_precisions_matrix(n_dim_obs=100, n_states=5):

    # possible combinations Probabilmente non necessario
    # N_tot_comb = 0
    # for i in range(1,dim+1):
    #     N_tot_comb += ncr(dim, i)
    # if N_states > N_tot_comb:
    #     raise ValueError('The number of states is too big, reduce the value of N_states')

    precisions = []
    covariances = []
    for i in range(n_states):
        precisions.append(generate_precision_matrix(n_dim_obs, precisions))
        covariances.append(linalg.pinv(precisions[-1]))
    return covariances, precisions


def generate_hmm(n_samples=100,
                 n_states=5,
                 n_dim_obs=10,
                 mode_precisions='complementary',
                 transition_type='fixed',
                 smooth_transition=10,
                 min_transition_window=1,
                 max_transition_window=20,
                 sigma=0.7,
                 stability_factor=12,
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
            n_dim_obs, n_states)
    else:
        for k in range(n_states):
            if mode_precisions == 'regain':
                precisions.append(
                    make_starting(n_dim_obs=n_dim_obs, n_dim_lat=0,
                                  **kwargs)[0])
                covariances.append(linalg.pinv(precisions[k]))
            else:
                covariances.append(make_spd_matrix(n_dim_obs))
    means = [np.random.normal(0, sigma, n_dim_obs) for k in range(n_states)]
    # Generate a transition matrix
    A = np.zeros((n_states, n_states))
    for i in range(n_states):
        alphas = np.ones(n_states)
        alphas[i] = stability_factor * alphas[i]
        A[i, :] = np.random.dirichlet(alphas, 1)

    state = 0
    states = []
    data = np.zeros((n_samples, n_dim_obs))
    gammas = np.zeros((n_samples, n_states))
    if transition_type == 'fixed':
        for i in range(n_samples):
            states.append(state)
            gammas[i, state] = 1
            data[i, :] = np.random.multivariate_normal(means[state],
                                                       covariances[state], 1)
            state = np.random.choice(np.arange(n_states),
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
