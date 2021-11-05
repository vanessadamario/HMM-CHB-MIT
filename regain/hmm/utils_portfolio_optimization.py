import numpy as np
from regain.hmm.utils import cross_validation
from regain.hmm.hmm_graphical_lasso import HMM_GraphicalLasso
from sklearn.covariance import empirical_covariance
from regain.hmm.utils import results_recap


def PO_minimum_variance(mu,cov):

    # Portfolio optimization with minimum variace

    N_ts = np.size(mu)

    A = np.zeros((N_ts + 1, N_ts + 1))

    A[:-1, :-1] = 2 * np.array(cov)
    A[:-1, -1] = np.repeat(1, N_ts)
    A[-1, :-1] = np.repeat(1, N_ts)

    b = np.zeros(N_ts + 1)
    b[-1] = 1

    weights = np.matmul(np.linalg.pinv(A), b)[:-1]
    exp_port = np.matmul(weights.transpose(), mu)
    var_port = np.matmul(np.matmul(weights.transpose(), cov), weights)

    return weights,exp_port,var_port


def PO_minimum_variance_and_fixed_return(mu, cov,mu_p):

    # Minimum variance portfolio with expected returns


    N_ts = np.size(mu)

    A = np.zeros((N_ts + 2, N_ts + 2))

    A[:-2, :-2] = 2 * np.array(cov)
    A[:-2, -2] = mu
    A[-2, :-2] = mu
    A[:-2, -1] = np.repeat(1, N_ts)
    A[-1, :-2] = np.repeat(1, N_ts)

    b = np.zeros(N_ts + 2)
    b[-2] = mu_p
    b[-1] = 1

    weights = np.matmul(np.linalg.pinv(A), b)[:-2]
    exp_port = np.matmul(weights.transpose(), mu)
    var_port = np.matmul(np.matmul(weights.transpose(), cov), weights)


    return weights, exp_port, var_port


def PO_minimum_tangency(mu, cov,rf):

    # Tangency portfolio

    prec = np.linalg.pinv(cov)
    diff = mu-np.repeat(rf,len(mu))
    num = np.matmul(prec, diff)
    den = np.matmul(np.repeat(1,len(mu)).transpose(),num)

    t = num/den

    mu_p = np.matmul(t.transpose(),mu)
    var_port = np.matmul(np.matmul(t.transpose(),cov),t)

    return t,mu_p,var_port

def leverage_weights(mean, cov):

    variances_pred = cov.diagonal()
    std_pred = np.sqrt(variances_pred)
    max_std_pred = np.max(std_pred)
    lev = np.reciprocal(std_pred / max_std_pred)
    mean_lev = lev * mean

    print(lev,mean,mean_lev)

    return mean_lev


def PO_with_HMM_GMM(returns, Prices, alpha_list, clus_list, N_test=100, Wealth=1e5, N_past_days=50,
                     meth='viterbi', OP_method = 'Min_Var',rf = 0.005, mu_p= 0.05,N_max_mem = 1000, N_CV = 10,
                    leverage=False):
    N_obs = np.size(returns, axis=0)
    N_ts = np.size(returns, axis=1)

    PL_tot_predict = []
    PL_tot_today = []
    PL_tot_emp = []

    N_invest_mem = np.zeros((N_ts, 3))

    alpha_best = 20
    cluster_best = 3

    cluster_mem_pred = 1000
    cluster_mem_today = 1000

    for i in range(N_test):

        N_obs_train = N_obs - N_test + i

        N_memory = max(N_obs_train-N_max_mem,0)

        X_train = returns[N_memory:N_obs_train, :]
        X_train_emp = returns[N_obs_train - N_past_days:N_obs_train, :]

        if np.mod(i,N_CV)==0:

            mdl = HMM_GraphicalLasso(alpha=1, n_clusters=3, verbose=False, mode='scaled',
                                         warm_restart=True, repetitions=50, n_jobs=-1)

            res = cross_validation(mdl,
                                    X_train,
                                    params={'alpha': alpha_list,
                                            'n_clusters': clus_list},
                                    n_repetitions=1)
            alpha_best = res[0][0]
            cluster_best = res[0][1]


        hmm_gmm = HMM_GraphicalLasso(alpha=alpha_best, n_clusters=cluster_best, verbose=False, mode='scaled',
                                     warm_restart=True, repetitions=50, n_jobs=-1)

        hmm_gmm.fit(X_train)
        state_adj = False
        pred_weights = False
        today_weights = False
        Cv_adj = False

        if i == 0 or np.mod(i,N_CV)==0:
            label_mem = hmm_gmm.labels_
            prec_mem = hmm_gmm.precisions_
            Cv_adj = True

        else:
            state_adj = True
            clus_comp= results_recap(label_mem,
                                     hmm_gmm.labels_,
                                     prec_mem,
                                     hmm_gmm.precisions_)
            corr_mat = np.zeros((cluster_best,2))
            for nn in range(cluster_best):

                corr_mat[nn,0] = int(clus_comp['max_probabilities_couples'][nn][9])
                corr_mat[nn, 1] = int(clus_comp['max_probabilities_couples'][nn][12])




        # Using HMM-GMM prediction

        pred = hmm_gmm.predict(X_train, method=meth)


        cov_pred =  pred['cov']
        mean_pred = pred['means']
        if state_adj:
            cluster_pred = corr_mat[corr_mat[:, 1] == pred['state'], 0]
        else:
            cluster_pred = pred['state']

        if Cv_adj or cluster_mem_pred != cluster_pred:

            print('Pred changed cluster')
            cluster_mem_pred = cluster_pred
            pred_weights = True



        # Using HMM-GMM today

        if state_adj:
            cluster_today = corr_mat[corr_mat[:, 1] == hmm_gmm.labels_[-1], 0]
        else:
            cluster_today = hmm_gmm.labels_[-1]

        cov_today = hmm_gmm.covariances_[hmm_gmm.labels_[-1]]
        mean_today = hmm_gmm.means_[hmm_gmm.labels_[-1]]

        if Cv_adj or cluster_mem_today != cluster_today:

            print('Today changed cluster')
            cluster_mem_today = cluster_today
            today_weights = True

        # Using Empirical covariance

        means_today_emp = np.mean(X_train_emp, axis=0)
        cov_today_emp = empirical_covariance(X_train_emp - means_today_emp, assume_centered=True)

        if leverage:
            mean_pred = leverage_weights(mean_pred, cov_pred)
            mean_today = leverage_weights(mean_today, cov_today)
            means_today_emp = leverage_weights(means_today_emp, cov_today_emp)



        # Portfolio optimization

        if OP_method == 'Min_Var':

            if pred_weights:
                weights_predict, *others = PO_minimum_variance(mean_pred,
                                                               cov_pred)
            if today_weights:
                weights_today, *others = PO_minimum_variance(mean_today,cov_today)

            if np.mod(i, 10) == 0:
                weights_emp, *others = PO_minimum_variance(means_today_emp,cov_today_emp)

        elif OP_method == 'Min_Var_Fix_return':

            if pred_weights:
                weights_predict, *others = PO_minimum_variance_and_fixed_return(mean_pred,
                                                                            cov_pred,
                                                                            mu_p=mu_p)
            if today_weights:
                weights_today, *others = PO_minimum_variance_and_fixed_return(mean_today,
                                                                          cov_today,
                                                                          mu_p=mu_p)
            if np.mod(i, 10) == 0:
                weights_emp, *others = PO_minimum_variance_and_fixed_return(means_today_emp,
                                                                        cov_today_emp,
                                                                        mu_p=mu_p)

        elif OP_method == 'Tangency':

            if pred_weights:
                weights_predict, *others = PO_minimum_tangency(mean_pred,cov_pred,rf=rf)

            if today_weights:
                weights_today, *others = PO_minimum_tangency(mean_today,cov_today,rf=rf)
            if np.mod(i, 10) == 0:
                weights_emp, *others = PO_minimum_tangency(means_today_emp,cov_today_emp,rf=rf)

        else:
            raise ValueError('Unknown portfolio optimization method ')


        PL_predict = 0
        PL_today = 0
        PL_emp = 0

        for ts in range(N_ts):

            if pred_weights:
                Wealth_on_ts_predict = Wealth * weights_predict[ts]
                N_on_ts_predict = int(Wealth_on_ts_predict / Prices[N_obs_train - 1, ts])
                if N_on_ts_predict == 0:
                    raise ValueError('Increase Wealth')
                N_invest_mem[ts,0] = N_on_ts_predict

            if today_weights:
                Wealth_on_ts_today = Wealth * weights_today[ts]
                N_on_ts_today = int(Wealth_on_ts_today / Prices[N_obs_train - 1, ts])
                if N_on_ts_today == 0:
                    raise ValueError('Increase Wealth')
                N_invest_mem[ts,1] = N_on_ts_today

            if np.mod(i, 10) == 0:
                Wealth_on_ts_emp = Wealth * weights_emp[ts]
                N_on_ts_emp = int(Wealth_on_ts_emp / Prices[N_obs_train - 1, ts])
                if N_on_ts_emp == 0:
                    raise ValueError('Increase Wealth')
                N_invest_mem[ts,2] = N_on_ts_emp


            print('Price difference', Prices[N_obs_train, ts] - Prices[N_obs_train - 1, ts],
                  'N purchased pred', N_invest_mem[ts,0],
                  'N purchased today', N_invest_mem[ts,1],
                  'N purchased emp', N_invest_mem[ts,2])

            PL_predict += (Prices[N_obs_train, ts] - Prices[N_obs_train - 1, ts]) * N_invest_mem[ts,0]
            PL_today += (Prices[N_obs_train, ts] - Prices[N_obs_train - 1, ts]) * N_invest_mem[ts,1]
            PL_emp += (Prices[N_obs_train, ts] - Prices[N_obs_train - 1, ts]) * N_invest_mem[ts,2]

        PL_tot_predict.append(PL_predict)
        PL_tot_today.append(PL_today)
        PL_tot_emp.append(PL_emp)

        print('P&L predict', np.cumsum(np.array(PL_tot_predict)))
        print('P&L today', np.cumsum(np.array(PL_tot_today)))
        print('P&L emp', np.cumsum(np.array(PL_tot_emp)))

    return PL_tot_predict, PL_tot_today, PL_tot_emp



def Cointegration_strategy_with_HMM_GMM(returns, Prices, alpha_list, clus_list, N_test=100, Wealth=1e5, N_past_days=50,
                     meth='viterbi', OP_method = 'Min_Var',rf = 0.005, mu_p= 0.05,N_max_mem = 1000, N_CV = 10,
                    leverage=False):
    N_obs = np.size(returns, axis=0)
    N_ts = np.size(returns, axis=1)

    PL_tot_predict = []
    PL_tot_today = []
    PL_tot_emp = []

    N_invest_mem = np.zeros((N_ts, 3))

    alpha_best = 20
    cluster_best = 3

    cluster_mem_pred = 1000
    cluster_mem_today = 1000

    for i in range(N_test):

        N_obs_train = N_obs - N_test + i

        N_memory = max(N_obs_train-N_max_mem,0)

        X_train = returns[N_memory:N_obs_train, :]
        X_train_emp = returns[N_obs_train - N_past_days:N_obs_train, :]

        if np.mod(i,N_CV)==0:

            mdl = HMM_GraphicalLasso(alpha=1, n_clusters=3, verbose=False, mode='scaled',
                                         warm_restart=True, repetitions=50, n_jobs=-1)

            res = cross_validation(mdl,
                                    X_train,
                                    params={'alpha': alpha_list,
                                            'n_clusters': clus_list},
                                    n_repetitions=1)
            alpha_best = res[0][0]
            cluster_best = res[0][1]


        hmm_gmm = HMM_GraphicalLasso(alpha=alpha_best, n_clusters=cluster_best, verbose=False, mode='scaled',
                                     warm_restart=True, repetitions=50, n_jobs=-1)

        hmm_gmm.fit(X_train)
        state_adj = False
        pred_weights = False
        today_weights = False
        Cv_adj = False

        if i == 0 or np.mod(i,N_CV)==0:
            label_mem = hmm_gmm.labels_
            prec_mem = hmm_gmm.precisions_
            Cv_adj = True

        else:
            state_adj = True
            clus_comp= results_recap(label_mem,
                                     hmm_gmm.labels_,
                                     prec_mem,
                                     hmm_gmm.precisions_)
            corr_mat = np.zeros((cluster_best,2))
            for nn in range(cluster_best):

                corr_mat[nn,0] = int(clus_comp['max_probabilities_couples'][nn][9])
                corr_mat[nn, 1] = int(clus_comp['max_probabilities_couples'][nn][12])




        # Using HMM-GMM prediction

        pred = hmm_gmm.predict(X_train, method=meth)


        cov_pred =  pred['cov']
        mean_pred = pred['means']
        if state_adj:
            cluster_pred = corr_mat[corr_mat[:, 1] == pred['state'], 0]
        else:
            cluster_pred = pred['state']

        if Cv_adj or cluster_mem_pred != cluster_pred:

            print('Pred changed cluster')
            cluster_mem_pred = cluster_pred
            pred_weights = True



        # Using HMM-GMM today

        if state_adj:
            cluster_today = corr_mat[corr_mat[:, 1] == hmm_gmm.labels_[-1], 0]
        else:
            cluster_today = hmm_gmm.labels_[-1]

        cov_today = hmm_gmm.covariances_[hmm_gmm.labels_[-1]]
        mean_today = hmm_gmm.means_[hmm_gmm.labels_[-1]]

        if Cv_adj or cluster_mem_today != cluster_today:

            print('Today changed cluster')
            cluster_mem_today = cluster_today
            today_weights = True

        # Using Empirical covariance

        means_today_emp = np.mean(X_train_emp, axis=0)
        cov_today_emp = empirical_covariance(X_train_emp - means_today_emp, assume_centered=True)

        if leverage:
            mean_pred = leverage_weights(mean_pred, cov_pred)
            mean_today = leverage_weights(mean_today, cov_today)
            means_today_emp = leverage_weights(means_today_emp, cov_today_emp)



        # Portfolio optimization

        if OP_method == 'Min_Var':

            if pred_weights:
                weights_predict, *others = PO_minimum_variance(mean_pred,
                                                               cov_pred)
            if today_weights:
                weights_today, *others = PO_minimum_variance(mean_today,cov_today)

            if np.mod(i, 10) == 0:
                weights_emp, *others = PO_minimum_variance(means_today_emp,cov_today_emp)

        elif OP_method == 'Min_Var_Fix_return':

            if pred_weights:
                weights_predict, *others = PO_minimum_variance_and_fixed_return(mean_pred,
                                                                            cov_pred,
                                                                            mu_p=mu_p)
            if today_weights:
                weights_today, *others = PO_minimum_variance_and_fixed_return(mean_today,
                                                                          cov_today,
                                                                          mu_p=mu_p)
            if np.mod(i, 10) == 0:
                weights_emp, *others = PO_minimum_variance_and_fixed_return(means_today_emp,
                                                                        cov_today_emp,
                                                                        mu_p=mu_p)

        elif OP_method == 'Tangency':

            if pred_weights:
                weights_predict, *others = PO_minimum_tangency(mean_pred,cov_pred,rf=rf)

            if today_weights:
                weights_today, *others = PO_minimum_tangency(mean_today,cov_today,rf=rf)
            if np.mod(i, 10) == 0:
                weights_emp, *others = PO_minimum_tangency(means_today_emp,cov_today_emp,rf=rf)

        else:
            raise ValueError('Unknown portfolio optimization method ')


        PL_predict = 0
        PL_today = 0
        PL_emp = 0

        for ts in range(N_ts):

            if pred_weights:
                Wealth_on_ts_predict = Wealth * weights_predict[ts]
                N_on_ts_predict = int(Wealth_on_ts_predict / Prices[N_obs_train - 1, ts])
                if N_on_ts_predict == 0:
                    raise ValueError('Increase Wealth')
                N_invest_mem[ts,0] = N_on_ts_predict

            if today_weights:
                Wealth_on_ts_today = Wealth * weights_today[ts]
                N_on_ts_today = int(Wealth_on_ts_today / Prices[N_obs_train - 1, ts])
                if N_on_ts_today == 0:
                    raise ValueError('Increase Wealth')
                N_invest_mem[ts,1] = N_on_ts_today

            if np.mod(i, 10) == 0:
                Wealth_on_ts_emp = Wealth * weights_emp[ts]
                N_on_ts_emp = int(Wealth_on_ts_emp / Prices[N_obs_train - 1, ts])
                if N_on_ts_emp == 0:
                    raise ValueError('Increase Wealth')
                N_invest_mem[ts,2] = N_on_ts_emp


            print('Price difference', Prices[N_obs_train, ts] - Prices[N_obs_train - 1, ts],
                  'N purchased pred', N_invest_mem[ts,0],
                  'N purchased today', N_invest_mem[ts,1],
                  'N purchased emp', N_invest_mem[ts,2])

            PL_predict += (Prices[N_obs_train, ts] - Prices[N_obs_train - 1, ts]) * N_invest_mem[ts,0]
            PL_today += (Prices[N_obs_train, ts] - Prices[N_obs_train - 1, ts]) * N_invest_mem[ts,1]
            PL_emp += (Prices[N_obs_train, ts] - Prices[N_obs_train - 1, ts]) * N_invest_mem[ts,2]

        PL_tot_predict.append(PL_predict)
        PL_tot_today.append(PL_today)
        PL_tot_emp.append(PL_emp)

        print('P&L predict', np.cumsum(np.array(PL_tot_predict)))
        print('P&L today', np.cumsum(np.array(PL_tot_today)))
        print('P&L emp', np.cumsum(np.array(PL_tot_emp)))

    return PL_tot_predict, PL_tot_today, PL_tot_emp
