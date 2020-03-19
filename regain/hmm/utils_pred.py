import numpy as np
import matplotlib.pyplot as plt
from regain.hmm.utils import cross_validation,cov2corr
from regain.hmm.hmm_graphical_lasso import HMM_GraphicalLasso
from sklearn.covariance import empirical_covariance
import scipy
from scipy.integrate import quad
import scipy.stats
import pandas as pd
import seaborn as sns
from regain.utils import structure_error

def Value_from_returns(Data, results,N_pred, pred_meth, perc_var):

    N_samples = np.size(Data,axis=0)
    N_TS = np.size(Data, axis=1)
    N_start = N_samples - N_pred - 1
    starting_point = Data[N_start]

    Value_pred = np.zeros((N_pred,N_TS))
    Value_pred_mean = np.zeros((N_pred, N_TS))

    for ts in range(N_TS):
        tempsigp = []
        tempsigm = []
        tempmean = []
        tempreal = []
        temppred = []
        valup = starting_point[ts]
        valmean = starting_point[ts]
        valdown = starting_point[ts]
        real = starting_point[ts]
        pred = starting_point[ts]
        for prev in range(N_pred):

            if pred_meth == 'rolling':
                starting_point = Data[N_start + prev]
                valup = starting_point[ts]
                valmean = starting_point[ts]
                valdown = starting_point[ts]
                real = starting_point[ts]
                pred = starting_point[ts]

            if perc_var:
                valmean = valmean * results[5][str(prev)][ts] / 100 + valmean
                pred = pred * results[0][prev][ts] / 100 + pred

            else:
                valmean = valmean + results[5][str(prev)][ts]
                pred = pred + results[0][prev][ts]

            Value_pred[prev,ts] = pred
            Value_pred_mean[prev, ts] = valmean

    return Value_pred,Value_pred_mean


def thetas_comparison(thetas_true,thetas_pred):

    mcc = np.zeros(len(thetas_true))
    f1_score = np.zeros(len(thetas_true))

    for i in range(len(thetas_true)):

        ss = structure_error(thetas_true[str(i)], thetas_pred[str(i)], no_diagonal=True)
        mcc[i] = ss['mcc']
        f1_score[i] = ss['f1']
        print(mcc[i])

    return np.mean(mcc),np.mean(f1_score)





def plot_pred(Data, results, N_pred, pred_meth, columns = None, mem_days=10, N_per_rows=2, figsizex=10, figsizey=20, perc_var=False):

    N_samples = np.size(Data,axis=0)
    N_TS = np.size(Data, axis=1)

    N_rows = int(np.ceil(N_TS / N_per_rows))
    f, axes = plt.subplots(N_rows, N_per_rows, figsize=(figsizex, figsizey))

    x_mem = np.arange(mem_days + N_pred)
    x = np.arange(mem_days, mem_days + N_pred)
    N_start = N_samples - N_pred - 1
    starting_point = Data[N_start]
    Data_mem = Data[N_start - mem_days + 1:N_start + 1 + N_pred, :]

    for ts in range(N_TS):
        tempsigp = []
        tempsigm = []
        tempmean = []
        tempreal = []
        temppred = []
        valup = starting_point[ts]
        valmean = starting_point[ts]
        valdown = starting_point[ts]
        real = starting_point[ts]
        pred = starting_point[ts]
        for prev in range(N_pred):

            if pred_meth == 'rolling':
                starting_point = Data[N_start + prev]
                valup = starting_point[ts]
                valmean = starting_point[ts]
                valdown = starting_point[ts]
                real = starting_point[ts]
                pred = starting_point[ts]

            if perc_var:

                valup = valup * (results[5][str(prev)][ts] + 2 * results[2][str(prev)][ts]) / 100 + valup
                valdown = valdown * (results[5][str(prev)][ts] - 2 * results[2][str(prev)][ts]) / 100 + valdown
                valmean = valmean * results[5][str(prev)][ts] / 100 + valmean
                real = real * results[1][prev][ts] / 100 + real
                pred = pred * results[0][prev][ts] / 100 + pred

            else:
                valup = valup + results[5][str(prev)][ts] + 2 * results[2][str(prev)][ts]
                valdown = valdown + results[5][str(prev)][ts] - 2 * results[2][str(prev)][ts]
                valmean = valmean + results[5][str(prev)][ts]
                real = real + results[1][prev][ts]
                pred = pred + results[0][prev][ts]

            tempsigp.append(valup)
            tempsigm.append(valdown)
            tempmean.append(valmean)
            tempreal.append(real)
            temppred.append(pred)

        i = int(ts / N_per_rows)
        j = np.remainder(ts, N_per_rows)

        axes[i, j].plot(x_mem, Data_mem[:, ts])
        axes[i, j].plot(x, tempmean, label='mean')
        axes[i, j].fill_between(x, tempsigp, tempsigm, alpha=0.2, label='confidence')
        axes[i, j].plot(x, tempreal, 'o', label='real')
        axes[i, j].plot(x, temppred, '*', label='pred')
        if columns is None:
            axes[i, j].set_title('var ' + str(ts) + ' forecast')
        else:
            axes[i, j].set_title(str(columns[ts]) + ' forecast')

        axes[i, j].legend()


def pred_HMM_GMM(returns,data,alpha_list, clus_list, N_test=100, columns = None, meth='viterbi', pred_meth='rolling',
                 N_retrain = 5 ,recrossval=True, plot=True, mem_days=10, N_per_rows=2, figsizex=10,
                 figsizey=20, perc_var=False):
    N_obs = np.size(returns, axis=0)
    Y_HMM_GMM_pred = np.zeros((N_test, np.size(returns, axis=1)))
    Y_real = np.zeros((N_test, np.size(returns, axis=1)))

    cov_pred = {}
    stds_pred = {}
    prec_pred = {}
    means_pred = {}

    for i in range(N_test):

        N_obs_train = N_obs - N_test + i

        if i == 0 or pred_meth == 'rolling':

            X_train = returns[:N_obs_train, :]

        else:

            X_train = np.vstack((X_train, X_new))

        if i == 0 or recrossval or (pred_meth == 'rolling' and np.remainder(i,N_retrain)== 0):

            mdl = HMM_GraphicalLasso(alpha=1, n_clusters=3, verbose=False, mode='scaled',
                                     warm_restart=True, repetitions=5, n_jobs=-1)

            res = cross_validation(mdl,
                                   X_train,
                                   params={'alpha': alpha_list,
                                           'n_clusters': clus_list},
                                   n_repetitions=3)

        hmm_gmm = HMM_GraphicalLasso(alpha=res[0][0], n_clusters=res[0][1], verbose=False, mode='scaled',
                                     warm_restart=True, repetitions=5, n_jobs=-1)

        hmm_gmm.fit(X_train)

        # real

        Y_real[i, :] = returns[N_obs_train, :]

        # predictions
        X_new = hmm_gmm.predict(X_train, method=meth)['pred']
        Y_HMM_GMM_pred[i, :] = hmm_gmm.predict(X_train, method=meth)['pred']
        stds_pred[str(i)] = hmm_gmm.predict(X_train, method=meth)['stds']
        cov_pred[str(i)] = hmm_gmm.predict(X_train, method=meth)['cov']
        prec_pred[str(i)] = hmm_gmm.predict(X_train, method=meth)['prec']
        means_pred[str(i)] = hmm_gmm.predict(X_train, method=meth)['means']

    results = [Y_HMM_GMM_pred,
               Y_real,
               stds_pred,
               prec_pred,
               cov_pred,
               means_pred]

    if plot:
        plot_pred(data[1:,:], results, N_test, pred_meth,columns, mem_days, N_per_rows, figsizex, figsizey, perc_var)

    return results

def pred_from_N_past_days(returns,data,N_past_days,N_test=100,pred_meth = 'rolling',columns = None,plot=True, mem_days=10,
                          N_per_rows=2, figsizex=10,figsizey=20, perc_var=False):

    N_obs = np.size(returns, axis=0)
    Y_pred = np.zeros((N_test, np.size(returns, axis=1)))
    Y_real = np.zeros((N_test, np.size(returns, axis=1)))

    cov_pred = {}
    stds_pred = {}
    prec_pred = {}
    means_pred = {}

    for i in range(N_test):

        N_obs_train = N_obs - N_test + i

        if i == 0 or pred_meth == 'rolling':

            X_train = returns[N_obs_train-N_past_days:N_obs_train, :]

        else:

            X_train = np.vstack((X_train[1:,:], X_new))

        # real

        Y_real[i, :] = returns[N_obs_train, :]

        # predictions
        means_pred[str(i)] = np.mean(X_train, axis=0)
        cov_pred[str(i)] = empirical_covariance(X_train - means_pred[str(i)], assume_centered=True)
        prec_pred[str(i)] = np.linalg.pinv(cov_pred[str(i)])
        Y_pred[i, :] = np.random.multivariate_normal(means_pred[str(i)],cov_pred[str(i)], 1)
        X_new = Y_pred[i, :]
        stds_pred[str(i)] = np.sqrt(cov_pred[str(i)] .diagonal())


    results = [Y_pred,
               Y_real,
               stds_pred,
               prec_pred,
               cov_pred,
               means_pred]
    if plot:
        plot_pred(data[1:,:], results, N_test, pred_meth,columns, mem_days, N_per_rows, figsizex, figsizey, perc_var)

    return results

def returns_recap(means, covariances, labels=None):

    N_ts = np.size(means['0'])
    mean_std = []

    for k in means.keys():

        temp = np.sqrt(covariances[k].diagonal())
        for n in range(N_ts):
            mean_std_row = []
            # cluster
            mean_std_row.append('day '+str(int(k)+1))
            # time series
            if labels is None:
                mean_std_row.append('ts ' + str(n))
            else:
                mean_std_row.append(labels[n])
            # mean
            mean_std_row.append(means[k][n])
            # std
            mean_std_row.append(temp[n])
            # prob positive trend
            if means[k][n] + 3 * temp[n] < 0:
                mean_std_row.append(0)
            else:
                f = lambda x: scipy.stats.norm.pdf(x, means[k][n], temp[n])
                mean_std_row.append(quad(f, 0, means[k][n] + 3 * temp[n])[0] * 100)

            # prob negative trend
            if means[k][n] - 3 * temp[n] > 0:
                mean_std_row.append(0)
            else:
                f = lambda x: scipy.stats.norm.pdf(x, means[k][n], temp[n])
                mean_std_row.append(quad(f, means[k][n] - 3 * temp[n], 0)[0] * 100)

            mean_std.append(mean_std_row)

    df_recap = pd.DataFrame(mean_std, columns=['day', 'TS', 'mean', 'std', 'Prob positive return %',
                                               'Prob negative return %'])


    return df_recap

def corr_pred_plot(corrs,sizeplotx=20,sizeploty=10, labels=None, covcorr=False, numbers=True):

    if covcorr:
        correlations = []
        for k in corrs.keys():
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
                    axes[j].set_title('Correlation matrix for day '+str(count+1))
                else:
                    sns.heatmap(corrs[count], annot=numbers,
                                ax=axes[i, j])
                    axes[i, j].set_title('Correlation matrix for day ' + str(count+1))

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
                    axes[j].set_title('Correlation matrix for day '+str(count+1))
                else:
                    sns.heatmap(corrs[count], annot=numbers,
                                xticklabels=labels,
                                yticklabels=labels,
                                ax=axes[i, j])
                    axes[i, j].set_title('Correlation matrix for day '+str(count+1))

                count += 1
                if count == N_plots:
                    break
    plt.show()
