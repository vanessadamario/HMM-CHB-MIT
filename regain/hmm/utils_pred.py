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
from tqdm import tqdm

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

    return np.mean(mcc),np.std(mcc)





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

        N_obs_train = N_obs - N_test + i + 1

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
                                   n_repetitions=1)

        hmm_gmm = HMM_GraphicalLasso(alpha=res[0][0], n_clusters=res[0][1], verbose=False, mode='scaled',
                                     warm_restart=True, repetitions=5, n_jobs=-1)

        hmm_gmm.fit(X_train)

        # real

        Y_real[i, :] = returns[N_obs_train-1, :]

        # predictions

        pred_res = hmm_gmm.predict(X_train, method=meth)
        state = pred_res['state']


        # organization for regression

        X_new = pred_res['pred']
        Y_HMM_GMM_pred[i, :] = pred_res['pred']
        stds_pred[str(i)] = pred_res['stds']
        cov_pred[str(i)] = pred_res['cov']
        prec_pred[str(i)] = pred_res['prec']
        means_pred[str(i)] = pred_res['means']

    results = [Y_HMM_GMM_pred,
               Y_real,
               stds_pred,
               prec_pred,
               cov_pred,
               means_pred]

    if plot:
        plot_pred(data[1:,:], results, N_test, pred_meth,columns, mem_days, N_per_rows, figsizex, figsizey, perc_var)

    return results


def CV_reg_HMM_GGM(model,X,alpha_list, cluster_list,N_val,meth):

    N_obs,_ = X.shape
    results = []
    par_list = []

    for a in tqdm(alpha_list):
        for c in tqdm(cluster_list):

            model.alpha = a
            model.n_clusters = c

            Y_HMM_GMM_pred = np.zeros(N_val)
            MAE_HMM_GMM_pred = np.zeros(N_val)

            for i in range(N_val):

                N_obs_train = N_obs - N_val + i
                X_train = X[:N_obs_train, :]


                # training

                X_traning = X_train[:-1, :]

                # val
                X_pred = X_train[-1, :]

                model.fit(X_traning)
                pred_res = model.predict(X_traning, method=meth)
                Prec_pred = pred_res['prec'][:-1, :-1]
                Cov_row_pred = pred_res['cov'][:, -1]
                mean_out_pred = pred_res['means']
                Y_HMM_GMM_pred[i] = mean_out_pred[-1] + np.dot(np.dot((X_pred[:-1] - mean_out_pred[:-1]), Prec_pred),
                                                                  Cov_row_pred[:-1])

                MAE_HMM_GMM_pred[i] = np.abs(Y_HMM_GMM_pred[i] - X_pred[-1])

            results.append(np.mean(MAE_HMM_GMM_pred))
            par_list.append((a, c))

    print(par_list[np.nanargmin(results)])
    print(np.min(results))

    return [par_list[np.nanargmin(results)]]




def CV_and_pred(i,j,recrossval,pred_meth,N_retrain,alpha_list,clus_list,N_val,meth,Value_pred,CV_meth,
                X_traning, Y_training,X_pred,Y_HMM_GMM_pred,Err_HMM_GMM_pred,MultiY,perc_var,pred,single_var,al,cl):



    X_temp = np.column_stack((X_traning, Y_training[:, j]))

    if i == 0 or recrossval or (pred_meth == 'rolling' and np.remainder(i, N_retrain) == 0):

        mdl = HMM_GraphicalLasso(alpha=1, n_clusters=3, verbose=False, mode='scaled',
                                 warm_restart=True, repetitions=5, n_jobs=-1)

        if CV_meth == 'probabil':
            res = cross_validation(mdl,
                                   X_temp,
                                   params={'alpha': alpha_list,
                                           'n_clusters': clus_list},
                                   n_repetitions=1)
        else:
            res = CV_reg_HMM_GGM(mdl,
                                 X_temp,
                                 alpha_list,
                                 clus_list,
                                 N_val,
                                 meth)
        al = res[0][0]
        cl = res[0][1]

    hmm_gmm = HMM_GraphicalLasso(alpha=al, n_clusters=cl, verbose=False, mode='scaled',
                                 warm_restart=True, repetitions=5, n_jobs=-1)

    hmm_gmm.fit(X_temp)
    pred_res = hmm_gmm.predict(X_temp, method=meth)
    Prec_pred = pred_res['prec'][:-1, :-1]
    Cov_row_pred = pred_res['cov'][:, -1]
    mean_out_pred = pred_res['means']

    if single_var:

        Y_HMM_GMM_pred[i] = mean_out_pred[-1] + np.dot(np.dot((X_pred - mean_out_pred[:-1]), Prec_pred),
                                                          Cov_row_pred[:-1])
        print('MAE pred', i, 'Var', j, ':', np.abs(Y_HMM_GMM_pred[i] - MultiY[-1, j]))
        Err_HMM_GMM_pred[i] = np.sqrt(
            Cov_row_pred[-1] + np.dot(np.dot(Cov_row_pred[:-1], Prec_pred), Cov_row_pred[:-1]))

        if perc_var:
            Value_pred[i] = pred[j] * Y_HMM_GMM_pred[i] / 100 + pred[j]
        else:
            Value_pred[i] = pred[j] + Y_HMM_GMM_pred[i]


    else:

        Y_HMM_GMM_pred[i, j] = mean_out_pred[-1] + np.dot(np.dot((X_pred - mean_out_pred[:-1]), Prec_pred),
                                                          Cov_row_pred[:-1])
        print('MAE pred', i, 'Var', j, ':', np.abs(Y_HMM_GMM_pred[i, j] - MultiY[-1, j]))
        Err_HMM_GMM_pred[i, j] = np.sqrt(Cov_row_pred[-1] + np.dot(np.dot(Cov_row_pred[:-1], Prec_pred), Cov_row_pred[:-1]))

        if perc_var:
            Value_pred[i, j] = pred[j] * Y_HMM_GMM_pred[i, j] / 100 + pred[j]

        else:
            Value_pred[i, j] = pred[j] + Y_HMM_GMM_pred[i, j]

    return Y_HMM_GMM_pred,Err_HMM_GMM_pred,Value_pred,al,cl




def reg_pred_HMM_GMM(returns,data,alpha_list, clus_list,N_val=10, p=2, N_test=100, meth='viterbi', pred_meth='rolling',
                     N_retrain = 5 ,recrossval=True, perc_var=False,CV_meth = 'probabil',single_var = False,var=2):

    N_obs = np.size(returns, axis=0)
    if single_var:
        Value_pred = np.zeros(N_test)
        Y_HMM_GMM_pred = np.zeros(N_test)
        Err_HMM_GMM_pred = np.zeros(N_test)
    else:
        Value_pred = np.zeros((N_test, np.size(returns, axis=1)))
        Y_HMM_GMM_pred = np.zeros((N_test, np.size(returns, axis=1)))
        Err_HMM_GMM_pred = np.zeros((N_test, np.size(returns, axis=1)))

    for i in range(N_test):

        N_obs_train = N_obs - N_test + i+1

        if i == 0 or pred_meth == 'rolling':

            X_train = returns[:N_obs_train, :]
            pred = data[N_obs_train - 2, :]

        # else:
        #
        #     pred = Value_pred[i - 1, :]
        #     X_train = np.vstack((X_train, X_new))

        X, MultiY = prepare_data_to_predict(X_train, p=p)

        # training - validation subdivision

        X_traning = X[:-1, :]
        Y_training = MultiY[:-1, :]

        # prediction
        X_pred = X[-1, :]

        if i==0:
            al = 0
            cl = 2


        if single_var:

            j = var
            print('Prev', i, 'Var', j)

            Y_HMM_GMM_pred,Err_HMM_GMM_pred,Value_pred,al,cl = CV_and_pred(i, j, recrossval, pred_meth, N_retrain,
                                                                           alpha_list, clus_list, N_val, meth,
                                                                           Value_pred, CV_meth, X_traning, Y_training,
                                                                           X_pred, Y_HMM_GMM_pred,Err_HMM_GMM_pred,
                                                                           MultiY, perc_var, pred,single_var,al,cl)

        else:
            for j in range(np.size(MultiY, axis=1)):

                print('Prev',i, 'Var', j)

                Y_HMM_GMM_pred,Err_HMM_GMM_pred,Value_pred,al,cl = CV_and_pred(i, j, recrossval, pred_meth, N_retrain,
                                                                               alpha_list, clus_list, N_val, meth,
                                                                               Value_pred, CV_meth,X_traning, Y_training,
                                                                               X_pred, Y_HMM_GMM_pred,Err_HMM_GMM_pred,
                                                                               MultiY, perc_var, pred,single_var,al,cl)

        #X_new = Y_HMM_GMM_pred[i, :].reshape(1, np.size(returns, axis=1))

    return Y_HMM_GMM_pred,Err_HMM_GMM_pred,Value_pred


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

        N_obs_train = N_obs - N_test + i+1

        if i == 0 or pred_meth == 'rolling':

            X_train = returns[N_obs_train-N_past_days:N_obs_train, :]

        else:

            X_train = np.vstack((X_train[1:,:], X_new))

        # real

        Y_real[i, :] = returns[N_obs_train-1, :]

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
