import lightgbm as lgb
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot
from statsmodels.tsa.api import VAR
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from regain.hmm.hmm_graphical_lasso import HMM_GraphicalLasso
from regain.hmm.utils import cross_validation,results_recap
import matplotlib.pyplot as plt


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

def computeSqDistMat(X1, X2):

    n = np.size(X1,axis=0)
    m = np.size(X2,axis=0)
    Sx1 = np.sum(X1*X1,axis=1)

    SqDistMat = np.repeat(Sx1,m).reshape((m,m))
    Sx2 = np.sum(X2 * X2, axis=1)
    SqDistMat = SqDistMat +  np.repeat(Sx2,n).reshape((n,n))
    SqDistMat = SqDistMat - 2*X1.dot(X2.transpose())

    return SqDistMat

def pred_regression_methods(Data,returns, N_test=100,method= 'lgb', N_val = 10, pred_meth = 'rolling',p = 2,columns=None,
                            perc_var=False,plot = False,figsizex=10, figsizey=20,N_per_rows=2, N_mem = 10,Dates = None):

    N_TS = np.size(Data, axis=1)

    N_rows = int(np.ceil(N_TS / N_per_rows))
    f, axes = plt.subplots(N_rows, N_per_rows, figsize=(figsizex, figsizey))

    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2', 'l1'},
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }
    N_val +=1
    N_obs = np.size(returns,axis=0)
    Y_pred = np.zeros((N_test,np.size(returns,axis=1)))
    Value_pred = np.zeros((N_test, np.size(returns, axis=1)))

    for i in range(N_test):

        N_obs_train = N_obs - N_test + i+1

        if i == 0 or pred_meth == 'rolling':

            X_train = returns[:N_obs_train, :]
            pred = Data[N_obs_train-2,:]

        else:
            pred = Value_pred[i-1, :]
            X_train = np.vstack((X_train, X_new))

        X, MultiY = prepare_data_to_predict(X_train, p = p)

        # training - validation subdivision
        X_train1 = X[:np.size(X, axis=0) - N_val, :]
        Y_train = MultiY[:np.size(X, axis=0) - N_val, :]

        X_val = X[np.size(X, axis=0) - N_val:-1, :]
        Y_val = MultiY[np.size(X, axis=0) - N_val:-1, :]

        # prediction
        X_pred = X[-1, :].reshape(1, np.size(X[-1, :]))


        for j in range(np.size(MultiY, axis=1)):

            if method == 'lgb':

                lgb_train = lgb.Dataset(X_train1, Y_train[:,j])
                lgb_eval = lgb.Dataset(X_val, Y_val[:,j], reference=lgb_train)

                # train
                gbm = lgb.train(params,
                                lgb_train,
                                num_boost_round=100,
                                valid_sets=lgb_eval,
                                early_stopping_rounds=10)
                # predict
                Y_pred[i, j] = gbm.predict(X_pred, num_iteration=gbm.best_iteration)[0]

            elif method == 'LSTM':
                # reshape input to be 3D [samples, timesteps, features]
                train_X = X_train1.reshape((X_train1.shape[0], 1, X_train1.shape[1]))
                val_X = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

                # design network
                model = Sequential()
                model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
                model.add(Dense(1))
                model.compile(loss='mae', optimizer='adam')

                # fit network
                history = model.fit(train_X, Y_train[:,j], epochs=50, batch_size=72, validation_data=(val_X, Y_val[:,j]),
                                    verbose=2,
                                    shuffle=False)

                # make a prediction
                pred_X = X_pred.reshape((1, 1, X_pred.shape[1]))
                Y_pred[i, j] = model.predict(pred_X)

            elif method == 'VAR':

                model = VAR(X_train)
                results = model.fit(maxlags=15, ic='aic')
                lag_order = results.k_ar
                Y_pred[i, :] = results.forecast(X_train[-lag_order:], 1)

            elif method == 'Kernel_RBF':

                kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1),
                                  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3, 1e-4],
                                              "gamma": np.logspace(-2, 2, 5)})
                train_X = X[:-1, :]
                train_y = MultiY[:-1, j]
                kr.fit(train_X, train_y)
                Y_pred[i, j] = kr.predict(X_pred)
            else:
                raise ValueError('Unknown method '+str(method))

            if perc_var:
                Value_pred[i,j] = pred[j] * Y_pred[i, j] / 100 + pred[j]

            else:
                Value_pred[i, j] = pred[j] + Y_pred[i, j]


        X_new = Y_pred[i, :].reshape(1,np.size(returns,axis=1))

    if plot:
        for ts in range(N_TS):
            i = int(ts / N_per_rows)
            j = np.remainder(ts, N_per_rows)

            if Dates is None:
                x_mem = np.arange(N_mem + N_test)
                x = np.arange(N_mem,N_mem + N_test)

                axes[i, j].plot(x_mem, Data[-(N_mem+N_test):,ts])
                axes[i, j].plot(x, Data[-(N_test):,ts], 'o', label='real')
                axes[i, j].plot(x, Value_pred[:,ts], '*', label='pred')
                if columns is None:
                    axes[i, j].set_title('var ' + str(ts) + ' forecast')
                else:
                    axes[i, j].set_title(str(columns[ts]) + ' forecast')

            axes[i, j].legend()


    return Y_pred,Value_pred,Data[-N_test:,:]
