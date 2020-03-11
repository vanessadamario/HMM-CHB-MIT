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


def computeSqDistMat(X1, X2):

    n = np.size(X1,axis=0)
    m = np.size(X2,axis=0)
    Sx1 = np.sum(X1*X1,axis=1)

    SqDistMat = np.repeat(Sx1,m).reshape((m,m))
    Sx2 = np.sum(X2 * X2, axis=1)
    SqDistMat = SqDistMat +  np.repeat(Sx2,n).reshape((n,n))
    SqDistMat = SqDistMat - 2*X1.dot(X2.transpose())

    return SqDistMat

def pred_lgb(X,MultiY, N_test=100, N_val = 10):

    # specify your configurations as a dict
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

    Y_pred_lgb = np.zeros((N_test,np.size(MultiY,axis=1)))

    for i in range(N_test,0,-1):
        for j in range(np.size(MultiY,axis=1)):

            # create dataset for lightgbm
            XX = X[:(np.size(X,axis=0)-i),:]
            YY = MultiY[:(np.size(X,axis=0)-i),j]
            X_val = XX[np.size(XX,axis=0)-N_val:,:]
            X_train = XX[:np.size(XX,axis=0)-N_val,:]
            X_test = X[(np.size(X,axis=0)-i),:].reshape(1,np.size(X[(np.size(X,axis=0)-i),:]))
            lgb_train = lgb.Dataset(X_train, YY[:np.size(XX,axis=0)-10])
            lgb_eval = lgb.Dataset(X_val, YY[np.size(XX,axis=0)-10:], reference=lgb_train)

            # train
            gbm = lgb.train(params,
                            lgb_train,
                            num_boost_round=100,
                            valid_sets=lgb_eval,
                            early_stopping_rounds=10)

            # predict
            Y_pred_lgb[N_test-i,j] = gbm.predict(X_test, num_iteration=gbm.best_iteration)[0]

    return Y_pred_lgb

def pred_LSTM(X,MultiY,plots = False,N_test=100):

    N_obs = np.size(X,axis=0)
    N_obs_train = N_obs-N_test
    X_pred = X[N_obs_train:, :]
    Y_pred_LSTM = np.zeros((N_test,np.size(MultiY,axis=1)))

    for i in range(N_test):
        for j in range(np.size(MultiY,axis=1)):

            XX = np.column_stack((X,MultiY[:,j]))
            N_obs_train = N_obs - N_test + i

            # split into train and test sets
            n_train_hours = int(np.floor(N_obs_train * 2 / 3))
            train = XX[:n_train_hours, :]
            test = XX[n_train_hours:N_obs_train, :]

            # split into input and outputs
            train_X, train_y = train[:, :-1], train[:, -1]
            test_X, test_y = test[:, :-1], test[:, -1]

            # reshape input to be 3D [samples, timesteps, features]
            train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
            test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

            # design network
            model = Sequential()
            model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
            model.add(Dense(1))
            model.compile(loss='mae', optimizer='adam')
            # fit network
            history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                                shuffle=False)
            # plot history
            if plots:
                pyplot.plot(history.history['loss'], label='train')
                pyplot.plot(history.history['val_loss'], label='test')
                pyplot.legend()
                pyplot.show()

            # make a prediction
            pred_X = X_pred[i,:].reshape((1, 1, X_pred[i,:].shape[0]))
            Y_pred_LSTM[i,j] = model.predict(pred_X)

    return Y_pred_LSTM

def pred_VAR_VARMA(data,N_test=100, VARMA = False):

    N_obs = np.size(data,axis=0)
    N_obs_train = N_obs-N_test
    Y_real = data[N_obs_train:, :]
    Y_pred_VAR = np.zeros((N_test,np.size(data,axis=1)))
    Y_pred_VARMA = np.zeros((N_test, np.size(data, axis=1)))

    for i in range(N_test):

        N_obs_train = N_obs - N_test + i
        X_train = data[:N_obs_train, :]

        # VAR
        model = VAR(X_train)
        results = model.fit(maxlags=15, ic='aic')
        lag_order = results.k_ar
        Y_pred_VAR[i,:] = results.forecast(X_train[-lag_order:], 1)

        #VARMA
        if VARMA:
            mod = sm.tsa.VARMAX(X_train, order=(1, 1))
            res = mod.fit(maxiter=1000, disp=False)
            Y_pred_VARMA[i,:] = res.forecast(steps=1)

    if VARMA:
        return Y_pred_VAR, Y_pred_VARMA, Y_real
    else:
        return Y_pred_VAR, Y_real

def pred_Kernel_Ridge(X,MultiY,N_test=100):

    N_obs = np.size(X,axis=0)
    N_obs_train = N_obs-N_test
    X_pred = X[N_obs_train:, :]
    Y_kernel = np.zeros((N_test,np.size(MultiY,axis=1)))

    for i in range(N_test):

        for j in range(np.size(MultiY,axis=1)):

            XX = np.column_stack((X,MultiY[:,j]))
            N_obs_train = N_obs - N_test + i


            # split into train and test sets
            train = XX[:N_obs_train, :]

            # split into input and outputs
            train_X, train_y = train[:, :-1], train[:, -1]

            # Sigma guess
            #SS = computeSqDistMat(train_X, train_X)
            #D = np.sort(SS[np.tril_indices(N_obs_train, -1)])
            #temp = np.sqrt(np.median(D))

            kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1),
                                          param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3,1e-4],
                                          "gamma": np.logspace(-2, 2, 5)})
                                            #  np.logspace(np.log(temp *0.7), np.log(temp * 2),10)})
            kr.fit(train_X, train_y)
            Y_kernel[i, j] = kr.predict(X_pred[i,:].reshape((1, X_pred[i,:].shape[0])))

    return Y_kernel

def pred_HMM_GMM(data,dataset, K,N_test=100, meth = 'viterbi'):

    N_obs = np.size(data,axis=0)
    Y_HMM_GMM_Viterbi = np.zeros((N_test,np.size(data,axis=1)))
    Y_HMM_GMM_Hassan = np.zeros((N_test, np.size(data, axis=1)))
    Y_real = np.zeros((N_test,np.size(data,axis=1)))

    cov_real = {}
    prec_real = {}
    means_real = {}
    cov_Viterbi = {}
    prec_Viterbi = {}
    means_Viterbi = {}
    cov_Hassan = {}
    prec_Hassan = {}
    means_Hassan = {}
    save_res_rec ={}
    save_pred = {}

    for i in range(N_test):

        N_obs_train = N_obs - N_test + i
        X_train = data[:N_obs_train, :]
        mdl = HMM_GraphicalLasso(alpha=1, n_clusters=K, verbose=False, mode='scaled',
                                 warm_restart=True, repetitions=10, n_jobs=-1)


        mmode = 'stability'


        if K == 2:
            alpha_list = np.linspace(45, 65, 20)
        elif K == 5:
            alpha_list = np.linspace(25, 35, 20)
        elif K == 10:
            alpha_list = np.linspace(15, 25, 20)
        elif K == 15:
            alpha_list = np.linspace(5, 15, 20)

        res = cross_validation(mdl,
                               data,
                               params={'alpha': alpha_list,
                                       'n_clusters': [K]},
                               mode=mmode,
                               n_repetitions=1)

        # define three different models

        hmm_gmm = HMM_GraphicalLasso(alpha=res[0][0], n_clusters=K, verbose=False, mode='scaled',
                                     warm_restart=True, repetitions=5, n_jobs=-1)


        hmm_gmm.fit(X_train)

        save_res_rec[str(i)] = results_recap(dataset['states'][:N_obs_train],
                                             hmm_gmm.labels_,
                                             dataset['thetas'],
                                             hmm_gmm.precisions_,
                                             dataset['gammas'][:N_obs_train,:],
                                             hmm_gmm.gammas_)
        save_pred[str(i)] = [hmm_gmm.labels_,hmm_gmm.precisions_,hmm_gmm.covariances_]


        # real
        state = dataset['states'][N_obs_train]
        Y_real[i, :] =  data[N_obs_train, :]
        prec_real[str(i)] = dataset['thetas'][state]
        cov_real[str(i)] = dataset['covariances'][state]
        means_real[str(i)] = dataset['means'][state]



        # predictions
        Y_HMM_GMM_Viterbi[i,:] = hmm_gmm.predict(X_train,method =meth )['pred']
        Y_HMM_GMM_Hassan[i, :] = hmm_gmm.predict(X_train, method='hassan')['pred']
        cov_Viterbi[str(i)] = hmm_gmm.predict(X_train, method=meth)['cov']
        prec_Viterbi[str(i)] = hmm_gmm.predict(X_train, method=meth)['prec']
        means_Viterbi[str(i)] = hmm_gmm.predict(X_train, method=meth)['means']
        print(str(i), hmm_gmm.predict(X_train, method=meth)['means'])
        cov_Hassan[str(i)] = hmm_gmm.predict(X_train, method='hassan')['cov']
        prec_Hassan[str(i)] = hmm_gmm.predict(X_train, method='hassan')['prec']
        means_Hassan[str(i)] = hmm_gmm.predict(X_train, method='hassan')['means']
        print(str(i), hmm_gmm.predict(X_train, method='hassan')['means'])

    results = [Y_HMM_GMM_Viterbi,
               Y_HMM_GMM_Hassan,
               Y_real,
               cov_Viterbi,
               prec_Viterbi,
               means_Viterbi,
               cov_Hassan,
               prec_Hassan,
               means_Hassan,
               prec_real,
               cov_real,
               means_real,
               save_res_rec,
               save_pred ]


    return results
