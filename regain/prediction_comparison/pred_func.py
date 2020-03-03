import lightgbm as lgb
import numpy as np

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
