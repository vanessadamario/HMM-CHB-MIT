import os
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering

from regain.hmm.hmm_graphical_lasso import HMM_GraphicalLasso 
from regain.hmm.utils import cross_validation

from sklearn.metrics import adjusted_rand_score


class DataLoader():
    def __init__(self,
                 p_dataset_path,
                 standandize=True,
                 path_save_scaler=None):
        self.p_dataset_path = p_dataset_path
        self.standardize = standandize
        self.scaler = None
        self.path_save_scaler=None

    def merge(self):
        # TODO we need to exclude the heart 
        """ 
        Merge all files for all patients 
        """
        print(f'Path to the patient folder: {self.p_dataset_path}')
        for i_, chbfile in enumerate([f for f in sorted(os.listdir(self.p_dataset_path)) if f.startswith('chb')]):

            tmp = pd.read_csv(os.path.join(self.p_dataset_path, chbfile), index_col=0)

            if 'ECG' in tmp.columns:
                tmp = tmp.drop('ECG', axis=1)

            print(tmp.head())
            if i_ == 0:
                y = tmp['State'].values
                X = tmp.drop('State', axis=1).values
                
            else:
                y = np.append(y, tmp['State'].values)
                X = np.vstack((X, tmp.drop('State', axis=1).values))


        if self.standardize:
            # apply standardization
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(X)
            self.scaler = scaler
            if self.path_save_scaler is not None:
                pkl.dump(scaler, open(os.path.join(self.path_save_scaler, 'scaler.pkl'), 'wb'))
            X = scaled_data

        return X, y
    

def train_method(experiment, path_training_data):
    ###################################################################################
    # TODO we need to give path to scaler!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ###################################################################################
    
    loader = DataLoader(p_dataset_path=path_training_data) 
    X, _ = loader.merge()

    print(f'Shape of the input matrix {X.shape}')

    os.makedirs(experiment.output_path, exist_ok=True)
    filename = os.path.join(experiment.output_path, 'model.pkl')

    if experiment.method == 'hmm':
        print('HMM')
        hmm_gmm = HMM_GraphicalLasso(**experiment.hyperparams_method) # , n_jobs=-1)
        print(experiment.hyperparams_method)
        hmm_gmm.fit(X)
        pkl.dump(hmm_gmm, open(filename, 'wb'))
    
    elif experiment.method == 'kmeans':
        print('KMEANS')
        return
    
    elif experiment.method == 'spectral':
        print('SPECTRAL')
        return
    
    else:
        raise ValueError('The method is not recognized. Select one among \'hmm\', \'kmeans\', and \'spectral\'')
    
    completion_file = open(os.path.join(experiment.output_path, 'train_completed.txt'), 'w') 

    return