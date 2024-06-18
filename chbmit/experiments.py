import os
import json
import itertools
import pandas as pd


PATIENT_LIST = [1]
DATA_PATH = '/home/vdamario/output_representations' # path to data representations


def create_time_entries(time, patient_list, data_path):
    if time == -1:
        return [time]
    else:
        if len(patient_list) > 1:
            raise ValueError('You need to provide one patient at the time')
 
        p_str = f'0{patient_list[0]}' if patient_list[0] < 10 else str(patient_list[0])
        training_sets = pd.read_csv(os.path.join(data_path, f'chb{p_str}', 'hours_preseizure_3.csv'), index_col=0)
        if training_sets.shape[0] == 0:
            raise ValueError("No windows to train")
        return [[time, id_train] for id_train in range(training_sets.shape[0])] 

            
experiment_dict = {'patient_list': PATIENT_LIST,
                   'transformation_list': ['log10'], # 'linear', 'log10', 'linear-3-bands'
                   'time': create_time_entries(3, PATIENT_LIST, DATA_PATH),  # if entire time series use -1 else use # hours
                   'state_list': [2, 3, 4, 5, 8, 10, 20],
                   'method_type_list': ['hmm' , 'spectral', 'kmeans'],
                   'hyperparams_hmm': {'alpha': [0.01, 0.1, 1., 5., 20., 30., 40., 50., 70., 100.],
                                       'max_iter': [2000],
                                       'mode': ['scaled'],
                                       'tol': [0.01],
                                       'repetitions': [3],
                                       'verbose': [1]
                                      },
                    'hyperparams_spectral': {'gamma': [1.],
                                             'affinity': ['nearest_neighbors'],
                                             'n_neighbors': [10],
                                             'degree': [3],
                                             'max_iter': [2000],
                                             'coef0': [1],
                                             'verbose': [1],},
                    'hyperparams_kmeans': {'init': ['k-means++'],
                                           'max_iter': [2000],
                                           'tol': [1e-4],
                                           'verbose': [1]
                                          }                 
                  }


class Dataset(object):
    def __init__(self,
                 patient=1,
                 transformation='linear',
                 time=-1):
        self.patient = patient
        self.transformation = transformation
        self.time = time


class HMMHyperParameters(object):
    def __init__(self,
                 n_clusters=8,
                 alpha=25,
                 max_iter=1000,
                 mode='scaled',
                 verbose=1,
                 tol=0.01,
                 clustering='kmeans',
                 probabilities='uniform',
                 repetitions=3):
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.max_iter = max_iter
        self.mode = mode
        self.verbose = verbose
        self.tol = tol
        self.clustering = clustering
        self.probabilities = probabilities
        self.repetitions = repetitions


class SpectralHyperParameters(object):
    """ Class for the spectral clustering method. """
    def __init__(self,
                 n_clusters=8,
                 gamma=1.,
                 affinity='nearest_neighbors',
                 n_neighbors=10,
                 degree=3,
                 coef0=1,
                 n_jobs=-1):
        """ 
        affinity: {'nearest_neighbors', 'rbf', 'precomputed', 'precomputer_nearest_neighbors'}
        assign_labels: {'kmeans', 'discretize', 'cluster_qr'}
        """
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.degree = degree
        self.coef0 = coef0
        self.n_jobs = n_jobs


class KMeansHyperParameters(object): 
    def __init__(self,
                 init='k-means++',
                 n_clusters=8,
                 max_iter=300,
                 tol=1e-4,
                 verbose=0
                 ):
        self.init = init
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        

class Experiment(object):
    def __init__(self,
                 id,
                 output_path,
                 train_completed,
                 method,
                 hyperparams_method,
                 dataset):
        self.id = id
        self.output_path = output_path
        self.train_completed = train_completed
        self.method = method
        self.hyperparams_method = hyperparams_method
        self.dataset = dataset
    

dict_hyper = {'hmm': HMMHyperParameters, 
              'spectral': SpectralHyperParameters, 
              'kmeans': KMeansHyperParameters}


def exp_exists(exp, info):
    """
    Check if experiment exists in json file to avoid duplicate experiments
    """
    dict_new = json.loads(json.dumps(exp, default=lambda o: o.__dict__))
    print(dict_new)
    dict_new_wo_id = {i: dict_new[i]
                      for i in dict_new if (i != 'id' and i != 'output_path' and i != 'train_completed')} # removing three keys
    for idx in info: # iterating on the indexes
        dict_old = info[idx] # for each experiment, remove the three identifiers
        dict_old_wo_id = {i: dict_old[i]
                          for i in dict_old if (i != 'id' and i != 'output_path' and i != 'train_completed')}
        if dict_old_wo_id == dict_new_wo_id:
            return idx
    return False

def generate_experiments(output_path):
    info = {}

    infopath = os.path.join(output_path, 'train.json')
    dirname = os.path.dirname(infopath)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        idx_base = 0
    elif os.path.isfile(infopath):
        with open(infopath) as infile:
            info = json.load(infile)
            if info:
                idx_base = int(list(info.keys())[-1]) + 1 # concatenating to previous experiments
            else:
                idx_base = 0
    else:
        idx_base = 0

    for patient_ in experiment_dict['patient_list']:
        for transf_ in experiment_dict['transformation_list']:
            for time_ in experiment_dict['time']:
                for state_ in experiment_dict['state_list']:
                    for method_ in experiment_dict['method_type_list']:
                    
                        dataset_ = Dataset(patient=patient_, transformation=transf_, time=time_)

                        multiple_hyperparams = bool(experiment_dict[f'hyperparams_{method_}'])
                        combinations = list(itertools.product(*[[(k, v) for v in vs] for k, vs in experiment_dict[f'hyperparams_{method_}'].items()])) if multiple_hyperparams else []
                        combinations = [dict(combo) for combo in combinations] if combinations else [{}]
                        for combo in combinations:

                            combo['n_clusters'] = state_
                            combo['init_params'] = dict(clustering='kmeans', probabilities='uniform')
                                        
                            experiment_ = Experiment(id=idx_base,
                                                    output_path=os.path.join(output_path, f'train_{idx_base}'),
                                                    train_completed=False,
                                                    method=method_,
                                                    hyperparams_method=combo, # method_,
                                                    dataset=dataset_
                                                    )
                            idx = exp_exists(experiment_, info)                        
                            if idx is not False:
                                print(f'The experiment already exists with id: {idx}')
                                continue 
                            s = json.loads(json.dumps(experiment_, default=lambda o: o.__dict__))
                            print(s)
                            info[str(idx_base)] = s
                            idx_base += 1
    with open(infopath, 'w') as outfile:
        json.dump(info, outfile, indent=4)


def get_experiment(exp_specs, data_path): 
    i_ = exp_specs.index[0]
    print(exp_specs.dataset[i_]['patient'])
    p_id = exp_specs.dataset[i_]['patient']
    if p_id < 10:
        patient_id = f'0{p_id}'
    else:
        patient_id = f'{p_id}'
    patient_data_path = os.path.join(data_path, f'chb{patient_id}', exp_specs.dataset[i_]['transformation'])
    
    exp = Experiment(id=exp_specs.id[i_], output_path=exp_specs.output_path[i_], train_completed=bool(exp_specs.train_completed[i_]),
                     method=exp_specs.method[i_], hyperparams_method=exp_specs.hyperparams_method[i_], dataset=exp_specs.dataset[i_])

    return exp, patient_data_path


    

        
        



