import os
import argparse
import pandas as pd
from chbmit import experiments
from joblib import Parallel, delayed


output_path = '/home/vdamario/output_HMM_CHBMIT' # path to experiment results
data_path = '/home/vdamario/output_representations' # path to data representations
recordings_path = '/home/data/physionet.org/files/chbmit/1.0.0' # path to EEG recordings
logs_path = '/home/vdamario/HMM-CHB-MIT/logfiles'

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_index', 
                    nargs='?', 
                    const=0, 
                    type=int, 
                    help='Experiment index for training and test - Irrelevant otherwise')

parser.add_argument('--run', type=str, 
                    required=True, 
                    choices=['gen_exp', 'gen_rep', 'train', 'set_completed', 'set_train_windows'],
                    help='Action to be performed among data generation, training, testing')

parser.add_argument('--transformation', 
                    type=str, 
                    required=False, 
                    choices=['linear', 'log10'], # , '3-bands-linear'],
                    help='Representation for the data, required when creating the data representation')


FLAGS = parser.parse_args()
print(f'Test: {FLAGS.run}, {FLAGS.experiment_index}')


def generate_experiments(id):
    """ Generation of the experiments. """
    experiments.generate_experiments(output_path=output_path)
    

def generate_representations(id):
    from chbmit import chunks_processing, open_edf_preprocessing
    patient_ids = [f'0{i}' for i in range(1, 10)] + [str(i) for i in range(10, 24)]
    # TODO a control is needed here
    # Parallel(n_jobs=-1)(delayed(chunks_processing.create_files_w_times)(p, data_path, recordings_path) for p in patient_ids)
    Parallel(n_jobs=-1)(delayed(open_edf_preprocessing.build_representation)(p, recordings_path, data_path, FLAGS.transformation) for p in patient_ids) 
    

def set_experiment_as_completed(id):
    from chbmit import set_as_completed
    print("Work in progress")
    set_as_completed.mark_complete(os.path.join(output_path, 'train.json'), logs_path)


def generate_file_for_windows_training(id):
    from chbmit import find_training_windows
    patient_ids = [f'0{i}' for i in range(1, 10)] + [str(i) for i in range(10, 24)]
    for p in patient_ids:
        find_training_windows.txt_file_windows(data_path, hours=3, patient_id=p)


def train(id):
    """ Run training for experiment id. """
    from chbmit import train
    scheduler = pd.read_json(os.path.join(output_path, 'train.json')).T

    if scheduler[scheduler['id'] == id]['train_completed'][id]:
        return
    exp, path_tr_data = experiments.get_experiment(scheduler[scheduler['id'] == id], data_path)
    train.train_method(exp, path_tr_data)

    # create a text file after training completion (if Parallel, overwrite of the file could be an issue)
    # update status all together
    return
    

switcher = {'gen_exp': generate_experiments,
            'gen_rep': generate_representations,
            'set_train_windows': generate_file_for_windows_training,
            'set_completed': set_experiment_as_completed,
            'train': train}


switcher[FLAGS.run](FLAGS.experiment_index)