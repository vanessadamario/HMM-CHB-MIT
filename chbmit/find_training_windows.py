import os
from os.path import join, dirname
import numpy as np
import pandas as pd


def _creating_dataset(timepoint_seizure, training_points, path_fileseizure, preseizure=True):
    """ 
    Function to test that txt_file_window works correctly
    Given the window from which we start our dataset, we include n_windows_included to the dataset.
    We need to specify if preseizure (we go backward) or postseizure (we go forward).
    filepath variable is important to look for the correct patient. 
    """
    filesdir = dirname(path_fileseizure)
    fileseizure = f"{path_fileseizure.split('/')[-1].split('.')[0]}.csv"
    filelist = np.array(sorted(os.listdir(filesdir)))
    id_seizure = np.argwhere(filelist == fileseizure)[0][0]
    filelist =  filelist[:id_seizure+1] if preseizure else filelist

    for id_, filename in enumerate(filelist):
        
        tmp_data = pd.read_csv(join(filesdir, filename), index_col=0)
        if id_ == 0:
            data = tmp_data
        else:
            data = pd.concat([data, tmp_data])

    if preseizure:
        return data[timepoint_seizure - training_points: timepoint_seizure]
    else:
        return data[timepoint_seizure: timepoint_seizure + training_points]
    


        

def txt_file_windows(path_output_representation, transformation='log10', hours=3, patient_id='04', window_size=10):
    
    path_data = join(path_output_representation, f'chb{patient_id}', transformation)
    
    start_stop_seizure_filename = join(dirname(path_data), f'start_stop_seizure_{patient_id}.csv')
    start_stop_seizure_df = pd.read_csv(start_stop_seizure_filename, index_col=0)
    time_anticipating_last_seizure = np.append(start_stop_seizure_df['Start'][0], np.diff(start_stop_seizure_df['Start']))
    bm_candidate_training = time_anticipating_last_seizure > 3600 * hours

    df = start_stop_seizure_df[bm_candidate_training]

    windows_file = pd.read_csv(join(dirname(path_data), f'split_labeled_ts_{patient_id}.csv'))

    df_training_windows = {}
    ID_training = []
    df_training_windows['Seizure File'] = []
    df_training_windows['Start Seizure'] = []
    df_training_windows['Stop Seizure'] = []

    df_training_windows['ID window start'] = [] # index for start
    df_training_windows['ID window stop'] = [] # index at which we stop training

    counter = 0
    
    for i, seizure_filename in enumerate(df.index): # we loop all seizures 
        print(f'\n{seizure_filename}')
        start_seizure = df.iloc[i]['Start']
        stop_seizure = df.iloc[i]['Stop']
    
        id_window_start_ = np.where(windows_file['Start'] <= start_seizure)[0][-1] ## first ictal window
        id_window_stop_ = np.where(windows_file['End'] > stop_seizure)[0][0]

        ids_preseizure = int(3600/window_size*hours)
        ids_postseizure = int(3600/(2*window_size)) + (id_window_stop_ - id_window_start_) # postictal and ictal
        
        ids_start_train = id_window_start_ - int(3600/window_size*hours)
        ids_stop_train = id_window_start_ + ids_postseizure

        if not np.array_equal(np.unique( windows_file.iloc[id_window_start_ - ids_preseizure : id_window_start_]['State']),
                              np.array(['Interictal', 'Preictal'])):
            
            continue

        ID_training.append(counter)

        df_training_windows['Seizure File'].append(seizure_filename)
        df_training_windows['Start Seizure'].append(start_seizure)
        df_training_windows['Stop Seizure'].append(stop_seizure)
        df_training_windows['ID window start'].append(ids_start_train)
        df_training_windows['ID window stop'].append(ids_stop_train)
        counter += 1

    if len(df_training_windows) > 0:
        df_training_windows = pd.DataFrame(df_training_windows, index=ID_training)
        print(df_training_windows.head())

    df_training_windows.to_csv(join(dirname(path_data), f'hours_preseizure_{hours}.csv'))

    return
                