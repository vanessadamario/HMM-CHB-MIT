import os
import mne
import pandas as pd
import numpy as np
from os.path import join, exists


transformation_dict = {'linear': lambda x: x,
                       'log10': lambda x: np.log(x)}

def build_representation(patient, recordings_path, output_path, transformation):
    """ 
    This function creates the data representation that will be provided to the machine learning algorithm.
    The transformations are linear and logarithm 10 
    :param patient: str, patient ID
    :param recordings_path: str, path to the physionet folder containing the list of patients 
    :param output_path: str, path to the output folder containing a subfolder per patient
    :param transformation: str, specify the type of transformation applied on the time series
    """

    path_output_dirname = join(output_path, f"chb{patient}", transformation)
    os.makedirs(path_output_dirname, exist_ok=True)

    df_split = pd.read_csv(join(output_path, f"chb{patient}", f"split_labeled_ts_{patient}.csv"))
    df_split = df_split[["State", "Start", "End"]] # this contain the 10 seconds windows

    df_files = pd.read_csv(join(output_path,f"chb{patient}", f"start_stop_file_{patient}.csv"), index_col=0) # this contains the files

    list_chbfiles = [chbfile for chbfile in os.listdir(join(recordings_path, f"chb{patient}"))
                        if chbfile.endswith('.edf')]
    list_chbfiles = sorted(list_chbfiles)

    for chbfile in list_chbfiles: 
        raw = mne.io.read_raw(join(recordings_path, f"chb{patient}", chbfile),
                    preload=True)
        raw = raw.filter(l_freq=0.1, h_freq=40., n_jobs=-1) # filter between 0.1 Hz and 40 Hz

        bm1 = df_split['Start'] - df_files['Absolute Start Time'][chbfile] >= 0
        bm2 = df_split['End'] - df_files['Absolute End Time'][chbfile] <= 0 
        
        df_split_file = df_split[bm1 * bm2] # we select all windows that are contained in file chbfile
        df_split_file_cp = df_split_file.copy()
        df_split_file_cp.loc[:, ['Start', 'End']] = df_split_file_cp.loc[:, ['Start', 'End']] - df_files['Absolute Start Time'][chbfile] # we reset the starting time to zero

        spectrum_list = []
        for i_ in range(df_split_file_cp.shape[0]):   # for all windows, we create the representation
            crop = raw.copy().crop(tmin=df_split_file_cp.iloc[i_]['Start'], 
                    tmax=df_split_file_cp.iloc[i_]['End'] - 1/raw.info['sfreq'])
            spectrum = crop.compute_psd(fmin=0.1, fmax=40.)
            spectrum_list.append(transformation_dict[transformation](spectrum[:]).sum(axis=1))
        
        df = pd.DataFrame(spectrum_list, columns=raw.info.ch_names, index=df_split_file_cp.index)
        df = pd.concat([df, df_split_file_cp['State'][:len(df)]], axis=1)
        df.to_csv(join(path_output_dirname, f"{chbfile.split('.edf')[0]}.csv"))
    