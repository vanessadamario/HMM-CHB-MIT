import os
import json
from os.path import join
import pandas as pd
import numpy as np


class Summary():

    def __init__(self, path=None):
        self.data = []
        self.path = path
        self.file_times = None
        self.file_seizures = None
        self.absolute_time = None

    def _load(self, path=None):
        # Loading the summary file. path: must specify if None in constructor
        if self.path is None:
            self.path = path
        with open(self.path, 'r') as f:
            self.data = f.read()

    def _find_absolute_time_file(self): # find_absolute_time in setting_type.py
        list_of_times = []
        filenames = []
        seizure_filenames = []
        list_of_seizures = []
        for line in self.data.split("\n"):
            if line.startswith('File Name: '):
                tmp_filename = line.split('File Name: ')[1]
                filenames.append(tmp_filename)
                
            if line.startswith('File Start Time: '):
                temp_time1 = line.split(': ')[1] 

            elif line.startswith('File End Time: '):
                temp_time2 = line.split(': ')[1]
                list_of_times.append([temp_time1, temp_time2])

            elif line.startswith('Seizure '):
                tmp_split = line.split(': ')
                if tmp_split[0].endswith('Start Time'):
                    seizure_filenames.append(tmp_filename)
                    seizure_time1 = int(tmp_split[1].split(' sec')[0])

                elif tmp_split[0].endswith('End Time'):
                    seizure_time2 = int(tmp_split[1].split(' sec')[0])
                    list_of_seizures.append([seizure_time1, seizure_time2])

        file_times = pd.DataFrame(list_of_times, columns=['File Start','File End'], index=filenames)
        self.file_times = file_times        
        self.file_seizures = pd.DataFrame(list_of_seizures, columns=['Start', 'Stop'], 
                                          index=seizure_filenames)
    
    def _split_times(self): # col_of_time
        start_stop_df = []
        for c in self.file_times.columns:
            time_df = self.file_times[c].str.split(':', expand=True).astype(int)
            time_df.columns =  ['Hours','Minutes','Seconds']
            # start_stop_df.append(time_df)
            
            time_df['Hours'] *= 3600
            time_df['Minutes'] *= 60
            start_stop_df.append(time_df.apply(sum, 1))

        absolute_time = pd.concat(start_stop_df, axis = 1)
        absolute_time = absolute_time.rename(columns={0: 'Absolute Start Time', 
                                                      1: 'Absolute End Time'})
        
        self.absolute_time = absolute_time

    def _make_time_continuous_and_referenced(self): # make time continuous
        idx = self.absolute_time.index
        for c in self.absolute_time:
            add_day_list = np.where(np.diff(self.absolute_time[c].values) < 0)[0] + 1
            for add_day in add_day_list: 
                self.absolute_time[c][idx[add_day]:] += 86400 

        start_time = self.absolute_time["Absolute Start Time"][idx[0]]
        self.absolute_time = self.absolute_time - start_time

    def _set_absolute_time_seizures(self):
        if self.absolute_time is not None and self.file_seizures is not None:
            indexes = self.file_seizures.index
            self.file_seizures = self.file_seizures.add(self.absolute_time.loc[indexes]['Absolute Start Time'], axis=0)

    def convert_start_end_files(self):
        if self.path is not None:
            self._load()
            self._find_absolute_time_file()
            self._split_times()
            self._make_time_continuous_and_referenced()
            self._set_absolute_time_seizures()
        else: 
            raise ValueError('Need to read summary file first.')
        

class Labeling():

    def __init__(self, summary, preictal_window=1800, postictal_window=1800):
        self.preictal_window = preictal_window
        self.postictal_window = postictal_window
        self.file_info = summary.absolute_time
        self.seizure_info = summary.file_seizures
        self.len_acquisition = self.file_info.iloc[-1]['Absolute End Time']

    def _label_time_seizure_proximity(self):
        list_states = []
        list_start = []
        list_stop = []

        for start_, stop_ in self.seizure_info.values:
            list_states.append("Ictal")
            list_start.append(start_)
            list_stop.append(stop_)

            list_states.append("Preictal")
            # TODO: FIX
            # bug: when two seizures are too close
            # the 30 minutes before the second seizure become preictal
            # even if they contained a seizure!!!!!!!!!!!!!
            tmp_start_preictal = start_ - self.preictal_window 
            if  tmp_start_preictal < 0: # talk w Krish
                tmp_start_preictal = 0
            list_start.append(tmp_start_preictal)
            list_stop.append(start_)

            list_states.append("Postictal")
            tmp_stop_postictal = stop_ + self.postictal_window
            if tmp_stop_postictal > self.len_acquisition: # end_file
                tmp_stop_postictal = self.len_acquisition
            list_start.append(stop_)
            list_stop.append(tmp_stop_postictal)

        df = pd.DataFrame(data=[list_states, list_start, list_stop], index=['State', 'Start', 'Stop']).T
        df.sort_values(by='Start', inplace=True)
        self.df = df

    def _label_time_seizure_far(self):
        list_start = []
        list_stop = []

        for i in range(self.df.shape[0]):
            if self.df.iloc[i]['State'] == 'Preictal':
                tmp_start_preictal = self.df.iloc[i]['Start']

                if i == 0 and tmp_start_preictal > 0:  # first interictal period
                    list_start.append(0)
                    list_stop.append(tmp_start_preictal)
                else:
                    if self.df.iloc[i-1]['Stop'] < tmp_start_preictal: # last postictal does not overlap with preictal
                        # so we add an interictal period before it
                        list_start.append(self.df.iloc[i-1]['Stop'])
                        list_stop.append(tmp_start_preictal)

        if self.df.iloc[-1]['State'] == 'Postictal':
            tmp_end_postictal = self.df.iloc[-1]['Stop']
            end_time = self.len_acquisition
            if end_time > tmp_end_postictal:
                list_start.append(tmp_end_postictal)
                list_stop.append(end_time)
        
        df_far =  pd.DataFrame(data=[['Interictal']*len(list_start), list_start, list_stop],
                               index=['State', 'Start', 'Stop']).T
        
        self.df = pd.concat([self.df, df_far])
        self.df.sort_values(by='Start', inplace=True) 

    def _hybrid_stages_finder(self):
        length_df = len(self.df)
        self.df.index = range(length_df)
        
        bm_keep = np.ones(length_df, dtype=bool)
        for i in range(1, length_df - 1): 
            if self.df['Start'][i+1] < self.df['Stop'][i]:
                bm_keep[i:i+2] = False
                self.df = self.df.append({'State': 'Overlap',
                                          'Start': self.df['Start'][i],
                                          'Stop': self.df['Stop'][i+1]}, 
                                          ignore_index=True)
                bm_keep = np.append(bm_keep, True)

        self.df = self.df[bm_keep].sort_values(by='Start')
        self.df.index = range(len(self.df))

    def label_times(self):
        self._label_time_seizure_proximity()
        self._label_time_seizure_far()
        self._hybrid_stages_finder()


def split_in_windows(states_times, files_times, window=10):
    # window has customizable size
    start_window_list = []
    end_window_list = []
    label_window_list = []

    for i_ in range(states_times.shape[0]):
        state = states_times.iloc[i_]['State']
        start_state = states_times.iloc[i_]['Start']
        stop_state = states_times.iloc[i_]['Stop']

        if state == 'Ictal':
            tmp_start = start_state # states_times.iloc[i_]['Start']
            
            while (tmp_start + window) < stop_state: # states_times.iloc[i_]['Stop']:
                # we do not go into another file during seizure
                start_window_list.append(tmp_start)
                end_window_list.append(tmp_start + window)
                tmp_start += window
                label_window_list.append(state)

        else:
            tmp_stop = stop_state # states_times.iloc[i_]['Stop']
            
            while (tmp_stop - window) >= start_state:
                # we check if start and stop are in the same files
                mask_start = files_times['Absolute Start Time'] <= (tmp_stop - window)  # need to think more about this <=
                mask_stop = files_times['Absolute Start Time'] <= tmp_stop 
                same_file = mask_start.equals(mask_stop)
                if same_file:
                    start_window_list.append(tmp_stop - window)
                    end_window_list.append(tmp_stop)
                    label_window_list.append(state)
                    tmp_stop -= window
                else:
                    tmp_stop = files_times['Absolute End Time'][mask_start].iloc[-1]
        
    tmp_df = pd.DataFrame(data=[label_window_list, start_window_list, end_window_list], 
                          index=['State', 'Start', 'End']).T
    tmp_df.sort_values(by='Start', inplace=True)
    return tmp_df


def create_files_w_times(patient, data_path, recordings_path):
    """ 
    This function generate two csv files containing the information of the time of start - end file 
    and the time of the seizures
    """
    # print(f'Patient ID: {patient}')
        
    patient_data_path = join(data_path, f'chb{patient}')
    os.makedirs(patient_data_path, exist_ok=True)
    annotation_path = join(recordings_path, f'chb{patient}', f'chb{patient}-summary.txt')
    SummaryPatient = Summary(path=annotation_path)
    SummaryPatient.convert_start_end_files()
    if patient == '10' or patient == '18': 
        # for these two patients, the last file corresponds to the data acquisition happened on another day
        # we remove this from the dataset
        SummaryPatient.absolute_time =  SummaryPatient.absolute_time[:-1]
        SummaryPatient.file_seizures = SummaryPatient.file_seizures[:-1]
    SummaryPatient.absolute_time.to_csv(join(patient_data_path, f'start_stop_file_{patient}.csv'))
    SummaryPatient.file_seizures.to_csv(join(patient_data_path, f'start_stop_seizure_{patient}.csv'))

    LabelPatient = Labeling(SummaryPatient)
    LabelPatient.label_times()
    LabelPatient.df.to_csv(join(patient_data_path, f'labeled_ts_{patient}.csv'))
    
    df_times = split_in_windows(LabelPatient.df, SummaryPatient.absolute_time, window=10)
    df_times.to_csv(join(patient_data_path, f'split_labeled_ts_{patient}.csv'))