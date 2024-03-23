import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.io as sio
from scipy import signal

import os
from os import makedirs
from os.path import join
from os.path import isfile
from os.path import dirname
from os.path import splitext
from os.path import basename

from normalizers import SavGolFilter
from ishneholterlib import Holter


from IPython import embed

class THEWParser(object):
    """Parses the THEW ECG and annotation data

    Parameters
    ----------
    file_path : str
                Path to .ful file

    Attributes
    ----------
    correction_dict : dict
        Maps a string of frequency to the amplitude correction factor.
    median_filter_params_dict : dict
        Maps a string of frequency to parameter for median filtering.
    freq : int
        Sampling frequency of current .ful file
    correction : float
        Current correction factor
    median_filter_params : list
        Current parameters for median filtering
    file_name : str
        Path to file
    data : np.ndarray
        Time series data of shape length x 8
    """
    def __init__(self, file_path):
        self.correction_dict = {'500': 2.5, '1000': 1.0}
        self.median_filter_params_dict = {'500': [101, 301], '1000': [201, 601]}

        # Determine parameters for filters
        self.freq = 1000
        self.correction = self.correction_dict[f'{self.freq}']
        self.median_filter_params = self.median_filter_params_dict[f'{self.freq}']

        self.data = self._get_raw(file_path)

        self.smoother = SavGolFilter(window_length=51)

    def apply_smoothing(self, raw: np.ndarray):
        """ Applies a Savitzky-Golay filter independently on all channels inplace.

        Parameters
        ----------
        raw : np.ndarray
            Timeseries of shape length x 8

        `Source <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html>`_

        """
        #print("Apply smoothing")
        for channel in range(raw.shape[1]):
            self.smoother.transform(None, raw.T, channel)

    def apply_butter(self, raw: np.ndarray, Wn: float):
        """ Applies a Butterwoth bandpass filter independently on all channels inplace.

        Parameters
        ----------
        raw : np.ndarray
            Timeseries of shape length x 8
        Wn : float
            Critial frequency

        `Source <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html>`_
        """
        #print("Apply bandpass filter")
        for channel in range(raw.shape[1]):
            sos = signal.butter(4, Wn, 'bandpass', analog=False, output='sos')
            raw[:, channel] = signal.sosfilt(sos, raw[:, channel])

    def apply_median(self, raw: np.ndarray):
        """ Applies median subtraction filtering according to `self.median_filter_params`.
        In a first step a moving median of the TS is calculated. On the resulting TS, a 
        second moving median will be applied. The result is subtracted from the original
        TS.

        Parameters
        ----------
        raw : np.ndarray
            Timeseries of shape length x num_channels
        """
        #print("Apply median subtraction")
        for channel in range(raw.shape[1]):
            med_1 = signal.medfilt(raw[:, channel], self.median_filter_params[0])
            raw[:, channel] = raw[:, channel] - signal.medfilt(med_1, self.median_filter_params[1])

    def apply_winsorizing(self, raw: np.ndarray, q_low, q_high):
        """ Applied winsorizing according to `q_loq` and `q_high`.

        Parameters
        ----------
        raw : np.ndarray
            Timeseries of shape length x num_channels

        q_low : float
            Percentile to which low outliers to be clipped.
        q_high : float
            Percentile to which high outliers to be clipped.
        """
        #print("Apply winsorizing")
        percentiles_low = np.percentile(raw, q_low, axis=0, keepdims=True)
        percentiles_high = np.percentile(raw, q_high, axis=0, keepdims=True)
        for channel in range(raw.shape[1]):
            indices_low = raw[:, channel] <= percentiles_low[0, channel]
            indices_high = raw[:, channel] >= percentiles_high[0, channel]
            raw[indices_low, channel] = percentiles_low[0, channel]
            raw[indices_high, channel] = percentiles_high[0, channel]


    def _get_raw(self, f: str):
        data = Holter(f)
        data.load_data()
        data.load_ann(f.replace('.ecg', '.ann'))
        
        lead_list = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1',
                     'V2', 'V3', 'V4', 'V5', 'V6']
        ecg_signal = np.asarray([[l.data for l in data.lead if str(l) == lead] for lead in lead_list])
        ecg_signal = ecg_signal.reshape(12, -1).T
        
        # Set annotation data
        self.anns = data.beat_anns
        return ecg_signal

    def downsample(self, factor: int = 2):
        """ Downsamples `self.data` by a factor of `factor`.
        
        Parameters
        ----------
        factor : int
            Downsampling factor (default 2)
        """
        print("Downsample data")
        self.data = signal.decimate(self.data, factor, axis=0)

class THEWPatientHelper():
    def __init__(self, file_path=join('/fs/pool/pool-mlsb/eth_group_share/Projects/IschemiaPrediction/data/', 'THEW', 'data/clinicalData/', 'E-OTH-12-0927-015.csv')):
        print(file_path)
        thew_data = pd.read_csv(file_path, delimiter=',')
        thew_data.rename(columns={'Spect Summary': 'ischemia',
                                  'termination reason': 'term_reason'},
                         inplace=True)
        print("before ischemia drop", thew_data.shape)
        thew_data.dropna(subset=['ischemia'], inplace=True)
        print("after ischemia drop", thew_data.shape)
        thew_data.loc[:, 'post_pred'] = thew_data.exercise_ECG_result.apply(lambda x: 0 if x == 'Non-ischemic' 
                                                                            else (1 if x == 'Ischemic' else (
                                                                            0.5 if x == 'Borderline ischemic' else None)))
        #thew_data = thew_data.query('term_reason == "Maximum HR"')
        print("before post_pred drop", thew_data.shape)
        #thew_data.dropna(subset=['post_pred'], inplace=True)
        print("after post_pred drop", thew_data.shape)
        print("before height drop", thew_data.shape)
        thew_data.dropna(subset=['height'], inplace=True)
        print("after height drop", thew_data.shape)
        print("before weight drop", thew_data.shape)
        thew_data.dropna(subset=['weight'], inplace=True)
        print("after weight drop", thew_data.shape)

        thew_data.loc[:, 'study'] = 'THEW'
        thew_data.ischemia = thew_data.ischemia.apply(lambda x: 1 if x == 'Ischemic' else 0)
        thew_data.gender = thew_data.gender.apply(lambda x: 2 if x == 'Female' else 1)
        self.clin_data = thew_data
        
    def get_label(self, pat_id):
        return self.clin_data.query('ID == @pat_id')['ischemia'].values[0]
    
    def get_NIP_data(self, pat_id, CAD_fill=0.0):
        feat_names = ['age', 'gender', 'height', 'rest_HR', 'weight',
              'rest_systolic_BP', 'rest_diastolic_BP', 'AnamnesebekannteKHK']
        #self.clin_data.loc[:, 'AnamnesebekannteKHK'] = CAD_fill
        self.clin_data.loc[:, 'AnamnesebekannteKHK'] = self.clin_data.loc[:, 'hx_cad'].astype(int)
        return self.clin_data.query('ID == @pat_id')[feat_names].values


def get_HR(beat_anns, sampling_rate=1000):
    one_min_length = sampling_rate * 60

    measured = 0
    num_beats = 0
    hr = []
    for beat in beat_anns:
        pos = beat['toc']
        last_beat = beat
        measured += pos
        num_beats += 1
        if measured >= one_min_length:
            hr.append({'hr': num_beats, 'loc': last_beat['samp_num'] // 2})
            measured = 0
            num_beats = 0
    return hr

def get_total_lengths(freq):
    one_min_length = freq * 60
    pre_tot_length = 2 * one_min_length
    stress_tot_length = 2 * one_min_length
    rec_tot_length = 3 * one_min_length
    
    return pre_tot_length, stress_tot_length, rec_tot_length

def get_input_seq(X, npz_file_path, pre_length_sec=2, stress_length_sec=6,
                  rec_length_sec=2, freq=500, leads=[11]):
    # length of resulting TS
    one_sec_length = freq
    pre_length = pre_length_sec * one_sec_length
    stress_length = stress_length_sec * one_sec_length
    rec_length = rec_length_sec * one_sec_length
    
    # Lengths of windows from which 2-6-2 will be
    # extracted.
    pre_tot_length, stress_tot_length, rec_tot_length = get_total_lengths(freq)
    
    # Read original ECG data
    parser = THEWParser(npz_file_path)
    hrs = get_HR(parser.anns)

    # Define PRE window
    pre_window = [0, pre_tot_length]
    
    # Define STRESS window
    # Sort HRs decreasingly
    hrs = sorted(hrs, key=lambda x: x['hr'], reverse=True)
    stress_end = hrs[0]['loc']
    stress_window = [stress_end - stress_tot_length, stress_end]
    
    # Define RECOVERY window
    rec_end = X.shape[1]
    rec_window = [rec_end - rec_tot_length, rec_end]
    
    # Extract 2-6-2 windows
    # First compute the max number of sequences we can extract
    max_seq_num = 1000
    for (window, req_length) in zip([pre_window, stress_window, rec_window],
                                    [pre_length, stress_length, rec_length]):
        win_length = window[1] - window[0]
        max_seq_num = np.min([max_seq_num, win_length // req_length])
       
    # Then extract them
    res_length = pre_length + stress_length + rec_length
    results = np.zeros((max_seq_num, len(leads), res_length))
    pre_begin = pre_window[0]
    stress_begin = stress_window[0]
    rec_begin = rec_window[0]    
    for i in range(max_seq_num):
        # Compute start indices
        pre_start = pre_begin + (i * pre_length)
        stress_start = stress_begin + (i * stress_length)
        rec_start = rec_begin + (i * rec_length)
        
        # Extract time series
        pre_seq = X[leads, pre_start:pre_start + pre_length]
        stress_seq = X[leads, stress_start:stress_start + stress_length]
        rec_seq = X[leads, rec_start:rec_start + rec_length]
        
        # Combine
        result_seq = np.concatenate((pre_seq, stress_seq, rec_seq), 1)
        if result_seq.shape[1] == 5000:
            results[i] = result_seq
    
    return results
