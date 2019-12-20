#!/usr/bin/python3 

# Feature engineering for the dataset
# Loads all train/test splited csvs and does the feature enginnering
# Or can do it just for one set - For debuging process
# 

import pandas as pd
import numpy as np
import sys
import subprocess
import glob, os
from scipy import signal
import matplotlib.pyplot as plt
from biosppy.signals import ecg, resp
import scipy.signal as sig
from tools import *
import re
from sklearn import preprocessing
import simdkalman
import numpy.random as random
from sklearn import preprocessing


# Idea is to create a new dataset, breaking the time siries in a fixed frequency
# This would create more meaningful data

def respiration(data, d_time, frequency=256):

    b, a = signal.butter(8,0.05)
    # get the raw data and filter it
    respiration = signal.filtfilt(b, a, data, padlen=150)
    # plt.plot(respiration)
    # plt.show()

    # we can then get the amplitude and clear signal of the data
    # and also the respiration rate
    out = resp.resp(respiration,sampling_rate=frequency, show=False)
    # plt.plot(out['resp_rate_ts'], out['resp_rate'])
    # plt.ylabel('Respiratory frequency [Hz]')
    # plt.xlabel('Time [s]');
    # plt.show()

    # In this case, shape is modify, since we're breaking into frequency
    resp_rate_ts = out['resp_rate_ts']
    resp_rate = out['resp_rate']
    resp_filtered = out['filtered']

    # What to Return ? ? ? ?
    
    resp_rate_interpolated = int_spline(resp_rate, resp_rate_ts, d_time)
    return lognorm2norm(resp_rate_interpolated)



def ecg_(data, d_time, frequency=256):

    # filter the data
    b, a = signal.butter(8,0.01)
    y = signal.filtfilt(b, a, data, padlen=150)

    # plt.plot(y[6000:14024])
    # plt.show()

    out = ecg.ecg(signal=data, sampling_rate=frequency, show=False)

    # plt.plot(out['heart_rate_ts'], out['heart_rate'])
    # plt.ylabel('Heart Rate (BPM)')
    # plt.xlabel('Time [s]');


    heart_rate_ts = out['heart_rate_ts']
    heart_rate = out['heart_rate']
    heart_filtered = out['filtered']
    # What to Return ? ? ? ?
    heart_rate_interpolated = int_spline(heart_rate, heart_rate_ts, d_time)
    return lognorm2norm(heart_rate_interpolated)


def eeg_(dataset, frequency=256):
    
    # There're diffent ways to handle eeg
    eeg_signals = {}
    eeg_signals['fp1_f7']  = dataset['eeg_fp1'] - dataset['eeg_f7']
    eeg_signals['f7_t3']   = dataset['eeg_f7'] - dataset['eeg_t3']
    eeg_signals['t3_t5']   = dataset['eeg_t3'] - dataset['eeg_t5']
    eeg_signals['t5_o1']   = dataset['eeg_t5'] - dataset['eeg_o1']
    eeg_signals['fp1_f3']  = dataset['eeg_fp1'] - dataset['eeg_f3']
    eeg_signals['f3_c3']   = dataset['eeg_f3'] - dataset['eeg_c3']
    eeg_signals['c3_p3']   = dataset['eeg_c3'] - dataset['eeg_p3']
    eeg_signals['p3_o1']   = dataset['eeg_p3'] - dataset['eeg_o1']

    eeg_signals['fz_cz']   = dataset['eeg_fz'] - dataset['eeg_cz']
    eeg_signals['cz_pz']   = dataset['eeg_cz'] - dataset['eeg_pz']
    eeg_signals['pz_poz']  = dataset['eeg_pz'] - dataset['eeg_poz']

    eeg_signals['fp2_f8']  = dataset['eeg_fp2'] - dataset['eeg_f8']
    eeg_signals['f8_t4']   = dataset['eeg_f8'] - dataset['eeg_t4']
    eeg_signals['t4_t6']   = dataset['eeg_t4'] - dataset['eeg_t6']
    eeg_signals['t6_o2']   = dataset['eeg_t6'] - dataset['eeg_o2']
    eeg_signals['fp2_f4']  = dataset['eeg_fp2'] - dataset['eeg_f4']
    eeg_signals['f4_c4']   = dataset['eeg_f4'] - dataset['eeg_c4']
    eeg_signals['c4_p4']   = dataset['eeg_c4'] - dataset['eeg_p4']
    eeg_signals['p4_o2']   = dataset['eeg_p4'] - dataset['eeg_o2']

    return eeg_signals


def get_pulse(ecg, window=256, ratio=2.0, thresh=0.5):
    n = ecg.shape[0]
    bf = sig.butter(5, [0.1, 10], btype='bandpass', fs=256, output='sos')    
    filtered = sig.sosfilt(bf, ecg)
    scale = np.empty(n)
    for i in range(window):
        scale[i] = np.max(np.abs(filtered[0:i+window]))
    for i in range(window,n-window):
        scale[i] = np.max(np.abs(filtered[i-window:i+window]))
    for i in range(n-window,n):
        scale[i] = np.max(np.abs(filtered[i-window:n]))
    pulse = np.empty(n)
    pr = 1
    pulse[0] = pr
    tu0 = -128
    tu1 = 0
    for i in range(1,n):
        if filtered[i-1]/scale[i-1] <= thresh and filtered[i]/scale[i] > thresh:
            npr = 256/(i-tu0)
            if npr > pr/ratio and npr < pr*ratio:
                pr = npr
            tu1=tu0
            tu0=i
        pulse[i] = pr
    return pulse/np.mean(pulse)


def feature_engineering(file):
    engineered_data = pd.DataFrame()
    print("Preprocessing file: ",file)
    if os.path.exists(file):
        try:
            dataset = pd.read_csv(file)

            engineered_data['resp_rate']        = respiration(dataset['r'], dataset['time'])
            engineered_data['heart_rate']       = ecg_(dataset['ecg'], dataset['time'])
            engineered_data['eeg_sig']           = kalman(file)
            # Maybe remove ??
            eeg_sig = eeg_(dataset)
            for key in eeg_sig:
                engineered_data[key] = eeg_sig[key]

            engineered_data['gsr']    = preprocessing.scale(dataset['gsr'])
            engineered_data['target'] = dataset['event']

            return engineered_data
            
        except OSError:
            print("Could not open/read file:", file)
            sys.exit()
    else:
        print("File: ", file, " Does not exist")
        sys.exit()



def kalman(file):

    print("Kalman filter Preprocessing file: ",file)
    if os.path.exists(file):
        try:
            dataset = pd.read_csv(file)
            time = dataset['time']
            # dataset['event'] = pd.factorize(dataset['event'])[0]
            # dataset['experiment'] = pd.factorize(dataset['experiment'])[0]
            # dataset = dataset.drop(columns=['event', 'experiment', 'crew', 'time', 'seat' ])

            #drop_these = list(set(list(dataset)) - set(['event', 'experiment', 'crew', 'time', 'seat' ]))
            drop_these = list(set(['event', 'experiment', 'crew', 'seat', 'time' ]))
            # data_time = dataset['time']
            # data_ecg = dataset['ecg']
            data_ = dataset.drop(drop_these, axis = 1).to_numpy()
            lower_b = 0 
            upper_b = 0 
            increment = 1000
            kalman_filtered = np.array(([]))
            for i in range(0,data_.shape[0]// increment, 1):
                lower_b = upper_b
                upper_b += increment
                data = preprocessing.scale(data_[lower_b:upper_b, 1:21])
                #data_ = data_.to_numpy()[0:300, 1::]
                # print(data_time.shape)
                # print(data_ecg.shape)
                # data_comb = np.stack((data_time,data_ecg))
                # data = preprocessing.scale(data_)
                # plt.plot(data)
                # plt.show()

                # data = data.reshape((data.shape[0],data.shape[0]))

                kf = simdkalman.KalmanFilter(
                    state_transition = np.array([[1,1],[0,1]]),
                    process_noise = np.diag([0.1, 0.01]),
                    observation_model = np.array([[1,0]]),
                    observation_noise = 1.0)
                

                kf = kf.em(data, n_iter=10)

                # smooth and explain existing data
                smoothed = kf.smooth(data)
                # predict new data
                pred = kf.predict(data, 15)
                #plot_kalman(smoothed,pred, data)
                smoothed_obs = smoothed.observations.mean[0,:]
                # plt.plot(smoothed_obs)
                # plt.show()
                kalman_filtered = np.append(kalman_filtered, smoothed_obs)

            k_inter = int_spline(kalman_filtered, np.arange(0, kalman_filtered.shape[0]), time)
            
            return k_inter
            
        except OSError:
            print("Could not open/read file:", file)
            sys.exit()
    else:
        print("File: ", file, " Does not exist")
        sys.exit()

def plot_kalman(smoothed, pred, data):

    for i in range(3):

        _, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        plt.title("time series %d" % (i+1))

        x = np.arange(0, data.shape[1])

        ax1.plot(x, data[i,:], 'b.', label="data")

        smoothed_obs = smoothed.observations.mean[i,:]
        obs_stdev = np.sqrt(smoothed.observations.cov[i,:])
        ax1.plot(x, smoothed_obs, 'r-', label="smoothed")
        ax1.plot(x, smoothed_obs - obs_stdev, 'k--', label="67% confidence")
        ax1.plot(x, smoothed_obs + obs_stdev, 'k--')

        x_pred = np.arange(data.shape[1], data.shape[1]+pred.observations.mean.shape[1])
        y_pred = pred.observations.mean[i,:]
        pred_stdev = np.sqrt(pred.observations.cov[i,:])
        ax1.plot(x_pred, y_pred, 'b-', label="predicted")
        ax1.plot(x_pred, y_pred + pred_stdev, 'k--')
        ax1.plot(x_pred, y_pred - pred_stdev, 'k--')
        ax1.legend()

        trend = smoothed.states.mean[i,:,1]
        trend_stdev = np.sqrt(smoothed.states.cov[i,:,1,1])
        ax2.plot(x, trend, 'g-', label="trend")
        ax2.plot(x, trend - trend_stdev, 'k--', label="67% confidence")
        ax2.plot(x, trend + trend_stdev, 'k--')

        trend_pred = pred.states.mean[i,:,1]
        trend_pred_stdev = np.sqrt(pred.states.cov[i,:,1,1])
        ax2.plot(x_pred, trend_pred, 'b-', label='predicted')
        ax2.plot(x_pred, trend_pred + trend_pred_stdev, 'k--')
        ax2.plot(x_pred, trend_pred - trend_pred_stdev, 'k--')
        ax2.legend()

        plt.show()

if __name__ == '__main__':

    dataset_path = "../data"

    # for a specific data/series input
    if len(sys.argv) > 1:
        file = sys.argv[1]
        engineered_data_1sample = feature_engineering(file)
        #engineered_data_1sample = kalman(file)
        engineered_data_1sample['target'] = pd.factorize(engineered_data_1sample['target'])[0]
        print("\n ##### Done with Feature Engineering #####\nSaving file.....\n")
        # plt.plot(engineered_data_1sample['eeg_sig'])
        # plt.show()
        engineered_data_1sample.to_csv(r'../data/engineered_train_1sample.csv', index = None, header=True)
    
        
    else:

        os.chdir(dataset_path)
        engineered_full_data = pd.DataFrame()
        for file in glob.glob("train_*.csv"):
            try:
                data = pd.read_csv(file)
                engineered_data_sample = feature_engineering(file)
                engineered_full_data = engineered_full_data.append(engineered_data_sample)
                # plt.plot(engineered_data_sample['eeg_sig'])
                # plt.show()
                # Then for each, combine as new features, instead of new rowns
            except OSError:
                print("Could not open/read file:", file)
                sys.exit()

        engineered_full_data['target'] = pd.factorize(engineered_full_data['target'])[0]
        out = '../data/engineered_train_full_kalman_2.csv'
        print("\nSaving to CSV ", out, "......\n")
        engineered_full_data.to_csv(r'../data/engineered_train_full_kalman_2.csv', index = None, header=True)

        print("Preprocessing Done Successfully")

    ###### 
    # Save the new engineered_data to file, and use it as input for any model to be tried
