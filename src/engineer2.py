#!/usr/bin/python3

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as pp
import scipy.signal as sig
from biosppy import ecg

def get_pulse2(ecgin):
    bf = sig.butter(5, [0.1, 10], btype='bandpass', fs=256, output='sos')
    out1 = ecg.ecg(signal=ecgin, sampling_rate=256)
    # out2 = ecg.ecg(signal=sig.sosfilt(bf, ecgin), sampling_rate=256)
    pp.plot(out1['heart_rate_ts'], out1['heart_rate'])
    # pp.plot(out2['heart_rate_ts'], out2['heart_rate'])
    pp.show()
    print(out1['heart_rate_ts'][1:]-out1['heart_rate_ts'][:-1])
    
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

get_pulse2(data['ecg'])
# pp.plot(data['time'], get_pulse(data['ecg']))
# pp.show()

