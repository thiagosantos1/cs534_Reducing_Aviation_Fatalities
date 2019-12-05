#!/usr/bin/python3

from biosppy import ecg
import matplotlib.pyplot as pp
import numpy as np
import pandas as pd
import sys
from interpolate import interpolate

def get_pulse(ecgin):
    out = ecg.ecg(signal=ecgin, sampling_rate=256, show=False)
    return out


data = pd.read_csv(sys.argv[1])
pulse = get_pulse(data['ecg'])
# pp.plot(pulse['heart_rate_ts'], pulse['heart_rate'])
pulse2 = interpolate(pulse['heart_rate'], pulse['heart_rate_ts'], data['time'])
norm = np.log(pulse2)
norm = norm/np.mean(norm)
pp.plot(data['time'], np.log(norm))
pp.show()

                 
