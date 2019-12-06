#!/usr/bin/python3

from biosppy import ecg
import matplotlib.pyplot as pp
import numpy as np
import pandas as pd
import sys
from tools import int_cubic, int_spline, lognorm2norm

def get_pulse(ecgin):
    out = ecg.ecg(signal=ecgin, sampling_rate=256, show=False)
    return out


data0 = np.array(range(10))+.5
ts1 = (np.array(range(100))+.5)/10
pp.plot(data0, data0*2, label='original')
data1=int_cubic(data0*2, data0, ts1)
pp.plot(ts1, data1+.1, label='cubic')
data1=int_spline(data0*2, data0, ts1)
pp.plot(ts1, data1+.2, label='spline')
pp.legend()
pp.show()
data = pd.read_csv(sys.argv[1])
pulse = get_pulse(data['ecg'])
# pp.plot(pulse['heart_rate_ts'], pulse['heart_rate'])
pulse1 = int_cubic(pulse['heart_rate'], pulse['heart_rate_ts'], data['time'])
pulse2 = int_spline(pulse['heart_rate'], pulse['heart_rate_ts'], data['time'])
pp.plot(pulse['heart_rate_ts'], pulse['heart_rate'], label='original')
pp.plot(data['time'], pulse1, label='cubic')
pp.plot(data['time'], pulse2, label='spline')
pp.legend()
pp.show()
norm1 = lognorm2norm(pulse1)
norm2 = lognorm2norm(pulse2)
pp.hist(norm1, bins=20)
pp.hist(norm2, bins=20)
pp.show()
                 
