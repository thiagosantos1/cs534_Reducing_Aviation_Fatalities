#!/usr/bin/python3

from biosppy import ecg
import matplotlib.pyplot as pp
import numpy as np
import pandas as pd
import sys
from tools import interpolate, lognorm2norm

def get_pulse(ecgin):
    out = ecg.ecg(signal=ecgin, sampling_rate=256, show=False)
    return out


data = pd.read_csv(sys.argv[1])
pulse = get_pulse(data['ecg'])
# pp.plot(pulse['heart_rate_ts'], pulse['heart_rate'])
pulse2 = interpolate(pulse['heart_rate'], pulse['heart_rate_ts'], data['time'])
norm = lognorm2norm(pulse2)
pp.plot(data['time'], norm)
pp.show()
pp.hist(norm, bins=20)
pp.show()
                 
