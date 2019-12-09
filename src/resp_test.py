#!/usr/bin/python3

from biosppy import ecg, resp
import matplotlib.pyplot as pp
import numpy as np
import pandas as pd
import sys
from tools import int_cubic, int_spline, lognorm2norm

def get_pulse(ecgin):
    out = ecg.ecg(signal=ecgin, sampling_rate=256, show=False)
    return out


def get_resp(rin):
    out = resp.resp(signal=rin, sampling_rate=256, show=False)
    return out


data = pd.read_csv(sys.argv[1])
pp.plot(data['time'],data['r'], label='Respiration')
pp.legend()
pp.show()
resp = get_resp(data['r'])
# pp.plot(pulse['heart_rate_ts'], resp['heart_rate'])
resp1 = int_cubic(resp['resp_rate'], resp['resp_rate_ts'], data['time'])
resp2 = int_spline(resp['resp_rate'], resp['resp_rate_ts'], data['time'])
pp.plot(resp['resp_rate_ts'], resp['resp_rate'], label='original')
pp.plot(data['time'], resp1, label='cubic')
pp.plot(data['time'], resp2, label='spline')
pp.legend()
pp.show()
norm1 = lognorm2norm(resp1)
norm2 = lognorm2norm(resp2)
pp.hist(norm1, bins=20)
pp.hist(norm2, bins=20)
pp.show()
                 
