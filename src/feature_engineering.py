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

# Idea is to create a new dataset, breaking the time siries in a fixed frequency
# This would create more meaningful data

def respiration(dataset, frequency=256):

	b, a = signal.butter(8,0.05)
	# get the raw data and filter it
	respiration = signal.filtfilt(b, a, dataset['r'], padlen=150)
	# plt.plot(resp[3000:4024])
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
	
	return resp_rate



def ecg_(dataset, frequency=256):

	# filter the data
	b, a = signal.butter(8,0.01)
	y = signal.filtfilt(b, a, dataset['ecg'], padlen=150)

	# plt.plot(y[6000:14024])
	# plt.show()

	out = ecg.ecg(signal=dataset['ecg'], sampling_rate=frequency, show=False)

	# plt.plot(out['heart_rate_ts'], out['heart_rate'])
	# plt.ylabel('Heart Rate (BPM)')
	# plt.xlabel('Time [s]');


	heart_rate_ts = out['heart_rate_ts']
	heart_rate = out['heart_rate']
	heart_filtered = out['filtered']

	# What to Return ? ? ? ?
	print(heart_rate.shape)
	return heart_rate


def eeg_(dataset, frequency=256):
	
	# There're diffent ways to handle eeg
	fp1_f7 = dataset['eeg_fp1'] - dataset['eeg_f7']
	f7_t3 = dataset['eeg_f7'] - dataset['eeg_t3']
	t3_t5 = dataset['eeg_t3'] - dataset['eeg_t5']
	t5_o1 = dataset['eeg_t5'] - dataset['eeg_o1']
	fp1_f3 = dataset['eeg_fp1'] - dataset['eeg_f7']
	f3_c3 = dataset['eeg_f3'] - dataset['eeg_c3']
	c3_p3 = dataset['eeg_c3'] - dataset['eeg_p3']
	p3_o1 = dataset['eeg_p3'] - dataset['eeg_o1']

	fz_cz = dataset['eeg_fz'] - dataset['eeg_cz']
	cz_pz = dataset['eeg_cz'] - dataset['eeg_pz']
	pz_poz = dataset['eeg_pz'] - dataset['eeg_poz']

	fp2_f8 = dataset['eeg_fp2'] - dataset['eeg_f8']
	f8_t4 = dataset['eeg_f8'] - dataset['eeg_t4']
	t4_t6 = dataset['eeg_t4'] - dataset['eeg_t6']
	t6_o2 = dataset['eeg_t6'] - dataset['eeg_o2']
	fp2_f4 = dataset['eeg_fp2'] - dataset['eeg_f4']
	f4_c4 = dataset['eeg_f4'] - dataset['eeg_c4']
	c4_p4 = dataset['eeg_c4'] - dataset['eeg_p4']
	p4_o2 = dataset['eeg_p4'] - dataset['eeg_o2']


if __name__ == '__main__':

	dataset_path = "../data"
	engineered_data = pd.DataFrame()

	# for a specific data/series input
	if len(sys.argv) > 1:
		file = sys.argv[1]
		if os.path.exists(file):
			try:

				
				dataset = pd.read_csv(sys.argv[1])

				engineered_data['respiration_rate'] = respiration(dataset)
				ecg_(dataset)
				eeg_(dataset)
				#engineered_data['heart_rate'] = ecg_(dataset)

			except OSError:
				print("Could not open/read file:", file)
				sys.exit()
		else:
			print("File: ", file, " Does not exist")
			sys.exit()
	
		
	else:

		os.chdir(dataset_path)
		for file in glob.glob("train_*.csv"):
			try:
				data = pd.read_csv(file)
				# first, do some feature engineering in each


				# Then for each, combine as new features, instead of new rowns
			except OSError:
				print("Could not open/read file:", file)
				sys.exit()


	###### 
	# Save the new engineered_data to file, and use it as input for any model to be tried
