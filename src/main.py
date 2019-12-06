import pandas as pd
import numpy as np
import sys
import subprocess
import glob, os
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from models import *


if __name__ == '__main__':
    file = "../data/engineered_train_full.csv"

    if os.path.exists(file):
        try:
            dataset_train = pd.read_csv(file)
            train, validation = train_test_split(dataset_train, test_size=0.2, shuffle=True)
            y_train = train['target']
            X_train = train.drop(['target'], axis=1)

            y_validaiton= validation['target']
            X_validation = validation.drop(['target'], axis=1)

            lgbm_model = run_lgb(X_train, y_train, X_validation, y_validaiton)


        except OSError:
            print("Could not open/read file:", file)
            sys.exit()

    else:
        print("File: ", file, " Does not exist")
        sys.exit()