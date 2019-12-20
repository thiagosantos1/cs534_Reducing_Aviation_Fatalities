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
    file = "../data/engineered_train_full_kalman.csv"

    if os.path.exists(file):
        try:

            print("\n#### Loading Data ####")
            dataset_train = pd.read_csv(file)
            # plt.plot( dataset_train['gsr'])
            # plt.show()

            # data_norm = preprocessing.scale(dataset_train['gsr'][1000000:])
            # plt.plot(data_norm)
            # plt.show()

            train, validation = train_test_split(dataset_train, test_size=0.2, shuffle=True)
            y_train = train['target']
            X_train = train.drop(['target'], axis=1)

            y_validaiton= validation['target']
            X_validation = validation.drop(['target'], axis=1)


            #### Base Models
            logistic = run_logisticRegression(X_train, y_train, X_validation, y_validaiton)
            ridge = run_ridge(X_train, y_train, X_validation, y_validaiton)
            lasso = run_lasso(X_train, y_train, X_validation, y_validaiton)

            #### "Best" Models
            lgbm_model = run_lgb(X_train, y_train, X_validation, y_validaiton)
            print("\n##############")
            decision_tree = run_DecisionTree(X_train, y_train, X_validation, y_validaiton)
            print("\n##############")
            rand_forest = run_RandomForest(X_train, y_train, X_validation, y_validaiton)

        except OSError:
            print("Could not open/read file:", file)
            sys.exit()

    else:
        print("File: ", file, " Does not exist")
        sys.exit()