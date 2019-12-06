import pandas as pd
import numpy as np
import sys
import subprocess
import glob, os
import matplotlib.pyplot as plt
from sklearn import preprocessing

#### MODELS #####
import lightgbm as lgb


def run_lgb(X_train, y_train, X_test, y_test):

    
    params = {"objective" : "multiclass",
              "num_class": 4,
              "metric" : "multi_error",
              "num_leaves" : 30,
              "min_child_weight" : 50,
              "learning_rate" : 0.1,
              "bagging_fraction" : 0.7,
              "feature_fraction" : 0.7,
              "verbosity" : -1
             }
    
    lg_train = lgb.Dataset(X_train, label=(y_train))
    lg_test = lgb.Dataset(X_test, label=(y_test))
    model = lgb.train(params, lg_train, 1000, valid_sets=[lg_test], early_stopping_rounds=50, verbose_eval=100)
    
    return model