import pandas as pd
import numpy as np
import sys
import subprocess
import glob, os
import matplotlib.pyplot as plt
from sklearn import preprocessing


#### MODELS #####
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from xgboost import XGBClassifier

##### METRICS
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
import time
from sklearn.metrics import roc_auc_score

##### Base Models #####

def run_logisticRegression(X_train, y_train, X_test, y_test):

    print("\n### Running Logistic Regression\n")
    model = LogisticRegression().fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print("Logistic Score: ", score)
    return model


def run_ridge(X_train, y_train, X_test, y_test, alpha_=0.9):

    print("\n### Running Ridge Regression\n")
    model = Ridge(alpha=alpha_)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print("Ridge Score: ", score)

    return model


def run_lasso(X_train, y_train, X_test, y_test, alpha_=0.2):

    print("\n### Running Lasso Regression\n")
    model = Lasso(alpha=alpha_)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print("Lasso Score: ", score)

    return model



##### Searching for Better Models #####

# Lightgbm
def run_lgb(X_train, y_train, X_test, y_test):

    print("\n### Running Light GBM Classifier\n")

    params1 = {
        "objective" : "multiclass",
        "num_class": 4,
        "metric" : "multi_error",
        "num_leaves" : 30,
        "min_child_weight" : 50,
        "learning_rate" : 0.1,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.7,
        "verbosity" : -1
    }

    params2 = {
        "objective" : "multiclass", 
        "metric" : "multi_error", 
        'num_class':4,
        "num_leaves" : 30, 
        "learning_rate" : 0.01, 
        "bagging_fraction" : 0.8,
        "feature_fraction" : 0.7,
        "num_threads" : 4,
        "colsample_bytree" : 0.5,
        'min_data_in_leaf':100, 
        'min_split_gain':0.00019
    }
    
    lg_train = lgb.Dataset(X_train, label=(y_train))
    lg_test = lgb.Dataset(X_test, label=(y_test))
    model = lgb.train(params1, lg_train, 1000, valid_sets=[lg_test], early_stopping_rounds=80, verbose_eval=100)

    return model


def run_DecisionTree(X_train, y_train, X_test, y_test, max_depth_=30):
    print("\n### Running Decision Tree Classifier\n")

    start = time.time()
    clf_1 = DecisionTreeClassifier(max_depth=max_depth_)
    clf_1 = clf_1.fit(X_train, y_train)
    end = time.time()
    y_pred = clf_1.predict(X_test)
    dec_tree_time = end - start

    dec_tree_score = clf_1.score(X_test, y_test)
    pr_score = precision_score(y_test, y_pred, average='weighted')
    rc_score = recall_score(y_test, y_pred, average='weighted')
    f1_score_ = f1_score(y_test, y_pred, average='weighted')
    # roc = roc_auc_score(y_test, y_pred)
    print('dec_tree_score ',dec_tree_score)
    print('precision_score', pr_score)
    print('recall_score', rc_score)
    print('f1_score', f1_score_)
    # print("roc_auc_score: ", roc)
    print('dec_tree_time (s)', dec_tree_time)
    
    cm = confusion_matrix(y_test, y_pred)
    print('\nconfusion_matrix\n',cm)
    # plt.matshow(cm)
    # plt.show()


def run_RandomForest(X_train, y_train, X_test, y_test, max_depth_=30, n_estimators_=50):

    print("\n### Running Random Forest Classifier\n")
    start = time.time()
    clf_2 = RandomForestClassifier(n_estimators=n_estimators_, max_depth=max_depth_)
    clf_2 = clf_2.fit(X_train, y_train)
    end = time.time()
    y_pred = clf_2.predict(X_test)
    RF_time = end - start

    RF_score = clf_2.score(X_test, y_test)
    pr_score = precision_score(y_test, y_pred, average='weighted')
    rc_score = recall_score(y_test, y_pred, average='weighted')
    # roc = roc_auc_score(y_test, y_pred)
    print('RF_score ',RF_score)
    print('precision_score', pr_score)
    print('recall_score', rc_score)
    # print("roc_auc_score: ", roc)
    print('RF_time (s)', RF_time)

    cm = confusion_matrix(y_test, y_pred)
    print('confusion_matrix\n',cm)
    # plt.matshow(cm)
    # plt.show()


def run_XGBOST(X_train, y_train, X_test, y_test, max_depth_=30):

    print("\n### Running XGBOST\n")
    start = time.time()
    Xgbost = XGBClassifier(max_depth=max_depth_)
    Xgbost = Xgbost.fit(X_train, y_train)
    end = time.time()
    y_pred = Xgbost.predict(X_test)
    RF_time = end - start

    RF_score = Xgbost.score(X_test, y_test)
    pr_score = precision_score(y_test, y_pred, average='weighted')
    rc_score = recall_score(y_test, y_pred, average='weighted')
    print('RF_score ',RF_score)
    print('precision_score', pr_score)
    print('recall_score', rc_score)
    print('RF_time (s)', RF_time)

    cm = confusion_matrix(y_test, y_pred)
    print('confusion_matrix\n',cm)
    # plt.matshow(cm)
    # plt.show()

