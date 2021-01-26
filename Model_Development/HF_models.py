#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HF_models.py

Created on Mon Jun 29 08:39:05 2020

@author: rtsearcy

Description: Functions that tests all model types on an input test case

"""

import pandas as pd
import os
import datetime
import numpy as np
from scipy import stats
from scipy.stats.mstats import gmean
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.stattools import durbin_watson
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pingouin as pg
import wq_modeling as wqm
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import auc

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 100)

# Important enviro vars (EV)
top_var = ['rad', 'tide','WVHT','APD','DPD','Wtemp_B','atemp','awind','owind'] 

def partition(df, f, interval, train_event, hf_test_event, rm_test_event):
    ## Partition training and test datasets
    # df - all data
    # f - FIB being modeled
    # interval - use all samples or select only those at the top of the hour
    # train_event - HF event(s) used for training
    # hf_test_event = HF event(s) used for testing
    # rm_test_event = RM seasons used for testing
    
    # Remove other FIB variables
    other_fib = [x for x in ['TC','FC','ENT'] if x != f]
    cols = df.columns
    for i in range(0, len(other_fib)):
            cols = [x for x in cols if other_fib[i] not in x] 
    df = df[cols]
    
    # Training subset
    train = df[df.event.isin(train_event)]
    
    # HF test subset (if applicable)
    hf_test = df[df.event.isin(list(hf_test_event))]
    
    # Adjust for interval
    if interval == 'hour_avg':
        train = train.groupby(['dayofyear','hour']).mean().reset_index()
        #hf_test = hf_test.groupby(['dayofyear','hour']).mean().reset_index()
    elif interval == 'top_of_hour':
        train = train[train.index.minute==0]
        #hf_test = hf_test[hf_test.index.minute==0]
        
    # Routine Monitoring (RM) test subset
    b = train.beach[0]
    rm_test = df[(df.beach == b) & (df.event.str.contains('RM'))]   
    rm_test = rm_test.loc[~rm_test.index.duplicated(keep='last')]   # drop duplicates
    rm_test.dropna(subset=top_var, inplace=True)    # drop rows of missing vars
    
    # Keep only the RM data in the summer season(s) following the final event in training
    train_year=train.index[-1].year
    train_end=train.index[-1].date()
    rm_test = rm_test[(rm_test.index.date > train_end) & 
                      (rm_test.index.year.isin(rm_test_event)) & 
                      (rm_test.index.month.isin([4,5,6,7,8,9,10]))]
    
    # Print what is in the subsets
    print('- Modeling Setup -\n')
    print('Beach: ' + b)
    print('FIB: ' + f)
    if interval == 'hourly':
        print('* Hour averages *')
    elif interval == 'top_of_hour':
        print('* Top of Hour *')
    print('Training: ' + str(train_event) + ' (N=' + str(len(train)) + ')')
    print('HF Test: ' + str(hf_test_event) + ' (N=' + str(len(hf_test)) + ')')
    print('RM Test: ' + datetime.date.strftime(rm_test.index[0].date(), format='%Y/%m/%d')
          + ' - ' + datetime.date.strftime(rm_test.index[-1].date(), format='%Y/%m/%d') + ' (N=' + str(len(rm_test)) + ')')
    print('   Wet Days in RM Dataset: ' + str(rm_test['wet3'].sum()))
    
    return train, hf_test, rm_test
    
def get_interaction(x, interact_var):
    # Create interaction variables
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_vars = pd.DataFrame(poly.fit_transform(x[interact_var]), 
                             columns=poly.get_feature_names(interact_var), index=x.index)
    poly_vars = poly_vars[[v for v in poly_vars.columns if v not in x.columns]]
    return poly_vars

def select_vars(train, f, EV, interaction=True, interact_var=[], var_select_method=None):
    ## Select independent variables using the training dataset
    # train - training dataframe
    # f - FIB modeled
    # EV - list of environmental variables to start with
    # interaction - include polynomial interaction terms
    # interact_var - short list of variables to create interaction terms with
    # var_select_method = random forest (rf), LassoCV (lasso), Recursive feature elemination (RFE)
    
    
    # Interaction terms
    if interaction:    
        poly_vars = get_interaction(train, interact_var)
        train = train.merge(poly_vars, left_index=True, right_index=True)
        to_select = EV + list(poly_vars.columns)
    else:
        to_select = EV
        
    # Select variables
    X_train = wqm.select_vars(train['log'+f], train[to_select].dropna(axis=1), 
                              method=var_select_method, no_model=[], corr_thresh=0, vif=5)
    
    final_vars = list(X_train.columns)
    
    return X_train, final_vars


def compute_AUROC(y, y_pred, f):
    # Calculate AUROC given the observed and predicted SSS exceedances
    tune_range = np.arange(0.7, 2.25, 0.005)
    sens_spec = np.array([wqm.pred_eval(y, (y_pred * j), thresh=np.log10(wqm.fib_thresh(f)), tune=True) for j in tune_range])
    tpr = sens_spec[:,0]
    fpr = 1 - sens_spec[:,1]
    auroc=auc(fpr,tpr)
    return auroc

        