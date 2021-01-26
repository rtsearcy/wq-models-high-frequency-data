#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HF_predictive_models.py

Script to develop and test models individually

Created on Mon Jun 29 08:39:05 2020

@author: rtsearcy
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

def get_interaction(x, interact_var):
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_vars = pd.DataFrame(poly.fit_transform(x[interact_var]), 
                             columns=poly.get_feature_names(interact_var), index=x.index)
    poly_vars = poly_vars[[v for v in poly_vars.columns if v not in x.columns]]
    return poly_vars

def compute_AUROC(y, y_pred, f):
    tune_range = np.arange(0.7, 2.25, 0.005)
    sens_spec = np.array([wqm.pred_eval(y, (y_pred * j), thresh=np.log10(wqm.fib_thresh(f)), tune=True) for j in tune_range])
    tpr = sens_spec[:,0]
    fpr = 1 - sens_spec[:,1]
    auroc=auc(fpr,tpr)
    return auroc

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 100)

folder = '/Users/rtsearcy/Box/water_quality_modeling/thfs/EDA'
save_folder = '/Users/rtsearcy/Box/water_quality_modeling/thfs/EDA/summer2020/explanatory_models'
save_folder = '/Users/rtsearcy/Desktop'
### CHANGE THESE PARAMETERS ###
f = 'ENT'                             # FIB to model: ENT, FC
train_events = ['LP-16']              # Which events to include in the training set?
test_events = ['LP-18']               # Typically the following single event, if applicable
interval = None                # Sampling interval to analyze: None - include all samples, 'hour_avg','top_of_hour'
#agg = None        # How to aggregate the samples: 'None - take a single sample at the interval, 'mean','median'

vs_method = 'forest'                   # Variable Selection Method (None, 'lasso', 'forest')

interaction = True
interact_var = ['rad', 'solar_noon', 'tide','tide_gtm','WVHT','Wtemp_B','owind']

EV = ['rad', 'daytime','solar_noon','hour',
      'tide', 'tide_stage', 'tide_gtm', 'dtide_1', 'dtide_2',
      'WVHT','APD','DPD','Wtemp_B',
      'atemp','dtemp','awind','owind', #'wspd','wdir',
      'dayofyear','lograin3T', 'lograin7T', 'wet3', 'wet7', 
      'upwelling','days_since_full_moon', 'spring_tide']


top_var = ['rad', 'tide','WVHT','DPD','Wtemp_B','atemp','awind','owind'] # Important enviro vars (EV)



LP = pd.read_csv(os.path.join(folder, 'all_event_data_LP.csv'), parse_dates=['dt'], index_col=['dt'])
CB = pd.read_csv(os.path.join(folder, 'all_event_data_CB.csv'), parse_dates=['dt'], index_col=['dt'])
HSB = pd.read_csv(os.path.join(folder, 'all_event_data_HSB.csv'), parse_dates=['dt'], index_col=['dt'])

df = pd.concat([LP,CB,HSB])  # all data

# # Interaction terms hard code
# df['rad_tide'] = df['rad']*df['tide']
# df['WVHT_tide'] = df['WVHT']*df['tide']
# df['WVHT_Wtemp_B'] = df['WVHT']*df['Wtemp_B']
# df['Wtemp_B_tide'] = df['tide']*df['Wtemp_B']

# Daytime = between sunrise and sunset
df_log = pd.read_csv('/Users/rtsearcy/Box/water_quality_modeling/thfs/EDA/summer2020/logistics/logistics_all_events.csv', 
                     index_col=['event'])

for e in df_log.index.unique():
    sr = pd.to_datetime(df_log.loc[e]['sunrise1']).time()
    ss = pd.to_datetime(df_log.loc[e]['sunset1']).time()
    df.loc[df.event==e,'daytime'] = [1 if (x > sr) and (x < ss) else 0 for x in df[df.event==e].index.time] 

# Solar Noon
#Values: (1) = hours = 9-3, (0) = 6-9 or 3-7
sn = df[(df.index.time >= datetime.time(9,0)) & (df.index.time < datetime.time(15,0))]
df['solar_noon'] = [1 if x in sn.index else 0 for x in df.index]  # If sample was taken between 9a and 3p

hf = df[df.event.isin(['LP-13','LP-16','LP-18','CB-11','CB-12','HSB-02'])] # HF events

# Individual and Combined Events
lp13 = hf[hf.event=='LP-13']
lp16 = hf[hf.event=='LP-16']
lp18 = hf[hf.event=='LP-18']
lp13_16 = hf[hf.event.isin(['LP-13','LP-16'])]  # First two LP events
lp = hf[hf.event.isin(['LP-13','LP-16','LP-18'])]  # All LP events

cb11 = hf[hf.event=='CB-11']
cb12 = hf[hf.event=='CB-12']
cb = hf[hf.event.isin(['CB-11','CB-12'])]  # All CB events

hsb02 = hf[hf.event=='HSB-02']

#%% Select Dataset, Interval, and FIB to Model

# Partition training and test datasets
train = hf[hf.event.isin(train_events)]
b = train.beach.unique()[0] # beach

# HF test subset (if applicable)
hf_test = hf[hf.event.isin(test_events)]

# Routine Monitoring (RM) test subset
rm_test = df[(df.beach == b) & (df.event.str.contains('RM'))]   
rm_test = rm_test.loc[~rm_test.index.duplicated(keep='last')]   # drop duplicates
rm_test.dropna(subset=top_var, inplace=True)    # drop rows of missing vars
# Keep only the RM data in the following summer season of the final event in training
train_year=train.index[-1].year
train_end=train.index[-1].date()
rm_test = rm_test[(rm_test.index.date > train_end) & (rm_test.index.date <= datetime.date(train_year,10,31))]

# Adjust for interval and aggregation
if interval == 'hour_avg':
    train = train.groupby(['dayofyear','hour']).mean().reset_index()
    hf_test = hf_test.groupby(['dayofyear','hour']).mean().reset_index()
elif interval == 'top_of_hour':
    train = train[train.index.minute==0]
    hf_test = hf_test[hf_test.index.minute==0]


# Replace LOD with 1/2 LOD
train['log'+f][(train[f+'_bloq'] == 1)] = np.log10(5)
hf_test['log'+f][(hf_test[f+'_bloq'] == 1)] = np.log10(5)
rm_test['log'+f][(rm_test[f+'_bloq'] == 1)] = np.log10(5)

# Remove other FIB variables
other_fib = [x for x in ['TC','FC','ENT'] if x != f]
cols = train.columns
for i in range(0, len(other_fib)):
        cols = [x for x in cols if other_fib[i] not in x] 
train = train[cols]
hf_test = hf_test[cols]
rm_test = rm_test[cols]



# Print what is in the test sets
print('- - Modeling Setup - -\n')
print('Beach: ' + b)
print('FIB: ' + f)
if interval == 'hourly':
    print('* Hour averages *')
elif interval == 'top_of_hour':
    print('* Top of Hour *')
print('Training: ' + str(train_events) + ' (N=' + str(len(train)) + ')')
print('HF Test: ' + str(test_events) + ' (N=' + str(len(hf_test)) + ')')
print('RM Test: ' + datetime.date.strftime(rm_test.index[0].date(), format='%Y/%m/%d')
      + ' - ' + datetime.date.strftime(rm_test.index[-1].date(), format='%Y/%m/%d') + ' (N=' + str(len(rm_test)) + ')')
print('   Wet Days in RM Dataset: ' + str(rm_test['wet3'].sum()))

#%% Variable Selection
to_select = EV

# Interaction terms
if interaction:    
    poly_vars = get_interaction(train, interact_var)
    train = train.merge(poly_vars, left_index=True, right_index=True)
    to_select += list(poly_vars.columns)
    
    # # Adjust test sets
    # if len(hf_test)>0:
    #     hf_test = hf_test.merge(get_interaction(hf_test, interact_var), left_index=True, right_index=True)
    # rm_test = rm_test.merge(get_interaction(rm_test, interact_var), left_index=True, right_index=True)

# Select variables
X_train = wqm.select_vars(train['log'+f], train[to_select].dropna(axis=1), 
                          method=vs_method, no_model=[], corr_thresh=0, vif=5)

final_vars = list(X_train.columns)

# Adjust test datasets
# Adjust test sets
if len(hf_test)>0:
    hf_test = hf_test.merge(get_interaction(hf_test, interact_var), left_index=True, right_index=True)
    hf_test_IV = hf_test[final_vars]
rm_test = rm_test.merge(get_interaction(rm_test, interact_var), left_index=True, right_index=True)
rm_test_IV = rm_test[final_vars]

print('\n')

#%% Basic MLR (No CV, STATSMODEL)
print('\n- - Ordinary Least Squares - -')

# Fit 
mlr = sm.OLS(train['log'+f], sm.add_constant(X_train), hasconst=True).fit()
print(mlr.summary2())

# Tune (TBD)

# Eval (Train)
print('\nMetrics (Training)')
rmse = np.sqrt(((mlr.predict() - train['log'+f])**2).mean())
print('RMSE - ' + str(round(rmse,3)))
mape = abs((mlr.predict() - train['log'+f])/train['log'+f]).mean()
print('MAPE - ' + str(round(mape,3)))
mlr_train_perf = wqm.pred_eval(train['log'+f], mlr.predict(), thresh=np.log10(wqm.fib_thresh(f)))
print('AUROC - ' + str(round(compute_AUROC(train['log'+f], mlr.predict(), f),3)))
print(mlr_train_perf)

# Eval (HF Test)
if len(hf_test)>0:
    print('\nMetrics (HF Testing)')
    hf_test_pred = mlr.predict(sm.add_constant(hf_test_IV, has_constant='add'))
    rmse = np.sqrt(((hf_test_pred - hf_test['log'+f])**2).mean())
    print('RMSE - ' + str(round(rmse,3)))
    mape = abs((hf_test_pred - hf_test['log'+f])/hf_test['log'+f]).mean()
    print('MAPE - ' + str(round(mape,3)))
    print('AUROC - ' + str(round(compute_AUROC(hf_test['log'+f], hf_test_pred, f),3)))
    mlr_hft_perf = wqm.pred_eval(hf_test['log'+f], hf_test_pred, thresh=np.log10(wqm.fib_thresh(f)))
    print(mlr_hft_perf)

# Eval (RM Test)
print('\nMetrics (RM Testing)')
rm_test_pred = mlr.predict(sm.add_constant(rm_test_IV, has_constant='add'))
rmse = np.sqrt(((rm_test_pred - rm_test['log'+f])**2).mean())
print('RMSE - ' + str(round(rmse,3)))
mape = abs((rm_test_pred - rm_test['log'+f])/rm_test['log'+f]).mean()
print('MAPE - ' + str(round(mape,3)))
print('AUROC - ' + str(round(compute_AUROC(rm_test['log'+f], rm_test_pred, f),3)))
mlr_rmt_perf = wqm.pred_eval(rm_test['log'+f], rm_test_pred, thresh=np.log10(wqm.fib_thresh(f)))
print(mlr_rmt_perf)
print('Current Method (RM)')
print(wqm.pred_eval(rm_test['log'+f], rm_test['log'+f+'1'], thresh=np.log10(wqm.fib_thresh(f))))


#%% Generalized Least Squares
print('\n- - Generalized Least Squares - -')

# Fit
gls = sm.GLSAR(train['log'+f], sm.add_constant(X_train), 
               rho=2, missing='drop', hasconst=True).iterative_fit(maxiter=5)
print(gls.summary2())

# Tune

# Eval (Train)
print('Metrics (Training)')
rmse = np.sqrt(((gls.predict() - train['log'+f])**2).mean())
print('RMSE - ' + str(round(rmse,3)))
mape = abs((gls.predict() - train['log'+f])/train['log'+f]).mean()
print('MAPE - ' + str(round(mape,3)))
print('AUROC - ' + str(round(compute_AUROC(train['log'+f], gls.predict(), f),3)))
gls_train_perf = wqm.pred_eval(train['log'+f], gls.predict(), thresh=np.log10(wqm.fib_thresh(f)))
print(gls_train_perf)

# Eval (HF Test)
if len(hf_test)>0:
    print('\nMetrics (HF Testing)')
    hf_test_pred = gls.predict(sm.add_constant(hf_test_IV, has_constant='add'))
    rmse = np.sqrt(((hf_test_pred - hf_test['log'+f])**2).mean())
    print('RMSE - ' + str(round(rmse,3)))
    mape = abs((hf_test_pred - hf_test['log'+f])/hf_test['log'+f]).mean()
    print('MAPE - ' + str(round(mape,3)))
    print('AUROC - ' + str(round(compute_AUROC(hf_test['log'+f], hf_test_pred, f),3)))
    gls_hft_perf = wqm.pred_eval(hf_test['log'+f], hf_test_pred, thresh=np.log10(wqm.fib_thresh(f)))
    print(gls_hft_perf)

# Eval (RM Test)
print('\nMetrics (RM Testing)')
rm_test_pred = gls.predict(sm.add_constant(rm_test_IV, has_constant='add'))
rmse = np.sqrt(((rm_test_pred - rm_test['log'+f])**2).mean())
print('RMSE - ' + str(round(rmse,3)))
mape = abs((rm_test_pred - rm_test['log'+f])/rm_test['log'+f]).mean()
print('MAPE - ' + str(round(mape,3)))
print('AUROC - ' + str(round(compute_AUROC(rm_test['log'+f], rm_test_pred, f),3)))
gls_rmt_perf = wqm.pred_eval(rm_test['log'+f], rm_test_pred, thresh=np.log10(wqm.fib_thresh(f)))
print(gls_rmt_perf)
print('Current Method (RM)')
print(wqm.pred_eval(rm_test['log'+f], rm_test['log'+f+'1'], thresh=np.log10(wqm.fib_thresh(f))))


#%% Random Forests (Regression)
print('\n- - Random Forest - -')

# Fit
rf = RandomForestRegressor(n_estimators=1000, 
                           oob_score=True,
                           max_features=0.75,
                           random_state=0)
rf.fit(X_train,train['log'+f])

print('\nSummary of Model Fit')
print('\nR-sq: ' + str(round(rf.score(X_train,train['log'+f]),3)))
print('OOB RMSE: ' + str(round(rf.oob_score_**.5,3)))
print('Durbin-Watson: ' + str(round(durbin_watson(train['log'+f] - rf.predict(X_train)),3)))

# Eval (Train)
print('\nMetrics (Training)')
rmse = np.sqrt(((rf.predict(X_train) - train['log'+f])**2).mean())
print('RMSE - ' + str(round(rmse,3)))
mape = abs((rf.predict(X_train) - train['log'+f])/train['log'+f]).mean()
print('MAPE - ' + str(round(mape,3)))
print('AUROC - ' + str(round(compute_AUROC(train['log'+f], rf.predict(X_train), f),3)))
rf_train_perf = wqm.pred_eval(train['log'+f], rf.predict(X_train), thresh=np.log10(wqm.fib_thresh(f)))
print(rf_train_perf)

# Eval (HF Test)
if len(hf_test)>0:
    print('\nMetrics (HF Testing)')
    hf_test_pred = rf.predict(hf_test_IV)
    rmse = np.sqrt(((hf_test_pred - hf_test['log'+f])**2).mean())
    print('RMSE - ' + str(round(rmse,3)))
    mape = abs((hf_test_pred - hf_test['log'+f])/hf_test['log'+f]).mean()
    print('MAPE - ' + str(round(mape,3)))
    print('AUROC - ' + str(round(compute_AUROC(hf_test['log'+f], hf_test_pred, f),3)))
    rf_hft_perf = wqm.pred_eval(hf_test['log'+f], hf_test_pred, thresh=np.log10(wqm.fib_thresh(f)))
    print(rf_hft_perf)

# Eval (RM Test)
print('\nMetrics (RM Testing)')
rm_test_pred = rf.predict(rm_test_IV)
rmse = np.sqrt(((rm_test_pred - rm_test['log'+f])**2).mean())
print('RMSE - ' + str(round(rmse,3)))
mape = abs((rm_test_pred - rm_test['log'+f])/rm_test['log'+f]).mean()
print('MAPE - ' + str(round(mape,3)))
print('AUROC - ' + str(round(compute_AUROC(rm_test['log'+f], rm_test_pred, f),3)))
rf_rmt_perf = wqm.pred_eval(rm_test['log'+f], rm_test_pred, thresh=np.log10(wqm.fib_thresh(f)))
print(rf_rmt_perf)
print('Current Method (RM)')
print(wqm.pred_eval(rm_test['log'+f], rm_test['log'+f+'1'], thresh=np.log10(wqm.fib_thresh(f))))


#%% Artificial Neural Net
print('\n- - Artificial Neural Network - -')
n = 2*len(final_vars)  # number hidden layer nodes (see Park et al 2018)
ann = MLPRegressor(hidden_layer_sizes = (n,), 
                   activation='logistic',  #tanh, logistic
                   solver='sgd',   # adam, sgd, lbfgs
                   #alpha=0.00001,
                   #learning_rate_init=0.1,
                   max_iter=500,
                   random_state=0)

# Scale inputs
scaler = StandardScaler()
X_trainS = scaler.fit_transform(X_train)
if len(hf_test)>0:
    hf_testS = scaler.transform(hf_test_IV)
rm_testS = scaler.transform(rm_test_IV)   


ann.fit(X_trainS, train['log'+f])

print('\nSummary of Model Fit')
print('Number of Nodes in Hidden Layer: ' + str(n))
print('\nR-sq: ' + str(round(ann.score(X_trainS,train['log'+f]),3)))
print('Durbin-Watson: ' + str(round(durbin_watson(train['log'+f] - ann.predict(X_trainS)),3)))

# Eval (Train)
print('\nMetrics (Training)')
rmse = np.sqrt(((ann.predict(X_trainS) - train['log'+f])**2).mean())
print('RMSE - ' + str(round(rmse,3)))
mape = abs((ann.predict(X_trainS) - train['log'+f])/train['log'+f]).mean()
print('MAPE - ' + str(round(mape,3)))
print('AUROC - ' + str(round(compute_AUROC(train['log'+f], ann.predict(X_trainS), f),3)))
ann_train_perf = wqm.pred_eval(train['log'+f], ann.predict(X_trainS), thresh=np.log10(wqm.fib_thresh(f)))
print(ann_train_perf)

# Eval (HF Test)
if len(hf_test)>0:
    print('\nMetrics (HF Testing)')
    hf_test_pred = ann.predict(hf_testS)
    rmse = np.sqrt(((hf_test_pred - hf_test['log'+f])**2).mean())
    print('RMSE - ' + str(round(rmse,3)))
    mape = abs((hf_test_pred - hf_test['log'+f])/hf_test['log'+f]).mean()
    print('MAPE - ' + str(round(mape,3)))
    print('AUROC - ' + str(round(compute_AUROC(hf_test['log'+f], hf_test_pred, f),3)))
    ann_hft_perf = wqm.pred_eval(hf_test['log'+f], hf_test_pred, thresh=np.log10(wqm.fib_thresh(f)))
    print(ann_hft_perf)

# Eval (RM Test)
print('\nMetrics (RM Testing)')
rm_test_pred = ann.predict(rm_testS)
rmse = np.sqrt(((rm_test_pred - rm_test['log'+f])**2).mean())
print('RMSE - ' + str(round(rmse,3)))
mape = abs((rm_test_pred - rm_test['log'+f])/rm_test['log'+f]).mean()
print('MAPE - ' + str(round(mape,3)))
print('AUROC - ' + str(round(compute_AUROC(rm_test['log'+f], rm_test_pred, f),3)))
ann_rmt_perf = wqm.pred_eval(rm_test['log'+f], rm_test_pred, thresh=np.log10(wqm.fib_thresh(f)))
print(ann_rmt_perf)
print('Current Method (RM)')
print(wqm.pred_eval(rm_test['log'+f], rm_test['log'+f+'1'], thresh=np.log10(wqm.fib_thresh(f))))

#%% Other Algorithms 
#Basic MLR (w CV, SKLEARN)
# print('\n- - Ordinary Least Squares (w/ Cross-Validation) - -')

# # Fit (w/ CV)
# lm = LinearRegression()
# cv_type = KFold(n_splits=5, shuffle=True, random_state=0)
# CV = cross_validate(lm, X_train, train['log'+f], cv=cv_type,
#                     return_estimator=True, scoring="neg_root_mean_squared_error")
# best_est = int(np.where(CV['test_score'] == max(CV['test_score']))[0])  # maximizes the score
# mlr = CV['estimator'][best_est]

# print('\nSummary of Model Fit')
# print('\nVariable/Coefficient')
# count=0
# for c in X_train.columns:
#     print(c + ' - ' + str(round(mlr.coef_[count],5)))
#     count+=1
# print('Intercept - ' + str(round(mlr.intercept_,5)))
# print('\nR-sq: ' + str(round(mlr.score(X_train,train['log'+f]),3)))
# print('Durbin-Watson: ' + str(round(durbin_watson(train['log'+f] - mlr.predict(X_train)),3)))

# # Tune (TBD)

# # Eval (Train)
# print('\nMetrics (Training)')
# rmse = np.sqrt(((mlr.predict(X_train) - train['log'+f])**2).mean())
# print('RMSE - ' + str(round(rmse,3)))
# mape = abs((mlr.predict(X_train) - train['log'+f])/train['log'+f]).mean()
# print('MAPE - ' + str(round(mape,3)))
# mlr_train_perf = wqm.pred_eval(train['log'+f], mlr.predict(X_train), thresh=np.log10(wqm.fib_thresh(f)))
# print(mlr_train_perf)

# # Eval (HF Test)
# if len(hf_test)>0:
#     print('\nMetrics (HF Testing)')
#     hf_test_pred = mlr.predict(hf_test_IV)
#     rmse = np.sqrt(((hf_test_pred - hf_test['log'+f])**2).mean())
#     print('RMSE - ' + str(round(rmse,3)))
#     mape = abs((hf_test_pred - hf_test['log'+f])/hf_test['log'+f]).mean()
#     print('MAPE - ' + str(round(mape,3)))
#     mlr_hft_perf = wqm.pred_eval(hf_test['log'+f], hf_test_pred, thresh=np.log10(wqm.fib_thresh(f)))
#     print(mlr_hft_perf)

# # Eval (RM Test)
# print('\nMetrics (RM Testing)')
# rm_test_pred = mlr.predict(rm_test_IV)
# rmse = np.sqrt(((rm_test_pred - rm_test['log'+f])**2).mean())
# print('RMSE - ' + str(round(rmse,3)))
# mape = abs((rm_test_pred - rm_test['log'+f])/rm_test['log'+f]).mean()
# print('MAPE - ' + str(round(mape,3)))
# mlr_rmt_perf = wqm.pred_eval(rm_test['log'+f], rm_test_pred, thresh=np.log10(wqm.fib_thresh(f)))
# print(mlr_rmt_perf)
# print('Current Method (RM)')
# print(wqm.pred_eval(rm_test['log'+f], rm_test['log'+f+'1'], thresh=np.log10(wqm.fib_thresh(f))))


#% Basic BLR (with CV)
# print('\n- - Logistic Regression (w/ Cross-Validation) - -')

# # Fit (w/ CV)
# blr = LogisticRegressionCV(cv=KFold(shuffle=True),scoring='accuracy',random_state=0)

# blr.fit(X_train, train[f+'_exc'].astype(int))
# print('\nSummary of Model Fit')
# print('\nVariable/Coefficient')
# count=0
# for c in X_train.columns:
#     print(c + ' - ' + str(round(blr.coef_[0][count],5)))
#     count+=1
    
# # Tune (TBD)

# # Eval (Train)
# print('\nMetrics (Training)')
# blr_train_perf = wqm.pred_eval(train[f+'_exc'], blr.predict(X_train))
# print(mlr_train_perf)

# # Eval (HF Test)
# if len(hf_test)>0:
#     print('\nMetrics (HF Testing)')
#     hf_test_pred = blr.predict(hf_test_IV)
#     blr_hft_perf = wqm.pred_eval(hf_test[f+'_exc'], hf_test_pred)
#     print(mlr_hft_perf)

# # Eval (RM Test)
# print('\nMetrics (RM Testing)')
# rm_test_pred = mlr.predict(rm_test_IV)
# mlr_rmt_perf = wqm.pred_eval(rm_test[f+'_exc'], rm_test_pred)
# print(mlr_rmt_perf)
# print('Current Method (RM)')
# print(wqm.pred_eval(rm_test['log'+f], rm_test['log'+f+'1'], thresh=np.log10(wqm.fib_thresh(f))))


#% Multilevel modeling

        


        