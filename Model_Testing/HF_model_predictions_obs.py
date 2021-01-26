#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HF_model_predictions_obs.py

Script to plot predictions and observations for specific models

Created on Mon Jun 29 08:39:05 2020

@author: rtsearcy
"""

import pandas as pd
import os
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
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.externals import joblib
import HF_models

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 100)

### INPUTS ###
case = 'LP3'
f = 'ENT'
model_types = ['RF']

model_color = {'MLR':'b',
               'GLS': 'g',
               'RF':'k',
               'ANN':'r'}

folder = '/Users/rtsearcy/Box/water_quality_modeling/thfs/EDA/summer2020/prediction'
case_folder = folder + '/test_cases/' + case + '/'

test_cases = pd.read_csv(os.path.join(folder, 'test_cases.csv'), index_col=['test_case'])
train_events = test_cases.loc[case]['train_event'].split(',')

thresh = wqm.fib_thresh(f)

### Load data
df_train = pd.read_excel(os.path.join(case_folder,'train_test_subsets_'+f+'_'+case+'.xlsx'), sheet_name='Train', 
                         index_col = 'dt', parse_dates=['dt'])
print('Train: ' + str(len(df_train)))

df_rm = pd.read_excel(os.path.join(case_folder,'train_test_subsets_'+f+'_'+case+'.xlsx'), sheet_name='RM Test', 
                         index_col = 'dt', parse_dates=['dt'])
df_rm = df_rm[df_rm.index.year == df_rm.index[0].year]  # Remove Year 2 RM Data
print('RM: ' + str(len(df_rm)))

if type(test_cases.loc[case]['hf_test_1']) == str:
    df_hf = pd.read_excel(os.path.join(case_folder,'train_test_subsets_'+f+'_'+case+'.xlsx'), sheet_name='HF Test', 
                          index_col = 'dt', parse_dates=['dt'])
    print('HF Test: ' + str(len(df_hf)))
for m in model_types:
### Load model
    model = joblib.load(os.path.join(case_folder, 'model_' + f + '_' + m + '_' + case + '.pkl'))
    EV = pd.read_csv(os.path.join(case_folder, 'vars_'+f+'_'+case+'.csv'),header=None)
    EV = list(EV[0].values)
    

### Make predictions (train, test)
    
    if m == 'ANN':  # Scale inputs
        scaler = StandardScaler()
        scaler.fit(df_train[EV])
        df_train[m+'_pred'] = model.predict(scaler.transform(df_train[EV]))
        df_rm[m+'_pred'] = model.predict(scaler.transform(df_rm[EV]))
        if type(test_cases.loc[case]['hf_test_1']) == str:
            df_hf[m+'_pred'] = model.predict(scaler.transform(df_hf[EV]))
    
    else: 
        df_train[m+'_pred'] = model.predict(df_train[EV])
        df_rm[m+'_pred'] = model.predict(df_rm[EV])
        if type(test_cases.loc[case]['hf_test_1']) == str:
            df_hf[m+'_pred'] = model.predict(df_hf[EV])

#%% Plots

### Time series
for m in model_types:
    
    # Train (Y1)
    plt.figure(figsize=(7,2))
    y = int('20'+train_events[0][-2:])
    d = df_train[df_train.index.year==y]
    plt.plot('log'+f, data=d, ls='',marker='.', c='k', label='Observation')
    #plt.plot(m + '_pred', data=d, marker='', c='k', alpha=0.6, label= m + ' Prediction')
    plt.plot(m + '_pred', data=d, marker='', c='k', alpha=0.6, label='Prediction')
    # for m in model_types:
    #     plt.plot(m + '_pred', data=d, marker='', c=model_color[m], alpha=0.6, label= m + ' Prediction')
    
    plt.axhline(np.log10(thresh),ls='--',color='k',alpha=.4, linewidth=.8)
    plt.ylabel(r'$log_{10}$'+f)
    #plt.title('Training (' + case + ' - ' + f + ' - ' + m + ')')
    plt.title('Training', loc='left')
    
    ax = plt.gca()
    ax.spines['left'].set_position(('outward', 5))
    ax.spines['bottom'].set_position(('outward', 5))
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(top=False)
    
    plt.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    #plt.axis(option='tight')
    
    # Train (Y2)
    if len(train_events) > 1:
        plt.figure(figsize=(7,2.5))
        y = int('20'+train_events[1][-2:])
        d = df_train[df_train.index.year==y]
        plt.plot('log'+f, data=d,ls='', marker='.', c='k', label='Observation')
        plt.plot(m+'_pred', data=d, marker='', c='k', alpha=0.6, label= m + ' Prediction')
        plt.axhline(np.log10(thresh),ls='--',color='k',alpha=.4,linewidth=.8)
        plt.ylabel(r'$log_{10}$'+f)
        plt.title('Training (' + case + ' - ' + f + ' - ' + m + ')')
        ax.tick_params(top=False)
        
        #plt.legend(loc='best', frameon=False)
        plt.tight_layout()
    
    # Train (Y3)
    if len(train_events) == 3:
        plt.figure(figsize=(7,2.5))
        y = int('20'+train_events[2][-2:])
        d = df_train[df_train.index.year==y]
        plt.plot('log'+f, data=d,ls='', marker='.', c='k', label='Observation')
        plt.plot(m+'_pred', data=d, marker='', c='k', alpha=0.6, label= m + ' Prediction')
        plt.axhline(np.log10(thresh),ls='--',color='k',alpha=.4,linewidth=.8)
        plt.ylabel(r'$log_{10}$'+f)
        plt.title('Training (' + case + ' - ' + f + ' - ' + m + ')')
        ax.tick_params(top=False)
        
        #plt.legend(loc='best', frameon=False)
        plt.tight_layout()
    
    # HF Test
    if type(test_cases.loc[case]['hf_test_1']) == str: # if future event
        plt.figure(figsize=(7,2))
        d = df_hf
        plt.plot('log'+f, data=d,ls='', marker='.', c='k', label='Observation')
        plt.plot(m+'_pred', data=d, marker='', c='k', alpha=0.6, label= m + ' Prediction')
        plt.axhline(np.log10(thresh),ls='--',color='k',alpha=.4,linewidth=.8)
        plt.ylabel(r'$log_{10}$'+f)
        #plt.title('HF Test (' + case + ' - ' + f + ' - ' + m + ')')
        plt.title('HF Validation', loc='left')
        
        ax = plt.gca()
        ax.spines['left'].set_position(('outward', 5))
        ax.spines['bottom'].set_position(('outward', 5))
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(top=False)
        
        #plt.legend(loc='best', frameon=False)
        plt.tight_layout()
    
    # RM Test
    plt.figure(figsize=(7,2))
    d = df_rm
    plt.plot('log'+f, data=d,ls='', marker='.', c='k', label='Observation')
    plt.plot(m + '_pred', data=d, marker='', c='k', alpha=0.6, label= m + ' Prediction')
    # for m in model_types:
    #     plt.plot(m + '_pred', data=d, marker='', c=model_color[m], alpha=0.6, label= m + ' Prediction')
    plt.axhline(np.log10(thresh),ls='--',color='k',alpha=.4,linewidth=.8)
    
    plt.ylabel(r'$log_{10}$'+f)
    #plt.title('RM Test (' + case + ' - ' + f + ' - ' + m + ')')
    plt.title('RM Validation', loc='left')
    
    ax = plt.gca()
    ax.spines['left'].set_position(('outward', 5))
    ax.spines['bottom'].set_position(('outward', 5))
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(top=False)
    
    #plt.legend(loc='best', frameon=False)
    plt.tight_layout()
    
    
### Scatter
for m in model_types:
    plt.figure(figsize=(4,4))
    plt.scatter('log'+f, m+'_pred',data=df_train, label='Training', c='k', s=12)
    if type(test_cases.loc[case]['hf_test_1']) == str:
        plt.scatter('log'+f, m+'_pred', data=df_hf, label='Future Event Test',c='k',alpha=0.6, marker='^', s=14)
    plt.scatter('log'+f, m+'_pred', data=df_rm, label='RM Test' ,c='grey', marker='+',s=16)
    
    plt.axhline(np.log10(thresh),ls='--',color='k',alpha=.4, linewidth=.75) # SSS
    plt.axvline(np.log10(thresh),ls='--',color='k',alpha=.4, linewidth=.75)
    
    # Square axis
    ul = max([plt.ylim()[1],plt.xlim()[1]])
    ll = min([plt.ylim()[0],plt.xlim()[0]])
    plt.xlim(ll,ul)
    plt.ylim(ll,ul)
    
    plt.plot(range(0,5), range(0,5), color='k',linewidth=.5)  # 1 to 1 line
    
    plt.legend(loc='best', frameon=False)
    plt.ylabel('Prediction')
    plt.xlabel('Observation')
    plt.title(case + ' - ' + f + ' - ' + m) 
    
    plt.tight_layout()
        