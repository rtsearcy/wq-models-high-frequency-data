#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HF_test_all.py

Script to iterate through all HF test cases and develop and test models

Created on Mon Jun 29 08:39:05 2020

@author: rtsearcy
"""

import pandas as pd
import os
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
import warnings
import wq_modeling as wqm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.externals import joblib
import HF_models
import sys 


def metrics(obs, pred, f, q, m):
    # obs - log(observed), pred - prediction, f - FIB, q - subset, m - model
    rsq = round(r2_score(obs, pred), 3)
    dw = round(durbin_watson(obs - pred),3)  # Durbin-Watson 
    rmse = round(np.sqrt(((pred - obs)**2).mean()),3) # Root Mean Square Error
    mape = 100*round(abs((pred - obs)/obs).mean(),3) # Mean Absolute Percentage Error
    sens_spec = wqm.pred_eval(obs, 
                              pred, 
                              thresh=np.log10(wqm.fib_thresh(f))) # Sensitivity/Specificity
    auroc = round(HF_models.compute_AUROC(obs, pred,f),3) # Area Under the Receiver Operating Curve
    
    # Add to q performance for model m to perf dataframe
    mets = [[rsq, dw, rmse, mape, auroc, 
               sens_spec['Sensitivity'], sens_spec['Specificity'],
               sens_spec['Samples'], sens_spec['Exceedances']]]
    temp_perf = pd.DataFrame(data=mets, 
                        columns=['Rsq',
                                 'D-W',
                                 'RMSE',
                                 'MAPE',
                                 'AUROC',
                                 'sens',
                                 'spec',
                                 'N',
                                 'exc'], 
                        index=[[q],[m]])
    return temp_perf

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 100)

folder = '/Users/rtsearcy/Box/water_quality_modeling/thfs/EDA'
save_folder = '/Users/rtsearcy/Box/water_quality_modeling/thfs/EDA/summer2020/prediction'
save = True

EV = ['rad', 'daytime','solar_noon','hour',
      'tide', 'tide_stage', 'tide_gtm', 'dtide_1', 'dtide_2',
      'WVHT','APD','DPD','Wtemp_B',
      'atemp','dtemp','awind','owind', #'wspd','wdir',
      'dayofyear','lograin3T', 'lograin7T', 'wet3', 'wet7', 
      'upwelling','days_since_full_moon', 'spring_tide']

interact_var = ['rad', 'solar_noon', 'tide','tide_gtm','WVHT','Wtemp_B','owind']

model_types = ['MLR','GLS','RF','ANN']


# Metadata file with test cases (train event, test events, etc.)
test_cases = pd.read_csv(os.path.join(save_folder, 'test_cases.csv'), index_col=['test_case'])

toh_cases = ['LP8','LP9','LP10','LP11','LP12','LP13','LP14',
             'CB4','CB5','CB6','HSB2']

# Load data
LP = pd.read_csv(os.path.join(folder, 'all_event_data_LP.csv'), parse_dates=['dt'], index_col=['dt'])
CB = pd.read_csv(os.path.join(folder, 'all_event_data_CB.csv'), parse_dates=['dt'], index_col=['dt'])
HSB = pd.read_csv(os.path.join(folder, 'all_event_data_HSB.csv'), parse_dates=['dt'], index_col=['dt'])

df = pd.concat([LP,CB,HSB])  # all data

### Iterate through FIB and test cases
#for t in test_cases.index:
for t in toh_cases:
    for f in ['ENT','FC']:
        if ('HSB' in t) & (f=='FC'):
            print('\n- - No FC data at HSB - -')
            continue
        else:
            print('\n- - ' + t + '/' + f + ' - - ')

### Load test case metadata        
        beach = test_cases.loc[t]['beach']
        interval = test_cases.loc[t]['interval']
        train_event = test_cases.loc[t]['train_event'].split(',')
        interaction = test_cases.loc[t]['interaction']
        var_select_method = test_cases.loc[t]['var_select_method']
        hf_test_event = test_cases.loc[t][['hf_test_1','hf_test_2']].dropna().to_list()
        rm_test_event = test_cases.loc[t][['rm_test_1','rm_test_2']].dropna().to_list()
        
### Create test case folder (if doesn't exist)
        tc_folder = os.path.join(save_folder,'test_cases', t)
        os.makedirs(tc_folder, exist_ok=True)

### Partition data into training and test subsets, adjust for interval
        train, hf_test, rm_test = HF_models.partition(df,
                                                      f,
                                                      interval,
                                                      train_event, 
                                                      hf_test_event,
                                                      rm_test_event)

### Select variables w/ training data
        X_train, final_vars = HF_models.select_vars(train,
                                                    f,
                                                    EV,
                                                    interaction,
                                                    interact_var,
                                                    var_select_method)
        
### Adjust datasets with final variables
        if len(hf_test) > 0:
            hf_test = hf_test.merge(HF_models.get_interaction(hf_test, interact_var), 
                                    left_index=True, 
                                    right_index=True)
            hf_test_IV = hf_test[final_vars]
            
        rm_test = rm_test.merge(HF_models.get_interaction(rm_test, interact_var), 
                                left_index=True, 
                                right_index=True)
        rm_test_IV = rm_test[final_vars]

### Save subsets and variables
        if save:
            # Variables
            pd.Series(final_vars).to_csv(os.path.join(tc_folder,'vars_' + f + '_' + t + '.csv'),
                                                   index=False,
                                                   header=False)
            # Training set
            train_save = pd.merge(train['log'+f], X_train, left_index=True, right_index=True)
            
            # HF test sets
            if len(hf_test) > 0:
                hf_test_save = pd.merge(hf_test['log'+f], hf_test_IV, left_index=True, right_index=True)
            
            # RM test sets
            rm_test_save = pd.merge(rm_test['log'+f], rm_test_IV, left_index=True, right_index=True)
            
            with pd.ExcelWriter(os.path.join(tc_folder, 'train_test_subsets_' + f + '_' + t + '.xlsx')) as writer:  
                train_save.to_excel(writer, sheet_name='Train')
                if len(hf_test) > 0:
                    hf_test_save.to_excel(writer, sheet_name='HF Test')
                rm_test_save.to_excel(writer, sheet_name='RM Test')
            
            print('\n**Variables and Train/Test Sets Saved**')
                

### Iterate through model types (MLR, GLS, RF, ANN)
        print('\n- - Modeling - -')
        perf = pd.DataFrame()
        
        # Evaluation subsets (below)
        subsets = ['training','hf_test_1','hf_test_2','hf_test_agg',
                   'rm_test_1','rm_test_2','rm_test_agg']
    
        # Remove subsets if only one test case
        if len(hf_test_event) < 2:
            subsets.remove('hf_test_2')
            subsets.remove('hf_test_agg')
        if len(hf_test_event) < 1:
            subsets.remove('hf_test_1')
        if len(rm_test_event) < 2:
            subsets.remove('rm_test_2')
            subsets.remove('rm_test_agg')
        
        for m in model_types:
            print('\n- ' + m + ' -')
            if m == 'MLR':  # Multiple Linear Regression
                model = sm.OLS(train['log'+f], 
                               sm.add_constant(X_train), 
                               hasconst=True).fit()
                
            elif m == 'GLS':  # Generalized Least Squares
                model = sm.GLSAR(train['log'+f], 
                                 sm.add_constant(X_train), 
                                 rho=2, 
                                 missing='drop', 
                                 hasconst=True).iterative_fit(maxiter=5)
                
            elif m == 'RF':  # Random Forests
                print('\n- - Random Forest - -')
                model = RandomForestRegressor(n_estimators=1000, 
                                           oob_score=True,
                                           max_features=0.75,
                                           random_state=0)
                model.fit(X_train, train['log'+f])
                
            elif m == 'ANN':  # Artificial Neural Network (MLP)
                print('\n- - Artificial Neural Network - -')
                nodes = 2*len(final_vars)  # number hidden layer nodes (see Park et al 2018)
                model = MLPRegressor(hidden_layer_sizes = (nodes,), 
                                   activation='logistic',  #tanh, logistic
                                   solver='sgd',   # adam, sgd, lbfgs
                                   #alpha=0.00001,
                                   #learning_rate_init=0.1,
                                   max_iter=500,
                                   random_state=0)
                # Scale inputs
                scaler = StandardScaler()
                X_trainS = scaler.fit_transform(X_train)
                model.fit(X_trainS, train['log'+f])
            
#### Model summary
            if m in ['RF','ANN']:
                print('\nSummary of Model Fit')
                if m=='RF':
                    print('\nR-sq: ' + str(round(model.score(X_train, train['log'+f]),3)))
                    print('OOB RMSE: ' + str(round(model.oob_score_**.5,3)))
                    print('Durbin-Watson: ' + str(round(durbin_watson(train['log'+f] - model.predict(X_train)),3)))
                else:
                    print('\nR-sq: ' + str(round(model.score(X_trainS, train['log'+f]),3)))
                    print('Durbin-Watson: ' + str(round(durbin_watson(train['log'+f] - model.predict(X_trainS)),3)))
                
            elif m in ['MLR','GLS']:
                print(model.summary2())
            
            # Save models
            if save:  
                # Save model
                model_file = 'model_' + f + '_' + m + '_' + t + '.pkl'
                if m in ['RF','ANN']:
                    joblib.dump(model, os.path.join(tc_folder, model_file))
                    # use joblib.load to load this file in the model runs script 
                elif m in ['MLR','GLS']:
                    model.save(os.path.join(tc_folder, model_file))

### Metrics (Train/Test)                
            for q in subsets:
                
                if q == 'training':
                    obs = train['log'+f]
                    iv = X_train                 
                elif q in ['hf_test_1', 'hf_test_2']:
                    obs = hf_test[hf_test.event==test_cases.loc[t][q]]['log'+f].dropna()
                    iv = hf_test_IV
                elif q == 'hf_test_agg':
                    obs = hf_test['log'+f].dropna()
                    iv = hf_test_IV
                elif q in ['rm_test_1','rm_test_2']:
                    obs = rm_test[rm_test.year==test_cases.loc[t][q]]['log'+f].dropna()
                    #cm_pred = rm_test[rm_test.year==test_cases.loc[t][q]]['log'+f+'1'] # Current Method
                    cm_pred = rm_test.loc[obs.index]['log'+f+'1']
                    iv = rm_test_IV
                elif q == 'rm_test_agg':
                    obs = rm_test['log'+f].dropna()
                    cm_pred = rm_test.loc[obs.index]['log'+f+'1']  # Current Method
                    iv = rm_test_IV

                if m == 'RF':
                    pred = model.predict(iv.loc[obs.index])
                elif m == 'ANN':
                    # Scale inputs
                    pred = model.predict(scaler.transform(iv.loc[obs.index]))
                    
                elif m in ['MLR','GLS']:
                    pred = model.predict(sm.add_constant(iv.loc[obs.index]))
                
                # Metrics
                temp = metrics(obs, pred, f, q, m)
                perf = perf.append(temp)
                
                # Current/Persistence Method
                if (m==model_types[-1]) & ('rm' in q):
                    cm = metrics(obs, cm_pred, f, q, 'CM')
                    perf = perf.append(cm)
        
        # Print metrics
        perf.index.rename(['subset','model'], inplace=True)
        perf_save = pd.DataFrame()
        for q in subsets:
            q_temp = perf.loc[q]
            if q != 'training':
                q_temp['Rsq'] = np.nan
                q_temp['D-W'] = np.nan
            q_temp['subset']=q
            q_temp = q_temp.reset_index().set_index(['subset','model'])
            print('\n')
            print(q_temp)
            perf_save = perf_save.append(q_temp)
        
        if save:         
        # Save Performance
            perf_file = 'performance_' + f + '_' + t + '.csv'
            perf_save.to_csv(os.path.join(tc_folder, perf_file))
        
