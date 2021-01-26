#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HF_analyze_all_models.py

Script to aggregate modeling results to summarize

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

def bootstrap(x, alpha=0.05, num_straps = 1000, stat=np.median):
    # Calculates standard error on the statistic using the bootstrap method
    # Edited from http://www.jtrive.com/the-empirical-bootstrap-for-confidence-intervals-in-python.html
    x = x.dropna()
    simulations = list()
    sample_size = len(x)
    xbar_init = stat(x)
    for i in range(num_straps):
        itersample = np.random.choice(x, size=sample_size, replace=True)
        simulations.append(stat(itersample))
    simulations.sort()
    xbar_alpha = simulations[int(np.floor(num_straps*(1 - alpha/2)))]
    se = xbar_alpha - xbar_init
    return se

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 100)

folder = '/Users/rtsearcy/Box/water_quality_modeling/thfs/EDA'
save_folder = '/Users/rtsearcy/Box/water_quality_modeling/thfs/EDA/summer2020/prediction/'
case_folder = save_folder + '/test_cases'
save = True

# Categorical environmental variables (EVs)
EV_cat = {
    'sun_time': ['rad','daytime','solar_noon','hour'],
    'tide': ['tide','tide_stage','tide_gtm','dtide_1','dtide_2'],
    'wave': ['WVHT','APD','DPD','Wtemp_B'],
    'met': ['atemp','dtemp','awind','owind','wspd','wdir'],
    'low_freq': ['dayofyear','lograin3T', 'lograin7T', 'wet3', 'wet7', 
      'upwelling','days_since_full_moon', 'spring_tide']
    }

EV = [i for l in list(EV_cat.values()) for i in l]

model_types = ['GLS','RF','ANN']

# Metadata file with test cases (train event, test events, etc.)
test_cases = pd.read_csv(os.path.join(save_folder, 'test_cases.csv'), index_col=['test_case'])
test_cases['num_train'] = [len(x.split(',')) for x in test_cases['train_event']]

# subhourly Tests
raw_cases = list(test_cases[test_cases.interval != 'top_of_hour'].index)

# Hourly Tests
toh_cases = list(test_cases[test_cases.interval == 'top_of_hour'].index)

# Single event training
single_cases = list(test_cases[(test_cases.interval != 'top_of_hour')&(test_cases.num_train==1)].index)

# Multi/event training
multi_cases = list(test_cases[(test_cases.interval != 'top_of_hour')&(test_cases.num_train > 1)].index)


#%% Which variables showed up in the models?
# Desired results: How many of the models included XX variable type 
# (not the number of each type in the model)
idx = test_cases.index  # All test cases
included_vars = pd.DataFrame(index=idx, 
                             columns=[7*['ENT']+7*['FC'],
                                      2*(['N','interact']+list(EV_cat.keys()))])
var_count = pd.DataFrame(0,index=EV, columns=['ENT','FC'])

for t in idx:
    for f in ['ENT','FC']:
        if ('HSB' in t) & (f=='FC'):
            continue
        ivs = list(pd.read_csv(os.path.join(case_folder,t,'vars_'+f+'_'+t+'.csv'), header=None)[0])
        included_vars.loc[t,f].loc['N'] = len(ivs)  # Number of Variables
        for e in list(EV_cat.keys()):  # How many of each type of EV
            c = 0  # np.nan
            for i in ivs:
                i = i.split(' ')
                for j in i:
                    if j in EV_cat[e]:
                        c += 1      
            included_vars.loc[t,f].loc[e] = c
        # Interact
        included_vars.loc[t,f].loc['interact'] = sum([len(i.split(' ')) > 1 for i in ivs])
        
        # Count of models with each var
        iv_split = [i for l in [i.split(' ') for i in ivs] for i in l]
        for i in EV:
            if i in iv_split:
                var_count[f].loc[i] += 1
            
print('\n- - Included Variables - - \n')
for f in ['ENT','FC']:
    print(f)
    print(included_vars[f])
    print('\nNumber of Cases: ' + str(included_vars[f]['N'].count()))
    print('# Vars (mean/median): ' + str(round(included_vars[f]['N'].mean(),1)) + 
          '/' + str(included_vars[f]['N'].median()))
    se_var = bootstrap(included_vars[f]['N']) # Bootstrap error on the median
    se_var_mean = bootstrap(included_vars[f]['N'], stat=np.mean) # Bootstrap error on the mean
    print('  SE on Mean/Median: '+ str(se_var_mean) + '/'+ str(se_var))
    print('\n# Cases with Interact: ' + str((included_vars[f]['interact']>0).sum()))
    print('Interact/Case (mean/median): ' + str(round(included_vars[f]['interact'].mean(),1)) + 
          '/' + str(included_vars[f]['interact'].median()))
    print('\nVariable Types: ')
    print(included_vars[f].describe()[EV_cat.keys()].loc['count'])
    print('\n')
    
print(var_count)

# More variables in raw or top of hour cases?
included_vars[f]['N'].loc[toh_cases].median()
# Result: Slightly more variables in ENT TOH models. No sig difference in FC models

if save:
    save_file = os.path.join(save_folder, 'included_vars.xlsx')
    with pd.ExcelWriter(save_file) as writer:  
        included_vars.to_excel(writer, sheet_name='Included Variables')
        var_count.to_excel(writer, sheet_name='Models with Variables')


#%% Training: Rsq/RMSE statistics by model type
# Desired results: What is the distribution of fit metrics for each model type 
# Boxplots by model type, Table of R2 and RMSE for training subsets

idx = test_cases.index  # All test cases
lm = len(model_types)  # length of model type list
fit_metrics = pd.DataFrame(index=idx, 
                             columns=[2*lm*['ENT']+2*lm*['FC'],
                                      2*(lm*['Rsq'] + lm*['RMSE']),
                                      4*model_types])

for t in idx:
    for f in ['ENT','FC']:
        if ('HSB' in t) & (f=='FC'):
            continue
        fit = pd.read_csv(os.path.join(case_folder,t,'performance_'+f+'_'+t+'.csv'),
                          index_col=['subset']).loc['training'].reset_index().set_index('model')
        for m in model_types:    
            fit_metrics.loc[t,f].loc['Rsq',m] = fit.loc[m]['Rsq']
            fit_metrics.loc[t,f].loc['RMSE',m] = fit.loc[m]['RMSE']

print(fit_metrics)

ENT_rsq = fit_metrics['ENT']['Rsq']
FC_rsq = fit_metrics['FC']['Rsq']
ENT_rmse = fit_metrics['ENT']['RMSE']
FC_rmse = fit_metrics['FC']['RMSE']

### Grand mean
print('Grand Median:')
print('ENT (Rsq/RMSE)')
print(str(round(ENT_rsq.stack().median(),3)) + ' (' + 
      str(round(bootstrap(ENT_rsq.stack(), stat=np.median),3)) + ')')
print(str(round(ENT_rmse.stack().median(),3)) + ' (' + 
      str(round(bootstrap(ENT_rmse.stack(), stat=np.median),3)) + ')')
print('FC (Rsq/RMSE)')
print(str(round(FC_rsq.stack().median(),3)) + ' (' + 
      str(round(bootstrap(FC_rsq.stack(), stat=np.median),3)) + ')')
print(str(round(FC_rmse.stack().median(),3)) + ' (' + 
      str(round(bootstrap(FC_rmse.stack(), stat=np.median),3)) + ')')

### Average fit by case (across all model types -> For use for later)
fit_case = pd.concat([
    ENT_rsq.median(axis=1),
    ENT_rmse.median(axis=1),
    FC_rsq.median(axis=1),
    FC_rmse.median(axis=1)
    ], axis=1)
fit_case.columns = ['ENT_Rsq','ENT_RMSE','FC_Rsq','FC_RMSE']

### Average fit by model type (which model type fit data best?)
# Mean, boxplots, inference
fit_models = pd.concat([
    ENT_rsq.median(axis=0),
    pd.Series([round(bootstrap(ENT_rsq[x]),3) for x in ENT_rsq.columns], index=model_types),
    ENT_rmse.median(axis=0),
    pd.Series([round(bootstrap(ENT_rmse[x]),3) for x in ENT_rmse.columns], index=model_types),
    FC_rsq.median(axis=0),
    pd.Series([round(bootstrap(FC_rsq[x]),3) for x in FC_rsq.columns], index=model_types),
    FC_rmse.median(axis=0),
    pd.Series([round(bootstrap(FC_rmse[x]),3) for x in FC_rmse.columns], index=model_types)
    ], axis=1)
fit_models.columns = ['ENT_Rsq','SE', 'ENT_RMSE','SE','FC_Rsq','SE','FC_RMSE','SE']

print('\nMedian Fit by Model Type')
print(fit_models)

# Inference
# Are RF significantly better fitting than other model types? YES! to p<0.001
stats.mannwhitneyu(ENT_rsq['RF'], ENT_rsq['ANN'])
#stats.mannwhitneyu(ENT_rsq['RF'], ENT_rsq['MLR'])
stats.mannwhitneyu(ENT_rsq['RF'], ENT_rsq['GLS'])

stats.mannwhitneyu(ENT_rmse['RF'], ENT_rmse['ANN'])
#stats.mannwhitneyu(ENT_rmse['RF'], ENT_rmse['MLR'])
stats.mannwhitneyu(ENT_rmse['RF'], ENT_rmse['GLS'])

stats.mannwhitneyu(FC_rsq['RF'], FC_rsq['ANN'])
#stats.mannwhitneyu(FC_rsq['RF'], FC_rsq['MLR'])
stats.mannwhitneyu(FC_rsq['RF'], FC_rsq['GLS'])

stats.mannwhitneyu(FC_rmse['RF'], FC_rmse['ANN'])
#stats.mannwhitneyu(FC_rmse['RF'], FC_rmse['MLR'])
stats.mannwhitneyu(FC_rmse['RF'], FC_rmse['GLS'])

# Are ANN significantly worse fitting than other model types? 
# Result: 
stats.mannwhitneyu(ENT_rsq['RF'], ENT_rsq['ANN'])
#stats.mannwhitneyu(ENT_rsq['ANN'], ENT_rsq['MLR'])
stats.mannwhitneyu(ENT_rsq['ANN'], ENT_rsq['GLS'])

stats.mannwhitneyu(ENT_rmse['RF'], ENT_rmse['ANN'])
#stats.mannwhitneyu(ENT_rmse['ANN'], ENT_rmse['MLR'])
stats.mannwhitneyu(ENT_rmse['ANN'], ENT_rmse['GLS'])

stats.mannwhitneyu(FC_rsq['RF'], FC_rsq['ANN'])
#stats.mannwhitneyu(FC_rsq['ANN'], FC_rsq['MLR'])
stats.mannwhitneyu(FC_rsq['ANN'], FC_rsq['GLS'])

stats.mannwhitneyu(FC_rmse['RF'], FC_rmse['ANN'])
#stats.mannwhitneyu(FC_rmse['ANN'], FC_rmse['MLR'])
stats.mannwhitneyu(FC_rmse['ANN'], FC_rmse['GLS'])

# Both FIB
r2 = ENT_rsq.append(FC_rsq)
rmse = ENT_rmse.append(FC_rmse)
[bootstrap(r2[x]) for x in r2.columns] 
[bootstrap(rmse[x]) for x in rmse.columns]

# ### Compare Single vs. Double Event fits (which fit better?)
# # Average of model averages for each group, boxplots, inference

# # Cases with a paired train/test event
# ENT_rsq_single = ENT_rsq.loc[single_cases]
# ENT_rmse_single = ENT_rmse.loc[single_cases]
# ENT_rsq_multi = ENT_rsq.loc[multi_cases]
# ENT_rmse_multi = ENT_rmse.loc[multi_cases]

# FC_rsq_single = FC_rsq.loc[single_cases]
# FC_rmse_single = FC_rmse.loc[single_cases]
# FC_rsq_multi = FC_rsq.loc[multi_cases]
# FC_rmse_multi = FC_rmse.loc[multi_cases]

# # Inference
# # Result: Not a significant different in fit metrics (check prediction)
# stats.mannwhitneyu(ENT_rmse_single.stack(), ENT_rmse_multi.stack())
# stats.mannwhitneyu(ENT_rsq_single.stack(), ENT_rsq_multi.stack())
# stats.mannwhitneyu(FC_rmse_single.stack(), FC_rmse_multi.stack())
# stats.mannwhitneyu(FC_rsq_single.stack(), FC_rsq_multi.stack())

# # Boxplots
# def set_box_color(bp, color):
#     plt.setp(bp['boxes'], color=color)
#     plt.setp(bp['whiskers'], color=color)
#     plt.setp(bp['caps'], color=color)
#     plt.setp(bp['medians'], color=color)

# pal = ['#969696','#525252']
# pal = sns.color_palette(pal)
# params = {
#    'axes.labelsize': 11,
#    'font.size': 11,
#    'legend.fontsize': 10,
#    'xtick.labelsize': 11,
#    'ytick.labelsize': 9,
#    'font.family'  : 'sans-serif',
#    'font.sans-serif':'Helvetica',
#    'axes.axisbelow': True
#    }
# plt.rcParams.update(params)

# plt.figure(figsize=(8,8))

# plt.subplot(2,2,1) # Rsq ENT
# bpl = plt.boxplot(ENT_rsq_single.T, positions=np.array(range(len(ENT_rsq_single.T)))*2.0-0.4, sym='', widths=0.6)
# bpr = plt.boxplot(ENT_rsq_multi.T, positions=np.array(range(len(ENT_rsq_multi.T)))*2.0+0.4, sym='', widths=0.6)
# set_box_color(bpl, pal[1]) # colors are from http://colorbrewer2.org/
# set_box_color(bpr, pal[0])
# plt.text(-1.8, 0.92*plt.ylim()[1], r'$R^2$')
# plt.ylabel('ENT')

# plt.xticks(range(0, len(model_types) * 2, 2), model_types)
# plt.xlim(-2, len(model_types)*2)

# plt.subplot(2,2,2) # RMSE ENT
# bpl = plt.boxplot(ENT_rmse_single.T, positions=np.array(range(len(ENT_rmse_single.T)))*2.0-0.4, sym='', widths=0.6)
# bpr = plt.boxplot(ENT_rmse_multi.T, positions=np.array(range(len(ENT_rmse_multi.T)))*2.0+0.4, sym='', widths=0.6)
# set_box_color(bpl, pal[1]) # colors are from http://colorbrewer2.org/
# set_box_color(bpr, pal[0])
# plt.text(-1.85, 0.95*plt.ylim()[1], r'RMSE')

# plt.xticks(range(0, len(model_types) * 2, 2), model_types)
# plt.xlim(-2, len(model_types)*2)

# plt.subplot(2,2,3) # Rsq FC
# bpl = plt.boxplot(FC_rsq_single.T, positions=np.array(range(len(FC_rsq_single.T)))*2.0-0.4, sym='', widths=0.6)
# bpr = plt.boxplot(FC_rsq_multi.T, positions=np.array(range(len(FC_rsq_multi.T)))*2.0+0.4, sym='', widths=0.6)
# set_box_color(bpl, pal[1]) # colors are from http://colorbrewer2.org/
# set_box_color(bpr, pal[0])
# plt.text(-1.8, 0.92*plt.ylim()[1], r'$R^2$')
# plt.ylabel('FC')

# plt.xticks(range(0, len(model_types) * 2, 2), model_types)
# plt.xlim(-2, len(model_types)*2)

# # draw temporary lines and use them to create a legend
# plt.plot([], c=pal[1], label='Single Event')
# plt.plot([], c=pal[0], label='Multiple Events')
# plt.legend(loc='lower left', frameon=False)

# plt.subplot(2,2,4) # RMSE FC
# bpl = plt.boxplot(FC_rmse_single.T, positions=np.array(range(len(FC_rmse_single.T)))*2.0-0.4, sym='', widths=0.6)
# bpr = plt.boxplot(FC_rmse_multi.T, positions=np.array(range(len(FC_rmse_multi.T)))*2.0+0.4, sym='', widths=0.6)
# set_box_color(bpl, pal[1]) # colors are from http://colorbrewer2.org/
# set_box_color(bpr, pal[0])
# plt.text(-1.85, 0.95*plt.ylim()[1], r'RMSE')

# plt.xticks(range(0, len(model_types) * 2, 2), model_types)
# plt.xlim(-2, len(model_types)*2)

# plt.tight_layout()


### Compare raw vs. top of hour fits (which fit better?)
# Average of model averages for each group, boxplots, inference
ENT_rsq_raw = ENT_rsq.loc[raw_cases]
ENT_rmse_raw = ENT_rmse.loc[raw_cases]
ENT_rsq_toh = ENT_rsq.loc[toh_cases]
ENT_rmse_toh = ENT_rmse.loc[toh_cases]

FC_rsq_raw = FC_rsq.loc[raw_cases]
FC_rmse_raw = FC_rmse.loc[raw_cases]
FC_rsq_toh = FC_rsq.loc[toh_cases]
FC_rmse_toh = FC_rmse.loc[toh_cases]

# Inference

#By model type:
for m in model_types:
    print(m)
    print('ENT')
    print('RMSE/R2')
    print(stats.mannwhitneyu(ENT_rmse_raw[m], ENT_rmse_toh[m]))
    print(stats.mannwhitneyu(ENT_rsq_raw[m], ENT_rsq_toh[m]))
    print('FC')
    print('RMSE/R2')
    print(stats.mannwhitneyu(FC_rmse_raw[m], FC_rmse_toh[m]))
    print(stats.mannwhitneyu(FC_rsq_raw[m], FC_rsq_toh[m]))

#Aggregate all model types
# Result: Not a significant different in median (check prediction)
stats.mannwhitneyu(ENT_rmse_raw.stack(), ENT_rmse_toh.stack())
stats.mannwhitneyu(ENT_rsq_raw.stack(), ENT_rsq_toh.stack())
stats.mannwhitneyu(FC_rmse_raw.stack(), FC_rmse_toh.stack())
stats.mannwhitneyu(FC_rsq_raw.stack(), FC_rsq_toh.stack())

# All FIB
stats.mannwhitneyu(FC_rsq_raw.append(ENT_rsq_raw).stack(), FC_rsq_toh.append(ENT_rsq_toh).stack())
stats.mannwhitneyu(FC_rmse_raw.append(ENT_rmse_raw).stack(), FC_rmse_toh.append(ENT_rmse_toh).stack())


# # Boxplots
# def set_box_color(bp, color):
#     plt.setp(bp['boxes'], color=color)
#     plt.setp(bp['whiskers'], color=color)
#     plt.setp(bp['caps'], color=color)
#     plt.setp(bp['medians'], color=color)

# pal = ['#969696','#525252']
# pal = sns.color_palette(pal)
# params = {
#    'axes.labelsize': 11,
#    'font.size': 11,
#    'legend.fontsize': 10,
#    'xtick.labelsize': 11,
#    'ytick.labelsize': 9,
#    'font.family'  : 'sans-serif',
#    'font.sans-serif':'Helvetica',
#    'axes.axisbelow': True
#    }
# plt.rcParams.update(params)

# plt.figure(figsize=(8,8))

# plt.subplot(2,2,1) # Rsq ENT
# bpl = plt.boxplot(ENT_rsq_raw.T, positions=np.array(range(len(ENT_rsq_raw.T)))*2.0-0.4, sym='', widths=0.6)
# bpr = plt.boxplot(ENT_rsq_toh.T, positions=np.array(range(len(ENT_rsq_toh.T)))*2.0+0.4, sym='', widths=0.6)
# set_box_color(bpl, pal[1]) # colors are from http://colorbrewer2.org/
# set_box_color(bpr, pal[0])
# plt.text(-1.8, 0.92*plt.ylim()[1], r'$R^2$')
# plt.ylabel('ENT')

# plt.xticks(range(0, len(model_types) * 2, 2), model_types)
# plt.xlim(-2, len(model_types)*2)

# plt.subplot(2,2,2) # RMSE ENT
# bpl = plt.boxplot(ENT_rmse_raw.T, positions=np.array(range(len(ENT_rmse_raw.T)))*2.0-0.4, sym='', widths=0.6)
# bpr = plt.boxplot(ENT_rmse_toh.T, positions=np.array(range(len(ENT_rmse_toh.T)))*2.0+0.4, sym='', widths=0.6)
# set_box_color(bpl, pal[1]) # colors are from http://colorbrewer2.org/
# set_box_color(bpr, pal[0])
# plt.text(-1.85, 0.95*plt.ylim()[1], r'RMSE')

# plt.xticks(range(0, len(model_types) * 2, 2), model_types)
# plt.xlim(-2, len(model_types)*2)

# plt.subplot(2,2,3) # Rsq FC
# bpl = plt.boxplot(FC_rsq_raw.T, positions=np.array(range(len(FC_rsq_raw.T)))*2.0-0.4, sym='', widths=0.6)
# bpr = plt.boxplot(FC_rsq_toh.T, positions=np.array(range(len(FC_rsq_toh.T)))*2.0+0.4, sym='', widths=0.6)
# set_box_color(bpl, pal[1]) # colors are from http://colorbrewer2.org/
# set_box_color(bpr, pal[0])
# plt.text(-1.8, 0.92*plt.ylim()[1], r'$R^2$')
# plt.ylabel('FC')

# plt.xticks(range(0, len(model_types) * 2, 2), model_types)
# plt.xlim(-2, len(model_types)*2)

# # draw temporary lines and use them to create a legend
# plt.plot([], c=pal[1], label='All Data')
# plt.plot([], c=pal[0], label='Top of Hour Only')
# plt.legend(loc='lower left', frameon=False)

# plt.subplot(2,2,4) # RMSE FC
# bpl = plt.boxplot(FC_rmse_raw.T, positions=np.array(range(len(FC_rmse_raw.T)))*2.0-0.4, sym='', widths=0.6)
# bpr = plt.boxplot(FC_rmse_toh.T, positions=np.array(range(len(FC_rmse_toh.T)))*2.0+0.4, sym='', widths=0.6)
# set_box_color(bpl, pal[1]) # colors are from http://colorbrewer2.org/
# set_box_color(bpr, pal[0])
# plt.text(-1.85, 0.95*plt.ylim()[1], r'RMSE')

# plt.xticks(range(0, len(model_types) * 2, 2), model_types)
# plt.xlim(-2, len(model_types)*2)

# plt.tight_layout()



if save:
    save_file = os.path.join(save_folder, 'fit_metrics.xlsx')
    with pd.ExcelWriter(save_file) as writer:  
        fit_metrics.to_excel(writer, sheet_name='Rsq and RMSE by Model')
        fit_case.to_excel(writer, sheet_name='Median Rsq and RMSE by Case')
        fit_models.to_excel(writer, sheet_name='Median Rsq and RMSE by Model')
        
        

#%% Performance on HF Data (hf_test_1 only, single, multi, raw, top_of_hour)
print('\n- - HF Test - - ')
# aggregate all cases with hf_test_1 subsets
hf_test_metrics = pd.DataFrame()
perf_mets = ['RMSE','MAPE','AUROC','sens','spec','N','exc']

idx = list(test_cases.hf_test_1.dropna().index)  # All test cases with hf_test_1
lm = len(model_types)  # length of model type list
hf_test_metrics = pd.DataFrame(index=idx, 
                             columns=[7*lm*['ENT']+7*lm*['FC'],
                                      2*(lm*['RMSE']+lm*['MAPE']+lm*['AUROC']+
                                         lm*['sens']+lm*['spec']+lm*['N']+lm*['exc']),
                                      14*model_types])
for t in idx:
    for f in ['ENT','FC']:
        if ('HSB' in t) & (f=='FC'):
            continue
        
        hft = pd.read_csv(os.path.join(case_folder,t,'performance_'+f+'_'+t+'.csv'),
                          index_col=['subset'])
        if 'hf_test_1' not in hft.index:
            continue
        
        hft = hft.loc['hf_test_1'].reset_index().set_index('model')
        for p in perf_mets:
            for m in model_types:    
                hf_test_metrics.loc[t,f].loc[p,m] = hft.loc[m][p]

### Grand Mean per Metric/ Mean by Case/Modely Type
hf_cases = pd.DataFrame(index=hf_test_metrics.index)
hf_model_types = pd.DataFrame(index=model_types)
print('\nGrand Median (HF Test):')
for f in ['ENT','FC']:
    print('\n' + f)
    for p in perf_mets:
        case_med = round(hf_test_metrics[f][p].median(axis=1),3)
        case_med.name = f+'_'+p
        hf_cases = hf_cases.merge(case_med, left_index=True, right_index=True)
        
        model_median = round(hf_test_metrics[f].median().loc[p],3)
        model_median.name = f+'_'+p
        hf_model_types = hf_model_types.merge(model_median, left_index=True, right_index=True)
        model_se = round(pd.Series([bootstrap(hf_test_metrics[f][p][x]) for x in hf_test_metrics[f][p].columns], index=model_types),3)
        model_se.name = f+'_'+p + '_SE'
        hf_model_types = hf_model_types.merge(model_se, left_index=True, right_index=True)
        
        gm = hf_test_metrics[f][p].stack().median()
        se = bootstrap(hf_test_metrics[f][p].stack())
        print(p + ' - ' + str(round(gm,3)) + ' (' + str(round(se,3)) + ')')

all_test = hf_test_metrics['FC'].append(hf_test_metrics['ENT'])
all_test['RMSE'].median()
[bootstrap(all_test['RMSE'][m]) for m in model_types]
all_test['AUROC'].median()
[bootstrap(all_test['AUROC'][m]) for m in model_types]
(all_test['AUROC'] > 0.5).sum()

print('\nMedian By Case')
print(hf_cases[[c for c in hf_cases.columns if 'ENT' in c]])
print('\n')
print(hf_cases[[c for c in hf_cases.columns if 'FC' in c]])

print('\nMedian By Model Type')
print(hf_model_types[[c for c in hf_model_types.columns if 'ENT' in c]])
print('\n')
print(hf_model_types[[c for c in hf_model_types.columns if 'FC' in c]])
        

if save:
    save_file = os.path.join(save_folder, 'hf_test_metrics.xlsx')
    with pd.ExcelWriter(save_file) as writer:  
        hf_test_metrics.to_excel(writer, sheet_name='Performance by Model')
        hf_cases.to_excel(writer, sheet_name='Median Perf by Case')
        hf_model_types.to_excel(writer, sheet_name='Median Perf by Model Type')
        
### Difference between training and HF test
print('\nDifference in RMSE between Train and HF Test')
for f in ['ENT','FC']:
    print(f)
    diff = hf_test_metrics[f]['RMSE'] - fit_metrics[f]['RMSE'].loc[idx]
    se = pd.Series([round(bootstrap(diff[x]),3) for x in diff.columns], index=model_types)
    print(diff)
    print('\nMedian by model type: ')
    print(pd.concat([diff.median(), se], axis=1))
    print('\nGrand Median: ' + str(round(diff.stack().median(),3))+ ' (' + str(round(bootstrap(diff.stack()),3))+')')

# Grand median
diff = hf_test_metrics[f]['RMSE'] - fit_metrics[f]['RMSE'].loc[idx]
diff_e = hf_test_metrics['ENT']['RMSE'] - fit_metrics['ENT']['RMSE'].loc[idx]
diff_e.append(diff).median()
[bootstrap(diff_e.append(diff)[m]) for m in model_types]


### Raw vs. TOH
hf_raw = hf_test_metrics[hf_test_metrics.index.isin(raw_cases)]
hf_toh = hf_test_metrics[hf_test_metrics.index.isin(toh_cases)]

# Ex.       
#stats.mannwhitneyu(hf_raw['ENT','MAPE'].stack(), hf_toh['ENT','MAPE'].stack())
for p in ['RMSE','MAPE','AUROC']:
    print('\n' + p)
    for f in ['ENT','FC']:
        print(f)
        print('Raw/TOH Median - ' + str(hf_raw[f,p].stack().median()) + '/' + 
              str(hf_toh[f,p].stack().median()))
        mw = stats.mannwhitneyu(hf_raw[f,p].stack(), hf_toh[f,p].stack())
        if mw[1] < 0.05:
            print(mw)

diff_toh = (hf_toh.reset_index()[hf_raw.columns] - hf_raw.reset_index()[hf_raw.columns]).stack()
diff_toh = diff_toh['ENT'].append(diff_toh['FC'])
diff_toh.reset_index(inplace=True)
diff_toh.median()
[bootstrap(diff_toh[p])for p in perf_mets]


# Results: 
#No significant difference for RMSE, AUROC, MAPE or sens/spec (N=4 cases x 4 model types)
# Though, median RMSE for ENT is lower for raw data. Lower raw MAPE for both FC/ENT

### Single vs. Multiple Training Events
A=hf_raw['ENT'].T[['LP2','LP4']].loc[['RMSE','MAPE','AUROC']]
B=hf_toh['ENT'].T[['LP9','LP11']].loc[['RMSE','MAPE','AUROC']]

C=hf_raw['FC'].T[['LP2','LP4']].loc[['RMSE','MAPE','AUROC']]
D=hf_toh['FC'].T[['LP9','LP11']].loc[['RMSE','MAPE','AUROC']]

multi_comp = pd.concat([A,B,C,D], axis=1)
multi_comp.to_csv(os.path.join(save_folder,'hf_event_test_multi_comparison.csv'))

# LP2 vs LP 4
print('\nLP2 vs. LP 4')
for p in ['RMSE','MAPE','AUROC']:
    print(p)
    for f in ['ENT','FC']:
        print(f)
        
        med=(hf_raw[f].T.loc[p]['LP4'] - hf_raw[f].T.loc[p]['LP2']).median()
        if np.isnan(med):
            print('NAN')
            continue
        se = bootstrap(hf_raw[f].T.loc[p]['LP4'] - hf_raw[f].T.loc[p]['LP2'])
        print('Median Diff: ' + str(round(med,3)) + ' (' + str(round(se,3)) + ')')
        
        # mw = stats.mannwhitneyu(hf_raw[f].T.loc[p]['LP2'], 
        #                         hf_raw[f].T.loc[p]['LP4'])
        # if mw[1] < 0.05:
        #     print(mw)

# LP9 vs LP 11
print('\nLP9 vs. LP 11')
for p in ['RMSE','MAPE','AUROC']:
    print(p)
    for f in ['ENT','FC']:
        print(f)
        
        med=(hf_toh[f].T.loc[p]['LP11'] - hf_toh[f].T.loc[p]['LP9']).median()
        if np.isnan(med):
            print('NAN')
            continue
        se = bootstrap(hf_toh[f].T.loc[p]['LP11'] - hf_toh[f].T.loc[p]['LP9'])
        print('Median Diff: ' + str(round(med,3)) + ' (' + str(round(se,3)) + ')')
        
        # mw = stats.mannwhitneyu(hf_toh[f].T.loc[p]['LP9'], 
        #                         hf_toh[f].T.loc[p]['LP11'])
        # if mw[1] < 0.05:
        #     print(mw)


#%% Performance on RM Data (rm_test_1 for now, single, multi, raw, top_of_hour)
print('\n- - RM Test - - ')
# aggregate all cases with rm_test_1 subsets
rm_test_metrics = pd.DataFrame()
perf_mets = ['RMSE','MAPE','AUROC','sens','spec','N','exc']

idx = list(test_cases.rm_test_1.dropna().index)  # All test cases with hf_test_1
lm = len(model_types) + 1 # length of model type list plus CM
rm_test_metrics = pd.DataFrame(index=idx, 
                             columns=[7*lm*['ENT']+7*lm*['FC'],
                                      2*(lm*['RMSE']+lm*['MAPE']+lm*['AUROC']+
                                         lm*['sens']+lm*['spec']+lm*['N']+lm*['exc']),
                                      14*(model_types+ ['CM'])])
rm_test_metrics2 = rm_test_metrics.copy()

for t in idx:
    for f in ['ENT','FC']:
        if ('HSB' in t) & (f=='FC'):
            continue
        
        rmt = pd.read_csv(os.path.join(case_folder,t,'performance_'+f+'_'+t+'.csv'),
                          index_col=['subset'])
        if 'rm_test_1' not in rmt.index:
            continue
        
        rmt1 = rmt.loc['rm_test_1'].reset_index().set_index('model')
        rmt2 = rmt.loc['rm_test_2'].reset_index().set_index('model')
        for p in perf_mets:
            for m in model_types + ['CM']:    
                rm_test_metrics.loc[t,f].loc[p,m] = rmt1.loc[m][p]
                rm_test_metrics2.loc[t,f].loc[p,m] = rmt2.loc[m][p]

### Grand Mean per Metric/ Mean by Case/Modely Type
rm_cases = pd.DataFrame(index=rm_test_metrics.index)
rm_model_types = pd.DataFrame(index=model_types)
print('\nGrand Median (RM Test):')
for f in ['ENT','FC']:
    print('\n' + f)
    for p in perf_mets:
        case_med = round(rm_test_metrics[f][p].median(axis=1),3)
        case_med.name = f+'_'+p
        rm_cases = rm_cases.merge(case_med, left_index=True, right_index=True)
        
        model_median = round(rm_test_metrics[f].median().loc[p],3)
        model_median.name = f+'_'+p
        rm_model_types = rm_model_types.merge(model_median, left_index=True, right_index=True)
        model_se = round(pd.Series([bootstrap(rm_test_metrics[f][p][x]) for x in rm_test_metrics[f][p].columns], index=model_types+['CM']),3)
        model_se.name = f+'_'+p + '_SE'
        rm_model_types = rm_model_types.merge(model_se, left_index=True, right_index=True)
        
        gm = case_med.median()
        print(p + ' - ' + str(round(gm,3)))

all_rm_test = rm_test_metrics['FC'].append(rm_test_metrics['ENT'])
all_rm_test['RMSE'].median()
[bootstrap(all_rm_test['RMSE'][m]) for m in model_types+['CM']]
all_rm_test['MAPE'].median()
[bootstrap(all_rm_test['MAPE'][m]) for m in model_types+['CM']]
all_rm_test['AUROC'].median()
[bootstrap(all_rm_test['AUROC'][m]) for m in model_types+['CM']]
all_rm_test['sens'].median()
[bootstrap(all_rm_test['sens'][m]) for m in model_types+['CM']]
all_rm_test['spec'].median()
[bootstrap(all_rm_test['spec'][m]) for m in model_types+['CM']]
(all_rm_test['AUROC'] > 0.5).sum()

print('\nMedian By Case')
print(rm_cases[[c for c in rm_cases.columns if 'ENT' in c]])
print('\n')
print(rm_cases[[c for c in rm_cases.columns if 'FC' in c]])

print('\nMedian By Model Type')
print(rm_model_types[[c for c in rm_model_types.columns if 'ENT' in c]])
print('\n')
print(rm_model_types[[c for c in rm_model_types.columns if 'FC' in c]])
        

if save:
    save_file = os.path.join(save_folder, 'rm_test_metrics.xlsx')
    with pd.ExcelWriter(save_file) as writer:  
        rm_test_metrics.to_excel(writer, sheet_name='Performance by Model')
        rm_cases.to_excel(writer, sheet_name='Median Perf by Case')
        rm_model_types.to_excel(writer, sheet_name='Median Perf by Model Type')
        
### Difference between training and RM test
print('\nDifference in RMSE between Train and RM Test')
for f in ['ENT','FC']:
    print(f)
    diff = rm_test_metrics[f]['RMSE'] - fit_metrics[f]['RMSE'].loc[idx]
    diff = diff.drop('CM', axis=1)
    se = pd.Series([round(bootstrap(diff[x]),3) for x in diff.columns], index=model_types)
    print(diff)
    print('\nMedian by model type: ')
    print(pd.concat([diff.median(), se], axis=1))
    print('\nGrand Median: ' + str(round(diff.stack().median(),3))+ ' (' + str(round(bootstrap(diff.stack()),3))+')')

# Grand median
diff = rm_test_metrics[f]['RMSE'] - fit_metrics[f]['RMSE'].loc[idx]
diff_e = rm_test_metrics['ENT']['RMSE'] - fit_metrics['ENT']['RMSE'].loc[idx]
diff_e.append(diff).median()
[bootstrap(diff_e.append(diff)[m]) for m in model_types]


### Raw vs. TOH
rm_raw = rm_test_metrics[rm_test_metrics.index.isin(raw_cases)]
rm_raw = rm_raw[[c for c in rm_raw.columns if 'CM' not in c]]
rm_toh = rm_test_metrics[rm_test_metrics.index.isin(toh_cases)]
rm_toh = rm_toh[[c for c in rm_toh.columns if 'CM' not in c]]

metrics_models = rm_test_metrics[[c for c in rm_test_metrics.columns if 'CM' not in c]]

# Ex.       
#stats.mannwhitneyu(hf_raw['ENT','MAPE'].stack(), hf_toh['ENT','MAPE'].stack())
for p in perf_mets:
    print('\n' + p)
    for f in ['ENT','FC']:
        print(f)
        print('Raw/TOH Median - ' + str(rm_raw[f,p].stack().median()) + '/' + str(rm_toh[f,p].stack().median()))
        mw = stats.mannwhitneyu(rm_raw[f,p].stack(), rm_toh[f,p].stack())
        if mw[1] < 0.05:
            print(mw)
        print('Median Difference: ' + str(round(np.median(rm_raw[f,p].median(axis=1).dropna().values - rm_toh[f,p].median(axis=1).dropna().values),3)))


# By model type
all_mets = metrics_models['ENT'].append(metrics_models['FC'])
for p in ['RMSE','MAPE','AUROC']:
    print('\n' + p)
    for m in model_types:
        vals = all_mets[p][m]
        print(m + ' - (Raw/TOH Median) - ' + str(round(vals.loc[raw_cases].median(),3)) + ' (' 
              + str(round(bootstrap(vals.loc[raw_cases]),3)) + ') / ' 
              + str(round(vals.loc[toh_cases].median(),3)) + ' (' + str(round(bootstrap(vals.loc[toh_cases]),3)) + ')')
        mw = stats.mannwhitneyu(vals.loc[raw_cases].dropna(), vals.loc[toh_cases].dropna())
        if mw[1] < 0.05:
            print(mw)
        d = vals.loc[raw_cases].dropna().reset_index()[m] - vals.loc[toh_cases].dropna().reset_index()[m]
        print('Med Difference: ' + str(round(np.median(d),3)) + ' (' + str(round(bootstrap(d),3)) + ')')
              
                                                        


# Difference
diff_toh = (rm_toh.reset_index()[rm_raw.columns] - rm_raw.reset_index()[rm_raw.columns]).stack()
diff_toh = diff_toh['ENT'].append(diff_toh['FC'])
diff_toh.reset_index(inplace=True)
diff_toh = diff_toh[diff_toh['level_1']!='CM']
diff_toh.median()
[bootstrap(diff_toh[p])for p in perf_mets]

# Results: 
#No significant difference for RMSE, MAPE or sens/spec (N=4 cases x 4 model types)
# Significantly higher median AUROC for ENT, though low value still
# Though, median RMSE is lower for raw data. Lower raw MAPE for ENT

### Single vs. Multiple Training Events
A=rm_raw['ENT'].T[['LP2','LP4']].loc[['RMSE','MAPE','AUROC']]
diffA = A['LP4'] - A['LP2']
B=rm_toh['ENT'].T[['LP9','LP11']].loc[['RMSE','MAPE','AUROC']]
diffB = B['LP11'] - B['LP9']

C=rm_raw['ENT'].T[['LP3','LP5','LP6','LP7']].loc[['RMSE','MAPE','AUROC']]
D=rm_toh['ENT'].T[['LP10','LP12','LP13','LP14']].loc[['RMSE','MAPE','AUROC']]

E=rm_raw['ENT'].T[['CB2','CB3']].loc[['RMSE','MAPE','AUROC']]
diffE = E['CB3'] - E['CB2']
F=rm_toh['ENT'].T[['CB5','CB6']].loc[['RMSE','MAPE','AUROC']]

G=rm_raw['FC'].T[['LP2','LP4']].loc[['RMSE','MAPE','AUROC']]
H=rm_toh['FC'].T[['LP9','LP11']].loc[['RMSE','MAPE','AUROC']]

I=rm_raw['FC'].T[['LP3','LP5','LP6','LP7']].loc[['RMSE','MAPE','AUROC']]
J=rm_toh['FC'].T[['LP10','LP12','LP13','LP14']].loc[['RMSE','MAPE','AUROC']]

K=rm_raw['FC'].T[['CB2','CB3']].loc[['RMSE','MAPE','AUROC']]
L=rm_toh['FC'].T[['CB5','CB6']].loc[['RMSE','MAPE','AUROC']]

scenarios=[A,B,C,D,E,F,G,H,I,J,K,L]
sfib = 6*['ENT']+6*['FC']
multi_comp = pd.concat(scenarios, axis=1)
multi_comp.to_csv(os.path.join(save_folder,'rm_event_test_multi_comparison.csv'))

# Significantly different RMSE/MAPE/AUROC between single/multi
s_cases = []
m_cases = []

for s in scenarios[0:6]:
    s_cases.append(s.columns[0])
    for i in range(1,len(s.columns)):
        m_cases.append(s.columns[i])

# Result: Considering all models in each case, no sig diff between single(N=40)/multi (N=72)
        
#stats.mannwhitneyu(multi_comp[s_cases].stack().loc['AUROC'], multi_comp[m_cases].stack().loc['AUROC'])
# Out[1131]: MannwhitneyuResult(statistic=1238.0, pvalue=0.11051458338998349)

#stats.mannwhitneyu(multi_comp[s_cases].stack().loc['RMSE'], multi_comp[m_cases].stack().loc['RMSE'])
# Out[1132]: MannwhitneyuResult(statistic=1861.5, pvalue=0.3876412202990539)

# stats.mannwhitneyu(multi_comp[s_cases].stack().loc['MAPE'], multi_comp[m_cases].stack().loc['MAPE'])
# Out[1133]: MannwhitneyuResult(statistic=1695.5, pvalue=0.13511727316755567)

# Median difference (all models and by model)
diff_comp = pd.DataFrame()
pair_comp = pd.DataFrame()
for s in scenarios:
    sing = s[s.columns[0]]
    for i in range(1,len(s.columns)):
        diff_comp = diff_comp.append(s[s.columns[i]] - sing, ignore_index=True)
        temp = pd.DataFrame(data = [[sing.name, s[s.columns[i]].name]], 
                            columns = ['single','multiple'])
        pair_comp = pair_comp.append(temp)
pair_comp.drop_duplicates(inplace=True)
pair_comp.reset_index(inplace=True, drop=True)

# Result: Considering specific model types:
# RMSE/MAPE - Multiple event training led to a median decrease for MLR and GLS models, 
# but small increase for RF and ANN (not significant except for MAPE ANN)
# AUROC - AUROC improved significantly for MLR and ANN, imrpoved for GLS (but not sig.),
# and stayed the same for RF models
        
for p in ['RMSE','AUROC']:
    for m in model_types:
        print('\n' + m + ' - ' + p)
        mm = multi_comp[m_cases].loc[p,m].median()
        sm = multi_comp[s_cases].loc[p,m].median()
        
        me = bootstrap(multi_comp[m_cases].loc[p,m])
        se = bootstrap(multi_comp[s_cases].loc[p,m])

        # print('Median (Single/Double): ' + str(sm) + ' (' + str(round(se,3)) + ') / ' +
        #       str(mm) + ' (' + str(round(me,3)) + ')')
        
        dm = round(diff_comp[(p,m)].median(),3)
        de = round(bootstrap(diff_comp[(p,m)]),3)
        print('Median diff (similar cases multi-sing): ' + str(dm) + ' (' + str(de) + ')')
        
        # sig = 'NO'
        # if stats.mannwhitneyu(multi_comp[m_cases].loc[p,m], multi_comp[s_cases].loc[p,m])[1] < 0.05:
        #     sig = 'YES'
            
        # print('Significant? ' + sig)
        
        for e in pair_comp.index:
            s = pair_comp.loc[e]['single']
            mp = pair_comp.loc[e]['multiple']
            
            sENT = rm_test_metrics.loc[s]['ENT'][p,m]
            mENT = rm_test_metrics.loc[mp]['ENT'][p,m]
            sFC = rm_test_metrics.loc[s]['FC'][p,m]
            mFC = rm_test_metrics.loc[mp]['FC'][p,m]
            
            print('  ' + s +'/'+mp+ ': ' + 
                  'ENT - ' + str(round(sENT,2)) + '/' + str(round(mENT,2)) + ' | ' +
                  'FC - ' + str(round(sFC,2)) + '/' + str(round(mFC,2))
                  )
        

### Current Method
#number models improved on CM metrics in each category
cm = rm_test_metrics[[c for c in rm_test_metrics.columns if 'CM' in c]]
cm_diff = pd.DataFrame(index=metrics_models.index, columns=metrics_models.columns)
for f in ['ENT','FC']:
    for p in perf_mets:
        d = metrics_models[f,p] - cm[f,p].values
        cm_diff.loc[:][f,p] = d

cm_diff_all = cm_diff['ENT'].append(cm_diff['FC'])

# N models by type with improved performance over CM
print('\nN models by type with improved performance over CM')
print('RMSE ' + str(cm_diff_all['RMSE'].count()[0]))
print((cm_diff_all['RMSE'] < 0).sum())
print((cm_diff_all['RMSE'] < 0).sum()/cm_diff_all['RMSE'].count())

print('\nMAPE '+ str(cm_diff_all['MAPE'].count()[0]))
print((cm_diff_all['MAPE'] < 0).sum())
print((cm_diff_all['MAPE'] < 0).sum()/cm_diff_all['MAPE'].count())
      
print('\nAUROC '+ str(cm_diff_all['AUROC'].count()[0]))
print((cm_diff_all['AUROC'].dropna() > 0).sum())
print((cm_diff_all['AUROC'].dropna() > 0).sum()/cm_diff_all['AUROC'].count())

# Median difference in perf from CM
cm_diff_se = [bootstrap(cm_diff_all[c]) for c in cm_diff_all.columns]
cm_diff_med = pd.concat([cm_diff_all.median(), pd.Series(cm_diff_se, index=cm_diff_all.columns)], axis=1)

# Results: 
# RMSE - RF and ANN had lower RMSE on average, inconclusive with MLR/GLS (large SE)
# MAPE - All model types had higher MAPE than CM, though RF and ANN less so (ANN sometimes lower)
# AUROC - RF largely improved on AUROC, ANN and GLS little to no improvement, MLR was inconclusive (large SE)


## N Raw vs TOH with improved performnance
print('\nN where raw vs toh')
print('RMSE ')
print('Raw: ' + str((cm_diff_all.loc[raw_cases]['RMSE'].stack()<0).sum()/
                       (cm_diff_all.loc[raw_cases]['RMSE'].stack()).count()) + 
      ' (N = ' + str(cm_diff_all.loc[raw_cases]['RMSE'].stack().count())+')')
print('TOH: ' + str((cm_diff_all.loc[toh_cases]['RMSE'].stack()<0).sum()/
                       (cm_diff_all.loc[toh_cases]['RMSE'].stack()).count()) + 
      ' (N = ' + str(cm_diff_all.loc[toh_cases]['RMSE'].stack().count())+')')

print('\nMAPE ')
print('Raw: ' + str((cm_diff_all.loc[raw_cases]['MAPE'].stack()<0).sum()/
                       (cm_diff_all.loc[raw_cases]['MAPE'].stack()).count()) + 
      ' (N = ' + str(cm_diff_all.loc[raw_cases]['MAPE'].stack().count())+')')
print('TOH: ' + str((cm_diff_all.loc[toh_cases]['MAPE'].stack()<0).sum()/
                       (cm_diff_all.loc[toh_cases]['MAPE'].stack()).count()) + 
      ' (N = ' + str(cm_diff_all.loc[toh_cases]['MAPE'].stack().count())+')')

print('\nAUROC ')
print('Raw: ' + str((cm_diff_all.loc[raw_cases]['AUROC'].stack().dropna()>0).sum()/
                       (cm_diff_all.loc[raw_cases]['AUROC'].stack()).count()) + 
      ' (N = ' + str(cm_diff_all.loc[raw_cases]['AUROC'].stack().count())+')')
print('TOH: ' + str((cm_diff_all.loc[toh_cases]['AUROC'].stack().dropna()>0).sum()/
                       (cm_diff_all.loc[toh_cases]['AUROC'].stack()).count()) + 
      ' (N = ' + str(cm_diff_all.loc[toh_cases]['AUROC'].stack().count())+')')



## Single vs multi improve over CM
print('\nN by model typewith improved performnance')
print('RMSE ')
print('Single: ' + str((cm_diff_all.loc[s_cases]['RMSE'].stack()<0).sum()/
                       (cm_diff_all.loc[s_cases]['RMSE'].stack()).count()) + 
      ' (N = ' + str(cm_diff_all.loc[s_cases]['RMSE'].stack().count())+')')
print('Multiple: ' + str((cm_diff_all.loc[m_cases]['RMSE'].stack()<0).sum()/
                       (cm_diff_all.loc[m_cases]['RMSE'].stack()).count()) + 
      ' (N = ' + str(cm_diff_all.loc[m_cases]['RMSE'].stack().count())+')')

print('\nMAPE ')
print('Single: ' + str((cm_diff_all.loc[s_cases]['MAPE'].stack()<0).sum()/
                       (cm_diff_all.loc[s_cases]['MAPE'].stack()).count()) + 
      ' (N = ' + str(cm_diff_all.loc[s_cases]['MAPE'].stack().count())+')')
print('Multiple: ' + str((cm_diff_all.loc[m_cases]['MAPE'].stack()<0).sum()/
                       (cm_diff_all.loc[m_cases]['MAPE'].stack()).count()) + 
      ' (N = ' + str(cm_diff_all.loc[m_cases]['MAPE'].stack().count())+')')

print('\nAUROC ')
print('Single: ' + str((cm_diff_all.loc[s_cases]['AUROC'].stack().dropna()>0).sum()/
                       (cm_diff_all.loc[s_cases]['AUROC'].stack()).count()) + 
      ' (N = ' + str(cm_diff_all.loc[s_cases]['AUROC'].stack().count())+')')
print('Multiple: ' + str((cm_diff_all.loc[m_cases]['AUROC'].stack().dropna()>0).sum()/
                       (cm_diff_all.loc[m_cases]['AUROC'].stack()).count()) + 
      ' (N = ' + str(cm_diff_all.loc[m_cases]['AUROC'].stack().count())+')')

### Diff between year 1 and year 2
y1 = rm_test_metrics
y2 = rm_test_metrics2


#%% Plots
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

params = {
   'axes.labelsize': 9,
   'font.size': 9,
   'legend.fontsize': 9,
   'xtick.labelsize': 9,
   'ytick.labelsize': 9,
   'font.family'  : 'sans-serif',
   'font.sans-serif':'Helvetica',
   'axes.axisbelow': True
   }
plt.rcParams.update(params)


# ### Future Event Boxplots - RMSE, MAPE, AUROC
# hf_plot = hf_test_metrics.T.stack().reset_index()
# hf_plot.columns = ['FIB','metric','model','case','value']
# hf_plot['interval'] = ['All Data' if x in raw_cases else 'Top of Hour' for x in hf_plot['case']]


# pal = ['#969696','#525252']
# pal = sns.color_palette(pal)

# #Vertical
# plt.figure(figsize=(10,6))
# c=1

# for f in ['ENT','FC']:
#     A = hf_plot[hf_plot.FIB==f]
#     for p in ['RMSE','MAPE','AUROC']:
#         plt.subplot(2,3,c)
        
#         ax = sns.boxplot(x='model',
#                          y='value',
#                          #hue='interval',
#                          data=A[A.metric==p], 
#                          palette=pal)
        
#         if p == 'RMSE':
#             plt.ylabel(p + r' ($log_{10}MPN/100ml$)')
#         elif p == 'MAPE':
#             plt.ylabel(p + r' (%)')
#         else:
#             plt.ylabel(p)
        
#         plt.xlabel('')
        
#         # if c == 6:
#         #     #plt.legend(['All Data','Top of Hour'], frameon=False, loc='lower right')
#         #     plt.legend(frameon=False, loc='lower right')
#         # else:
#         plt.legend('',frameon=False)
        
#         if c in [1,4]:
#             plt.text(-.45, 0.92*plt.ylim()[1],f) 
        
#         c+=1
    
#         plt.tight_layout()


# ### RM  Boxplots - RMSE, MAPE, AUROC
# rm_plot = rm_test_metrics.T.stack().reset_index()
# rm_plot.columns = ['FIB','metric','model','case','value']
# rm_plot['interval'] = ['All Data' if x in raw_cases else 'Top of Hour' for x in rm_plot['case']]


# pal = ['#969696','#525252']
# pal = sns.color_palette(pal)

# #Vertical
# plt.figure(figsize=(10,6))
# c=1

# for f in ['ENT','FC']:
#     A = rm_plot[rm_plot.FIB==f]
#     for p in ['RMSE','MAPE','AUROC']:
#         plt.subplot(2,3,c)
        
#         ax = sns.boxplot(x='model',
#                          y='value',
#                          #hue='interval',
#                          data=A[A.metric==p], 
#                          palette=pal)
        
#         if p == 'RMSE':
#             plt.ylabel(p + r' ($log_{10}MPN/100ml$)')
#         elif p == 'MAPE':
#             plt.ylabel(p + r' (%)')
#         else:
#             plt.ylabel(p)
        
#         plt.xlabel('')
        
#         if c == 6:
#             #plt.legend(['All Data','Top of Hour'], frameon=False, loc='lower right')
#             plt.legend(frameon=False, loc='lower right')
#         else:
#             plt.legend('',frameon=False)
        
#         if c in [1,4]:
#             plt.text(-.45, 0.92*plt.ylim()[1],f) 
        
#         c+=1
    
#         plt.tight_layout()

### Boxplots - Training (R2, RMSE) /HF Test/ RM Test  (RMSE, AUROC)
# Combine FIB
# By model type (CM in the RM plots)
# N models per box?

fit_plot = fit_metrics.T.stack().reset_index()
fit_plot.columns = ['FIB','metric','model','case','value']
fit_plot['interval'] = ['All Data' if x in raw_cases else 'Top of Hour' for x in fit_plot['case']]

hf_plot = hf_test_metrics.T.stack().reset_index()
hf_plot.columns = ['FIB','metric','model','case','value']
hf_plot['interval'] = ['All Data' if x in raw_cases else 'Top of Hour' for x in hf_plot['case']]

rm_plot = rm_test_metrics.T.stack().reset_index()
rm_plot.columns = ['FIB','metric','model','case','value']
rm_plot['interval'] = ['All Data' if x in raw_cases else 'Top of Hour' for x in rm_plot['case']]


pal = ['#969696','#525252']
pal = sns.color_palette(pal)

sns.set_palette('colorblind', n_colors=5,desat=None)

plt.figure(figsize=(6,5))

# Fit
plt.subplot(231) # RMSE
ax = sns.boxplot(x='model',
                     y='value',
                     #hue='interval',
                     data=fit_plot[fit_plot.metric=='RMSE'],
                     width=0.4,
                     fliersize=3,
                     #palette=pal
                     )
plt.xlabel('')
ax.set_xticklabels(['','','',''])
plt.ylabel(r'RMSE ($log_{10}MPN/100ml$)')
plt.ylim(0,2.1)
plt.title('Training', loc='center')

ax.spines['left'].set_position(('outward', 10))
ax.spines['bottom'].set_position(('outward', 5))
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.get_lines()[2].set_color('white')
ax.get_lines()[3].set_color('white')
ax.get_lines()[8].set_color('white')
ax.get_lines()[9].set_color('white')
ax.get_lines()[14].set_color('white')
ax.get_lines()[15].set_color('white')


plt.subplot(234) # R2
ax = sns.boxplot(x='model',
                     y='value',
                     #hue='interval',
                     data=fit_plot[fit_plot.metric=='Rsq'],
                     width=0.4,
                     fliersize=3,
                     #palette=pal
                     )
plt.xlabel('')
plt.ylabel(r'$R^2$')
plt.ylim(-0.45,1.05)

ax.spines['left'].set_position(('outward', 10))
ax.spines['bottom'].set_position(('outward', 5))
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.get_lines()[2].set_color('white')
ax.get_lines()[3].set_color('white')
ax.get_lines()[8].set_color('white')
ax.get_lines()[9].set_color('white')
ax.get_lines()[14].set_color('white')
ax.get_lines()[15].set_color('white')


# Event Test
plt.subplot(232) # RMSE
ax = sns.boxplot(x='model',
                     y='value',
                     #hue='interval',
                     data=hf_plot[hf_plot.metric=='RMSE'],
                     width=0.4,
                     fliersize=3,
                     #palette=pal
                     )
plt.xlabel('')
ax.set_xticklabels(['','','',''])
plt.ylabel(r'RMSE ($log_{10}MPN/100ml$)')
plt.ylim(0,2.1)
plt.title('HF Test', loc='center')

ax.spines['left'].set_position(('outward', 10))
ax.spines['bottom'].set_position(('outward', 5))
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.get_lines()[2].set_color('white')
ax.get_lines()[3].set_color('white')
ax.get_lines()[8].set_color('white')
ax.get_lines()[9].set_color('white')
ax.get_lines()[14].set_color('white')
ax.get_lines()[15].set_color('white')

plt.subplot(235) # AUROC
ax = sns.boxplot(x='model',
                     y='value',
                     #hue='interval',
                     data=hf_plot[hf_plot.metric=='AUROC'],
                     width=0.4,
                     fliersize=3,
                     #palette=pal
                     )
plt.xlabel('')
plt.ylabel('AUROC')

ax.spines['left'].set_position(('outward', 10))
ax.spines['bottom'].set_position(('outward', 5))
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.get_lines()[2].set_color('white')
ax.get_lines()[3].set_color('white')
ax.get_lines()[8].set_color('white')
ax.get_lines()[9].set_color('white')
ax.get_lines()[14].set_color('white')
ax.get_lines()[15].set_color('white')

# RM Test
plt.subplot(233) # RMSE
ax = sns.boxplot(x='model',
                     y='value',
                     #hue='interval',
                     data=rm_plot[rm_plot.metric=='RMSE'],
                     fliersize=3,
                     width=0.5,
                     #palette=pal
                     )
plt.xlabel('')
ax.set_xticklabels(['','','','',''])
#plt.ylabel(r'RMSE ($log_{10}MPN/100ml$)')
plt.ylabel('')
plt.ylim(0,2.1)
plt.title('RM Test', loc='center')

ax.spines['left'].set_position(('outward', 10))
ax.spines['bottom'].set_position(('outward', 5))
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.get_lines()[2].set_color('white')
ax.get_lines()[3].set_color('white')
ax.get_lines()[8].set_color('white')
ax.get_lines()[9].set_color('white')
ax.get_lines()[14].set_color('white')
ax.get_lines()[15].set_color('white')
ax.get_lines()[20].set_color('white')
ax.get_lines()[21].set_color('white')

plt.subplot(236) # AUROC
ax = sns.boxplot(x='model',
                     y='value',
                     #hue='interval',
                     data=rm_plot[rm_plot.metric=='AUROC'],
                     fliersize=3,
                     width=0.5,
                     #palette=pal
                     )
plt.xlabel('')
#plt.ylabel('AUROC')
plt.ylabel('')

ax.spines['left'].set_position(('outward', 10))
ax.spines['bottom'].set_position(('outward', 5))
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.get_lines()[2].set_color('white')
ax.get_lines()[3].set_color('white')
ax.get_lines()[8].set_color('white')
ax.get_lines()[9].set_color('white')
ax.get_lines()[14].set_color('white')
ax.get_lines()[15].set_color('white')
ax.get_lines()[20].set_color('white')
ax.get_lines()[21].set_color('white')
    
# plt.text(-.45, 0.92*plt.ylim()[1],f) 


plt.tight_layout()
plt.subplots_adjust(top=0.935,
bottom=0.084,
left=0.125,
right=0.988,
hspace=0.126,
wspace=0.435)


# ### Bar chart median performance with all cases, single event, multi event, all data, 
# rm_plot = rm_test_metrics.T.stack().reset_index()
# rm_plot.columns = ['FIB','metric','model','case','value']
# rm_plot['interval'] = ['All Data' if x in raw_cases else 'Top of Hour' for x in rm_plot['case']]

# plt.figure()

                    