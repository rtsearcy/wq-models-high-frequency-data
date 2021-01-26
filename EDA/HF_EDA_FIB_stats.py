#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HF_EDA_FIB_stats.py

Created on Fri Jun 12 15:39:56 2020
@author: rtsearcy

Statistics related to the FIB variability for all HF events

Sections:
    - Basic Stats: Range, mean, variance, median, % exceedances/abloq
        - Variability: CV, # samples for % precision, RMSE of event, change between samples
    - Time of day stats: Morning vs. Afternoon vs. Night
        - Individual events
        - All events

"""

import pandas as pd
import os
import datetime
import numpy as np
from scipy import stats
from scipy.stats.mstats import gmean
import statsmodels.api as sm
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf, pacf
import wq_modeling as wqm

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 100)

folder = '/Users/rtsearcy/Box/water_quality_modeling/thfs/EDA'
save_folder = '/Users/rtsearcy/Box/water_quality_modeling/thfs/EDA/summer2020'

LP = pd.read_csv(os.path.join(folder, 'all_event_data_LP.csv'), parse_dates=['dt'], index_col=['dt'])
CB = pd.read_csv(os.path.join(folder, 'all_event_data_CB.csv'), parse_dates=['dt'], index_col=['dt'])
HSB = pd.read_csv(os.path.join(folder, 'all_event_data_HSB.csv'), parse_dates=['dt'], index_col=['dt'])

df = pd.concat([LP,CB,HSB])

hf = df[df.event.isin(['LP-13','LP-16','LP-18','CB-11','CB-12','HSB-02'])]
hf['logEC']= hf['logFC']

# Daytime = between sunrise and sunset
df_log = pd.read_csv('/Users/rtsearcy/Box/water_quality_modeling/thfs/EDA/summer2020/logistics/logistics_all_events.csv', 
                     index_col=['event'])

for e in hf.event.unique():
    sr = pd.to_datetime(df_log.loc[e]['sunrise1']).time()
    ss = pd.to_datetime(df_log.loc[e]['sunset1']).time()
    hf.loc[hf.event==e,'daytime'] = [1 if (x > sr) and (x < ss) else 0 for x in hf[hf.event==e].index.time] 

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

# trad = df[df.event.isin(['LP-RM-FIB','CB-RM','HSB-RM'])]
# trad = trad['2000':'2020']
# trad_vars = df[df.event.isin(['LP-RM','CB-RM','HSB-RM'])]
# trad_vars = trad_vars['2000':'2020']

#%% Basic Stats
# Note: Boxplots in HF_compare_beaches_bin_plots.py

df_basic = pd.DataFrame()
for e in hf.event.unique():
    print(e)
    df_fib = hf[hf.event==e][['TC','FC','ENT']]
    df_stats = df_fib.describe()
    df_stats.rename(index={'count':'N'}, inplace=True)

    # Geometric Mean
    gm = pd.Series(gmean(df_fib), index = ['TC','FC','ENT'])
    gm.name = 'gm'
    df_stats = df_stats.append(gm.T)

    # Variance
    vari = df_fib.describe().loc['std']**2
    vari.name = 'variance'
    df_stats = df_stats.append(vari.T)
    
    # CV [event standard deviation / mean]
    # stdev normalized by N-1
    cv = df_fib.describe().loc['std'] / np.abs(df_fib.describe().loc['mean'])
    cv.name = 'CV'
    df_stats = df_stats.append(cv.T)
    
    #
    
    # Change between samples (see Boehm 2007, Table 1)
    dif = abs(df_fib.diff())
    # Average
    change_mean = dif.mean()
    change_mean.name = 'change_mean'
    df_stats = df_stats.append(change_mean.T)
    # Max
    change_min = dif.min()
    change_min.name = 'change_min'
    df_stats = df_stats.append(change_min.T)
    # Min
    change_max = dif.max()
    change_max.name = 'change_max'
    df_stats = df_stats.append(change_max.T)
    
    
    # delta (see Boehm 2007) - change in FIB normalized by mean FIB
    dif = abs(df_fib.diff()/df_fib.mean())
    print('delta = 0 -> FC: ' + str((dif['FC']==0).sum()) + '  , ENT: ' + str((dif['ENT']==0).sum()))
    # Average
    delta_mean = dif.mean()
    delta_mean.name = 'delta_mean'
    df_stats = df_stats.append(delta_mean.T)
    # Max
    delta_min = dif.min()
    delta_min.name = 'delta_min'
    df_stats = df_stats.append(delta_min.T)
    # Min
    delta_max = dif.max()
    delta_max.name = 'delta_max'
    df_stats = df_stats.append(delta_max.T)
    
    # Skewness/Kurtosis
    skew = df_fib.skew()
    skew.name = 'skewness'
    df_stats = df_stats.append(skew.T)
    
    kurt = df_fib.kurtosis()
    kurt.name = 'kurtosis'
    df_stats = df_stats.append(kurt.T)
    
    # At or Below Level of Quantification
    bloq = (df_fib == 10).sum()
    bloq.name = 'abloq'
    df_stats = df_stats.append(bloq.T)
    df_stats.loc['abloq_%'] = round(100*df_stats.loc['abloq']/df_stats.loc['N'],1)
    
    # Exceedances
    exc = pd.Series()
    for f in ['TC','FC','ENT']:
        exc[f] = (df_fib[f] > wqm.fib_thresh(f)).sum()
    exc.name = 'exc'
    df_stats = df_stats.append(exc.T)
    df_stats.loc['exc_%'] = round(100*df_stats.loc['exc']/df_stats.loc['N'],1)

    # Shanon Entropy
    shan = pd.Series()
    for f in ['TC','FC','ENT']:
        vals, counts = np.unique(df_fib[f],return_counts=True)
        shan[f] = round(stats.entropy(counts/len(df_fib[f])),3)
    shan.name = 'Shannon'
    df_stats = df_stats.append(shan.T)
    
    # Append to basic stats df
    df_stats = df_stats.T
    df_stats.index.name = 'FIB'
    df_stats['event'] = e
    df_stats.reset_index(inplace=True)
    df_basic = df_basic.append(df_stats)

df_basic.set_index(['event'], inplace=True)

cols = ['FIB', 'N', 'abloq','abloq_%', 'exc', 'exc_%',
        'mean', 'gm', 'min', '25%', '50%', '75%', 'max',
       'std', 'variance','CV', 'skewness', 'kurtosis',
       'change_mean', 'change_min', 'change_max',
       'delta_mean', 'delta_min', 'delta_max','Shannon']
df_basic = df_basic[cols]  # Reorder
df_basic = round(df_basic,3)
df_basic.to_csv(os.path.join(save_folder,'basic_stats_FIB_all_events.csv'))

print(df_basic[['FIB', 'N', 'abloq','abloq_%', 'exc', 'exc_%']])
print('\n')
print(df_basic[['FIB', 'mean', 'gm', 'min', '50%', 'max', 'std']])
print('\n')
print(df_basic[['CV', 'change_mean', 'change_max','delta_mean', 'delta_max']])

#%% Daytime/Nightime - All Data
# # Daytime = sunrise to sunset (hours ~6-19)
day = hf[hf.daytime == 1]
night = hf[hf.daytime == 0]

# Overall

# Means
stats.mannwhitneyu(day['logENT'], night['logENT']) # Night mean = 1.4*Day mean
stats.mannwhitneyu(day['logFC'], night['logFC']) # Night mean = 1.24*Day mean
# Result: Both FC and ENT are higher at nighttime

# Variability
stats.levene(day['logENT'], night['logENT'])
stats.levene(day['logFC'].dropna(), night['logFC'].dropna())

# Overall FC different between night and day, but ENT is not


#%% Solar Noon - All data
# Solar Noon (1) = hours = 9-3, (0) = 6-9 or 3-7
sn = hf[(hf.index.time >= datetime.time(9,0)) & (hf.index.time < datetime.time(15,0))]
not_sn = hf[~hf.index.isin(sn.index)][hf.daytime==1]

stats.mannwhitneyu(sn['logENT'], not_sn['logENT'])
sn['logENT'].mean()/ not_sn['logENT'].mean()

stats.mannwhitneyu(sn['logFC'], not_sn['logFC'])
sn['logFC'].mean()/ not_sn['logFC'].mean()

# Results: Looking at all events, samples taken near solar noon (9-3) were 
# significantly lower than other daytime samples. THought not always the case individually

# Variance
stats.levene(sn['logENT'], not_sn['logENT'])
stats.levene(sn['logFC'].dropna(), not_sn['logFC'].dropna())

# Results - Overall, variance is different between solar noon and not solar noon
# ENT - higher during not sn; FC higher during sn

#%%  Distribution Means Difference
# Kolmogorov Smirnov - Tests if two samples come from same distribution
# significant test rejects the null hyp.

# RESULT: Samples generally showed different distributions between events
# All beach dist. sig different
#All event distr. sig. different EXCEPT ENT during LP-16 and LP-18
# Perhaps affected by large censorship of data?

stats.ks_2samp(lp13['logENT'],lp16['logENT'])
stats.ks_2samp(lp13['logENT'],lp18['logENT'])
stats.ks_2samp(lp16['logENT'],lp18['logENT'])
stats.ks_2samp(lp16['logFC'],lp18['logFC'])
stats.ks_2samp(lp13['logFC'],lp18['logFC'])
stats.ks_2samp(lp13['logFC'],lp16['logFC'])
stats.ks_2samp(cb11['logFC'],cb12['logFC'])
stats.ks_2samp(cb11['logENT'],cb12['logENT'])
stats.ks_2samp(cb['logENT'],lp['logENT'])
stats.ks_2samp(cb['logFC'],lp['logFC'])
stats.ks_2samp(hsb['logFC'],lp['logFC'])
stats.ks_2samp(hsb02['logFC'],lp['logFC'])
stats.ks_2samp(hsb02['logENT'],lp['logENT'])
stats.ks_2samp(hsb02['logENT'],cb['logENT'])


#%% Time of Day Comparison Stats [by EVENT] - Differences in Central Tendency

# Mann-Whitney U test - non-parametric test between two sets,
# significant test indicates different means

# Daytime/Nighttime by event (Daytime (1) = 6-19, Nightime (0) 19-6)
E = hf.groupby(['beach','event','daytime']).mean()['logENT']
F = hf.groupby(['beach','event','daytime']).mean()['logFC']
daytime_means = round(pd.merge(E,F, left_index=True, right_index=True),3)
count_daytime = hf.groupby(['beach','event','daytime']).size() # No. samples
count_daytime.name = 'N'
daytime_means = pd.merge(count_daytime,daytime_means, left_index=True, right_index=True)

daytime_means['ENT_exc_%'] = hf.groupby(['beach','event','daytime','ENT_exc']).size().xs(1, level=3, drop_level=True) / count_daytime
daytime_means['FC_exc_%'] = hf.groupby(['beach','event','daytime','FC_exc']).size().xs(1, level=3, drop_level=True) / count_daytime


daytime_sig = pd.DataFrame(index=hf.event.unique(),columns=['MW-U_ENT','p_ENT', 'MW-U_FC','p_FC'])
for e in hf.event.unique():
    daytime_sig.loc[e]['MW-U_ENT','p_ENT'] = list(stats.mannwhitneyu(day[day.event==e]['logENT'], night[night.event==e]['logENT']))
    daytime_sig.loc[e]['MW-U_FC','p_FC'] = list(stats.mannwhitneyu(day[day.event==e]['logFC'], night[night.event==e]['logFC']))

print('BY EVENT: ')
print('Daytime/Nighttime')
print(daytime_means)
print('\n')
print(daytime_sig)


# Solar Noon
hf['solar_noon'] = [1 if x in sn.index else 0 for x in hf.index]  # See previous section for SN

E = hf[hf.daytime==1].groupby(['beach','event','solar_noon']).mean()['logENT']
F = hf[hf.daytime==1].groupby(['beach','event','solar_noon']).mean()['logFC']
MAN_means = round(pd.merge(E,F, left_index=True, right_index=True),3)

count_MAN = hf[hf.daytime==1].groupby(['beach','event','solar_noon']).size()  # No. samples
count_MAN.name = 'N'
MAN_means = pd.merge(count_MAN,MAN_means, left_index=True, right_index=True)

MAN_means['ENT_exc_%'] = hf[hf.daytime==1].groupby(['beach','event','solar_noon','ENT_exc']).size().xs(1, level=3, drop_level=True) / count_MAN
MAN_means['FC_exc_%'] = hf[hf.daytime==1].groupby(['beach','event','solar_noon','FC_exc']).size().xs(1, level=3, drop_level=True) / count_MAN


# Test if morning and afternoon means are different
A = hf[hf.daytime==1][hf.solar_noon == 1]
B = hf[hf.daytime==1][hf.solar_noon == 0]

# By events
MAN_sig = pd.DataFrame(index=hf.event.unique(),columns=['MW-U_ENT','p_ENT', 'MW-U_FC','p_FC'])
for e in hf.event.unique():
    MAN_sig.loc[e]['MW-U_ENT','p_ENT'] = list(stats.mannwhitneyu(A[A.event==e]['logENT'], B[B.event==e]['logENT']))
    MAN_sig.loc[e]['MW-U_FC','p_FC'] = list(stats.mannwhitneyu(A[A.event==e]['logFC'], B[B.event==e]['logFC']))

print('\nSolar Noon')
print(MAN_means)
print('\n')
print(MAN_sig)

daytime_means_e = daytime_means
daytime_sig_e = daytime_sig
MAN_means_e = MAN_means
MAN_sig_e = MAN_sig


#%% Time of Day Comparison Stats [by BEACH] 

# Hour of day bin plots in other script

# Mann-Whitney U test - non-parametric test between two sets,
# significant test indicates different means

# Daytime/Nighttime by event (Daytime (1) = 6-19)
E = hf.groupby(['beach','daytime']).mean()['logENT']
F = hf.groupby(['beach','daytime']).mean()['logFC']
daytime_means = round(pd.merge(E,F, left_index=True, right_index=True),3)
count_daytime = hf.groupby(['beach','daytime']).size() # No. samples
count_daytime.name = 'N'
daytime_means = pd.merge(count_daytime,daytime_means, left_index=True, right_index=True)

daytime_means['ENT_exc_%'] = hf.groupby(['beach','daytime','ENT_exc']).size().xs(1, level=2, drop_level=True) / count_daytime
daytime_means['FC_exc_%'] = hf.groupby(['beach','daytime','FC_exc']).size().xs(1, level=2, drop_level=True) / count_daytime

# By beach
daytime_sig = pd.DataFrame(index=hf.beach.unique(),columns=['MW-U_ENT','p_ENT', 'MW-U_FC','p_FC'])
for e in hf.beach.unique():
    daytime_sig.loc[e]['MW-U_ENT','p_ENT'] = list(stats.mannwhitneyu(day[day.beach==e]['logENT'], night[night.beach==e]['logENT']))
    daytime_sig.loc[e]['MW-U_FC','p_FC'] = list(stats.mannwhitneyu(day[day.beach==e]['logFC'], night[night.beach==e]['logFC']))

print('BY BEACH: ')
print('Daytime/Nighttime')
print(daytime_means)
print('\n')
print(daytime_sig)


# Solar Noon
hf['solar_noon'] = [1 if x in sn.index else 0 for x in hf.index]  # See previous section for SN

E = hf[hf.daytime==1].groupby(['beach','solar_noon']).mean()['logENT']
F = hf[hf.daytime==1].groupby(['beach','solar_noon']).mean()['logFC']
MAN_means = round(pd.merge(E,F, left_index=True, right_index=True),3)

count_MAN = hf[hf.daytime==1].groupby(['beach','solar_noon']).size()  # No. samples
count_MAN.name = 'N'
MAN_means = pd.merge(count_MAN,MAN_means, left_index=True, right_index=True)

MAN_means['ENT_exc_%'] = hf[hf.daytime==1].groupby(['beach','solar_noon','ENT_exc']).size().xs(1, level=2, drop_level=True) / count_MAN
MAN_means['FC_exc_%'] = hf[hf.daytime==1].groupby(['beach','solar_noon','FC_exc']).size().xs(1, level=2, drop_level=True) / count_MAN


# Test if morning and afternoon means are different
A = hf[hf.daytime==1][hf.solar_noon == 1]
B = hf[hf.daytime==1][hf.solar_noon == 0]

# By beach
MAN_sig = pd.DataFrame(index=hf.beach.unique(),columns=['MW-U_ENT','p_ENT', 'MW-U_FC','p_FC'])
for e in hf.beach.unique():
    MAN_sig.loc[e]['MW-U_ENT','p_ENT'] = list(stats.mannwhitneyu(A[A.beach==e]['logENT'], B[B.beach==e]['logENT']))
    MAN_sig.loc[e]['MW-U_FC','p_FC'] = list(stats.mannwhitneyu(A[A.beach==e]['logFC'], B[B.beach==e]['logFC']))

print('\nSolar Noon')
print(MAN_means)
print('\n')
print(MAN_sig)


# SAVE
with pd.ExcelWriter(os.path.join(save_folder,'FIB_time_of_day_stats.xlsx')) as writer:  
    daytime_means.to_excel(writer, sheet_name='daytime_means_beach')
    daytime_sig.to_excel(writer, sheet_name='daytime_sig_beach')
    MAN_means.to_excel(writer, sheet_name='solar_noon_means_beach')
    MAN_sig.to_excel(writer, sheet_name='solar_noon_sig_beach')
    
    daytime_means_e.to_excel(writer, sheet_name='daytime_means_event')
    daytime_sig_e.to_excel(writer, sheet_name='daytime_sig_event')
    MAN_means_e.to_excel(writer, sheet_name='solar_noon_means_event')
    MAN_sig_e.to_excel(writer, sheet_name='solar_noon_sig_event')

#%% Time of Day Comparison Stats [by EVENT] - Variances

# Levene test - non-parametric test between two sets,
# significant test indicates different variances

# Daytime/Nighttime by event (Daytime (1) = 6-19, Nightime (0) 19-6)
E = hf.groupby(['beach','event','daytime']).var()['logENT']
F = hf.groupby(['beach','event','daytime']).var()['logFC']
daytime_means = round(pd.merge(E,F, left_index=True, right_index=True),3)
count_daytime = hf.groupby(['beach','event','daytime']).size() # No. samples
count_daytime.name = 'N'
daytime_means = pd.merge(count_daytime,daytime_means, left_index=True, right_index=True)

daytime_means['ENT_exc_%'] = hf.groupby(['beach','event','daytime','ENT_exc']).size().xs(1, level=3, drop_level=True) / count_daytime
daytime_means['FC_exc_%'] = hf.groupby(['beach','event','daytime','FC_exc']).size().xs(1, level=3, drop_level=True) / count_daytime


daytime_sig = pd.DataFrame(index=hf.event.unique(),columns=['Lev_ENT','p_ENT', 'Lev_FC','p_FC'])
for e in hf.event.unique():
    daytime_sig.loc[e]['Lev_ENT','p_ENT'] = list(stats.levene(day[day.event==e]['logENT'], night[night.event==e]['logENT']))
    daytime_sig.loc[e]['Lev_FC','p_FC'] = list(stats.levene(day[day.event==e]['logFC'], night[night.event==e]['logFC']))

print('VARIANCES BY EVENT: ')
print('Daytime/Nighttime')
print(daytime_means)
print('\n')
print(daytime_sig)


# Solar Noon
hf['solar_noon'] = [1 if x in sn.index else 0 for x in hf.index]  # See previous section for SN

E = hf[hf.daytime==1].groupby(['beach','event','solar_noon']).var()['logENT']
F = hf[hf.daytime==1].groupby(['beach','event','solar_noon']).var()['logFC']
MAN_means = round(pd.merge(E,F, left_index=True, right_index=True),3)

count_MAN = hf[hf.daytime==1].groupby(['beach','event','solar_noon']).size()  # No. samples
count_MAN.name = 'N'
MAN_means = pd.merge(count_MAN,MAN_means, left_index=True, right_index=True)

MAN_means['ENT_exc_%'] = hf[hf.daytime==1].groupby(['beach','event','solar_noon','ENT_exc']).size().xs(1, level=3, drop_level=True) / count_MAN
MAN_means['FC_exc_%'] = hf[hf.daytime==1].groupby(['beach','event','solar_noon','FC_exc']).size().xs(1, level=3, drop_level=True) / count_MAN


# Test if morning and afternoon means are different
A = hf[hf.daytime==1][hf.solar_noon == 1]
B = hf[hf.daytime==1][hf.solar_noon == 0]

# By events
MAN_sig = pd.DataFrame(index=hf.event.unique(),columns=['Lev_ENT','p_ENT', 'Lev_FC','p_FC'])
for e in hf.event.unique():
    MAN_sig.loc[e]['Lev_ENT','p_ENT'] = list(stats.levene(A[A.event==e]['logENT'], B[B.event==e]['logENT']))
    MAN_sig.loc[e]['Lev_FC','p_FC'] = list(stats.levene(A[A.event==e]['logFC'], B[B.event==e]['logFC']))

print('\nSolar Noon')
print(MAN_means)
print('\n')
print(MAN_sig)

daytime_means_e = daytime_means
daytime_sig_e = daytime_sig
MAN_means_e = MAN_means
MAN_sig_e = MAN_sig

#%% Time of Day Comparison Stats [by BEACH]  - Variances

# Hour of day bin plots in other script


# Daytime/Nighttime by event (Daytime (1) = 6-19)
E = hf.groupby(['beach','daytime']).var()['logENT']
F = hf.groupby(['beach','daytime']).var()['logFC']
daytime_means = round(pd.merge(E,F, left_index=True, right_index=True),3)
count_daytime = hf.groupby(['beach','daytime']).size() # No. samples
count_daytime.name = 'N'
daytime_means = pd.merge(count_daytime,daytime_means, left_index=True, right_index=True)

daytime_means['ENT_exc_%'] = hf.groupby(['beach','daytime','ENT_exc']).size().xs(1, level=2, drop_level=True) / count_daytime
daytime_means['FC_exc_%'] = hf.groupby(['beach','daytime','FC_exc']).size().xs(1, level=2, drop_level=True) / count_daytime

# By beach
daytime_sig = pd.DataFrame(index=hf.beach.unique(),columns=['Lev_ENT','p_ENT', 'Lev_FC','p_FC'])
for e in hf.beach.unique():
    daytime_sig.loc[e]['Lev_ENT','p_ENT'] = list(stats.levene(day[day.beach==e]['logENT'], night[night.beach==e]['logENT']))
    daytime_sig.loc[e]['Lev_FC','p_FC'] = list(stats.levene(day[day.beach==e]['logFC'], night[night.beach==e]['logFC']))

print('BY BEACH: ')
print('Daytime/Nighttime')
print(daytime_means)
print('\n')
print(daytime_sig)


# Solar Noon
hf['solar_noon'] = [1 if x in sn.index else 0 for x in hf.index]  # See previous section for SN

E = hf[hf.daytime==1].groupby(['beach','solar_noon']).var()['logENT']
F = hf[hf.daytime==1].groupby(['beach','solar_noon']).var()['logFC']
MAN_means = round(pd.merge(E,F, left_index=True, right_index=True),3)

count_MAN = hf[hf.daytime==1].groupby(['beach','solar_noon']).size()  # No. samples
count_MAN.name = 'N'
MAN_means = pd.merge(count_MAN,MAN_means, left_index=True, right_index=True)

MAN_means['ENT_exc_%'] = hf[hf.daytime==1].groupby(['beach','solar_noon','ENT_exc']).size().xs(1, level=2, drop_level=True) / count_MAN
MAN_means['FC_exc_%'] = hf[hf.daytime==1].groupby(['beach','solar_noon','FC_exc']).size().xs(1, level=2, drop_level=True) / count_MAN


# Test if morning and afternoon means are different
A = hf[hf.daytime==1][hf.solar_noon == 1]
B = hf[hf.daytime==1][hf.solar_noon == 0]

# By beach
MAN_sig = pd.DataFrame(index=hf.beach.unique(),columns=['Lev_ENT','p_ENT', 'Lev_FC','p_FC'])
for e in hf.beach.unique():
    MAN_sig.loc[e]['Lev_ENT','p_ENT'] = list(stats.levene(A[A.beach==e]['logENT'], B[B.beach==e]['logENT']))
    MAN_sig.loc[e]['Lev_FC','p_FC'] = list(stats.levene(A[A.beach==e]['logFC'], B[B.beach==e]['logFC']))

print('\nSolar Noon')
print(MAN_means)
print('\n')
print(MAN_sig)


# SAVE
with pd.ExcelWriter(os.path.join(save_folder,'FIB_time_of_day_stats_variance.xlsx')) as writer:  
    daytime_means.to_excel(writer, sheet_name='daytime_var_beach')
    daytime_sig.to_excel(writer, sheet_name='daytime_sig_beach')
    MAN_means.to_excel(writer, sheet_name='solar_noon_var_beach')
    MAN_sig.to_excel(writer, sheet_name='solar_noon_sig_beach')
    
    daytime_means_e.to_excel(writer, sheet_name='daytime_var_event')
    daytime_sig_e.to_excel(writer, sheet_name='daytime_sig_event')
    MAN_means_e.to_excel(writer, sheet_name='solar_noon_var_event')
    MAN_sig_e.to_excel(writer, sheet_name='solar_noon_sig_event')


#%% Regression on hour of morning (all beaches)
# morn_ENT = hf.groupby('hour').mean()['logENT'].loc[6:11]  # All samples together
# morn_FC = hf.groupby('hour').mean()['logFC'].loc[6:11]

# By event first, then mean
morn_ENT = hf.groupby(['event','hour']).mean()['logENT'].reset_index().pivot(index='hour'
                                                                  ,columns='event',values='logENT').mean(axis=1).loc[6:11]
morn_ENT.name='logENT'

morn_FC = hf.groupby(['event','hour']).mean()['logFC'].reset_index().pivot(index='hour'
                                                                  ,columns='event',values='logFC').mean(axis=1).loc[6:11]
morn_FC.name = 'logFC'

morn_ENT = sm.add_constant(morn_ENT).reset_index().reset_index()  # Add constant, create 0 index column from 6am
morn_FC = sm.add_constant(morn_FC).reset_index().reset_index()

lm_ENT = sm.OLS(morn_ENT.logENT,morn_ENT[['index','const']]).fit()
ENT_slope = lm_ENT.params.loc['index']  # -.146 / - .146
ENT_R2 = lm_ENT.rsquared  # .81 /.77

lm_FC = sm.OLS(morn_FC.logFC,morn_FC[['index','const']]).fit()
FC_slope = lm_FC.params.loc['index']  # - .239 / -.202
FC_R2 = lm_FC.rsquared  # .92 /0.9

x = np.array([-1,0,1,2,3,4,5,6])
ENT_reg = lm_ENT.params.loc['const'] + lm_ENT.params.loc['index']*x
FC_reg = lm_FC.params.loc['const'] + lm_FC.params.loc['index']*x


# Plot
pal = [#'#cccccc',
       '#969696',
       '#525252']
pal = sns.color_palette(pal)
params = {
   'axes.labelsize': 11,
   'font.size': 12,
   'legend.fontsize': 12,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'font.family'  : 'sans-serif',
   'font.sans-serif':'Helvetica',
   'axes.axisbelow': True
   }
plt.rcParams.update(params)

ax = hf.groupby('hour').mean()[['logENT','logFC']].plot(color=pal, marker='.',ms=8,linewidth=1.5)  # mean FIB by hour bin

plt.plot(x+6,ENT_reg, ls='--', color=pal[0])  # Regression lines
plt.plot(x+6,FC_reg, ls='--', color=pal[1])

plt.axvline(6, ls='-.',color='k', alpha=.3)  # delineate avg. sunrise/sunset
plt.axvline(19, ls='-.',color='k', alpha=.3)

plt.legend(['ENT','FC'], loc='lower left', frameon=False)
plt.ylabel(r'log$_{10}$ CFU/100 ml')
plt.ylim([.9, 3.1])

plt.xlabel('Hours Since Sunrise')  # X-axis
plt.xlim([-0.25,23.25])
t=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
plt.xticks(ticks=t,
           labels = ['-6','','-4','','-2','','0','','2','','4','','6','','8','','10','','12','','14','','16',''])

#ax = plt.gca()
ax2 = ax.twiny()
#ax2 = ax.secondary_xaxis('top')
ax2.set_xlabel('Hour of Day')
ax2.set_xlim([-0.25,23.25])

ax2.set_xticks(t)
ax2.set_xticklabels(labels = [0,'',2,'',4,'',6,'',8,'',10,'',12,'',14,'',16,'',18,'',20,'',22,''])

plt.gcf().set_size_inches(8,5)
plt.tight_layout()

#%% Variability between Events
# Levene's test - two samples have the same variance
# Results - Different between beaches. LP events all different for FC and ENT.
# CB events the same for FC and ENT!


stats.levene(lp13['logENT'],lp16['logENT'])
stats.levene(lp13['logENT'],lp18['logENT'])
stats.levene(lp13['logFC'],lp18['logFC'])
stats.levene(lp16['logFC'],lp18['logFC'])
stats.levene(cb11['logFC'],cb11['logFC'])
stats.levene(cb11['logFC'],cb12['logFC'])
stats.levene(cb11['logENT'],cb12['logENT'])
stats.levene(cb['logENT'],hsb02['logENT'])
stats.levene(lpo['logENT'],hsb02['logENT'])
stats.levene(lp['logENT'],hsb02['logENT'])
stats.levene(lp['logENT'],cb['logENT'])
stats.levene(lp['logFC'],cb['logFC'])

#%% Binned Analysis - CV by Hour - by EVENT

#Bin by hour
hf_ent = hf.groupby(['event','hour']).mean()['logENT']
ENT_h = hf_ent.reset_index().pivot(index='hour',columns='event',values='logENT')
ent_std = hf.groupby(['event','hour']).std()['logENT']
ENT_std = ent_std.reset_index().pivot(index='hour',columns='event',values='logENT')
ent_n = hf.groupby(['event','hour']).count()['logENT']
ent_err = 1.96 * ent_std / (ent_n**0.5)  # 95% CI
ENT_err = ent_err.reset_index().pivot(index='hour',columns='event',values='logENT')

ENT_err.mean()  # Mean standard error by hour by event
ENT_err.mean().mean()  # Mean SE by hour overall

cv_h_ENT = ENT_std / ENT_h  # CV by hour
cv_h_ENT.mean().mean()  # mean CV of composites

hf_fc = hf.groupby(['event','hour']).mean()['logFC']
FC_h = hf_fc.reset_index().pivot(index='hour',columns='event',values='logFC')
fc_std = hf.groupby(['event','hour']).std()['logFC']
FC_std = fc_std.reset_index().pivot(index='hour',columns='event',values='logFC')
fc_n = hf.groupby(['event','hour']).count()['logFC']
fc_err = 1.96 * fc_std / (fc_n**0.5)  # 95% CI
FC_err = fc_err.reset_index().pivot(index='hour',columns='event',values='logFC')

FC_err.mean()  # Mean standard error by hour by event
FC_err.mean().mean()  # Mean SE by hour overall

cv_h_FC = FC_std / FC_h  # CV by hour
cv_h_FC.mean().mean()  # mean CV of composites

# Plot CV by hour
params = {
   'axes.labelsize': 12,
   'font.size': 12,
   'legend.fontsize': 11,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'font.family'  : 'sans-serif',
   'font.sans-serif':'Helvetica',
   'axes.axisbelow': True
   }
plt.rcParams.update(params)
plt.figure(figsize=(8,7))
plt.subplot(2,1,1)
ax = cv_h_ENT.plot(ax=plt.gca(), marker='.',ms=5,linewidth=1.5, alpha=.25)
# use colormap='Grays' or somethine else, maube - https://matplotlib.org/3.2.1/tutorials/colors/colormaps.html
markers = ['.','^','x','D','v','>']
for i, line in enumerate(ax.get_lines()):
    line.set_marker(markers[i])
    
cv_h_ENT.mean(axis=1).plot(ax=plt.gca(),color='k', marker='.',ms=8,linewidth=1.5, label='Combined') # MEAN CV by hour
    
plt.xlabel('')
plt.xlim([-0.25,23.25])
plt.ylim([0,0.55])
plt.ylabel(r'CV')
#plt.legend(loc="upper left", ncol=2, fontsize=10, frameon=False)
plt.legend('',frameon=False)
plt.text(0.94, 0.92, r'ENT', transform=ax.transAxes)
plt.xticks(ticks= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
           labels = [0,'',2,'',4,'',6,'',8,'',10,'',12,'',14,'',16,'',18,'',20,'',22,''])


plt.subplot(2,1,2)
ax = cv_h_FC.plot(ax=plt.gca(), marker='.',ms=5,linewidth=1.5, alpha=.25)
markers = ['.','^','x','D','v','>']
for i, line in enumerate(ax.get_lines()):
    line.set_marker(markers[i])
cv_h_FC.mean(axis=1).plot(ax=plt.gca(),color='k', marker='.',ms=8,linewidth=1.5)
plt.xlabel('Hour of Day')
plt.xlim([-0.25,23.25])
plt.ylim([0,0.55])
plt.ylabel(r'CV')
plt.legend('',frameon=False)
plt.text(0.95, 0.92, r'FC', transform=ax.transAxes)
plt.xticks(ticks= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
           labels = [0,'',2,'',4,'',6,'',8,'',10,'',12,'',14,'',16,'',18,'',20,'',22,''])

plt.tight_layout()


# Is there more variation during the day than at night?

# Overall difference in CV
ncv = night.groupby('event').std()['logENT'] / night.groupby('event').mean()['logENT']
dcv = day.groupby('event').std()['logENT'] / day.groupby('event').mean()['logENT']
stats.kruskal(ncv,dcv)  # significant at p = 0.1
ncv.mean()
dcv.mean()

# Analysis after biining by hour first
dtENT = cv_h_ENT.loc[6:19]
ntENT = cv_h_ENT.loc[0:5].append(cv_h_ENT.loc[20:]) 

print('ENT')
for c in dtENT.columns:
    print(c)
    kw = stats.kruskal(dtENT[c].dropna(), ntENT[c].dropna())
    print(kw)
    print('day/night: ' + str(round(dtENT[c].dropna().mean() / ntENT[c].dropna().mean(),3)) )
print('Combined')
print(stats.kruskal(dtENT.mean(axis=1), ntENT.mean(axis=1)))

dtFC = cv_h_FC.loc[6:19]
ntFC = cv_h_FC.loc[0:5].append(cv_h_FC.loc[20:]) 

print('\nFC')
for c in dtFC.columns:
    print(c)
    kw = stats.kruskal(dtFC[c].dropna(), ntFC[c].dropna())
    print(kw)
    print('day/night: ' + str(round(dtFC[c].dropna().mean() / ntFC[c].dropna().mean(),3)) )
print('Combined')
print(stats.kruskal(dtFC.mean(axis=1), ntFC.mean(axis=1)))

# Results: Combined event and beach CVh distribution is higher during daytime for FC and ENT 
# but CB-11, HSB-02, and LP-13 not more variable for ENT during the day
# and CB-11 and CB-12 not more variable for FC during the day 

#%% Binned Analysis - CV by Hour - by BEACH

#Bin by hour
hf_ent = hf.groupby(['beach','hour']).mean()['logENT']
ENT_h = hf_ent.reset_index().pivot(index='hour',columns='beach',values='logENT')
ent_std = hf.groupby(['beach','hour']).std()['logENT']
ENT_std = ent_std.reset_index().pivot(index='hour',columns='beach',values='logENT')
ent_n = hf.groupby(['beach','hour']).count()['logENT']
ent_err = 1.96 * ent_std / (ent_n**0.5)  # 95% CI
ENT_err = ent_err.reset_index().pivot(index='hour',columns='beach',values='logENT')

ENT_err.mean()  # Mean standard error by hour by event
ENT_err.mean().mean()  # Mean SE by hour overall

cv_h_ENT = ENT_std / ENT_h  # CV by hour
cv_h_ENT.mean().mean()  # mean CV of composites

hf_fc = hf.groupby(['beach','hour']).mean()['logFC']
FC_h = hf_fc.reset_index().pivot(index='hour',columns='beach',values='logFC')
fc_std = hf.groupby(['beach','hour']).std()['logFC']
FC_std = fc_std.reset_index().pivot(index='hour',columns='beach',values='logFC')
fc_n = hf.groupby(['beach','hour']).count()['logFC']
fc_err = 1.96 * fc_std / (fc_n**0.5)  # 95% CI
FC_err = fc_err.reset_index().pivot(index='hour',columns='beach',values='logFC')

FC_err.mean()  # Mean standard error by hour by event
FC_err.mean().mean()  # Mean SE by hour overall

cv_h_FC = FC_std / FC_h  # CV by hour
cv_h_FC.mean().mean()  # mean CV of composites

# Plot CV by hour
params = {
   'axes.labelsize': 12,
   'font.size': 12,
   'legend.fontsize': 11,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'font.family'  : 'sans-serif',
   'font.sans-serif':'Helvetica',
   'axes.axisbelow': True
   }
plt.rcParams.update(params)
plt.figure(figsize=(8,7))
plt.subplot(2,1,1)
ax = cv_h_ENT.plot(ax=plt.gca(), marker='.',ms=5,linewidth=1.5, alpha=.25)
# use colormap='Grays' or somethine else, maube - https://matplotlib.org/3.2.1/tutorials/colors/colormaps.html
markers = ['.','^','x']
for i, line in enumerate(ax.get_lines()):
    line.set_marker(markers[i])
    
cv_h_ENT.mean(axis=1).plot(ax=plt.gca(),color='k', marker='.',ms=8,linewidth=1.5, label='Combined') # MEAN CV by hour
    
plt.xlabel('')
plt.xlim([-0.25,23.25])
plt.ylim([0,0.65])
plt.ylabel(r'CV')
plt.legend(loc="upper left", ncol=1, fontsize=10, frameon=False)
#plt.legend('',frameon=False)
plt.text(0.94, 0.92, r'ENT', transform=ax.transAxes)
plt.xticks(ticks= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
           labels = [0,'',2,'',4,'',6,'',8,'',10,'',12,'',14,'',16,'',18,'',20,'',22,''])


plt.subplot(2,1,2)
ax = cv_h_FC.plot(ax=plt.gca(), marker='.',ms=5,linewidth=1.5, alpha=.25)
markers = ['.','^','x']
for i, line in enumerate(ax.get_lines()):
    line.set_marker(markers[i])
cv_h_FC.mean(axis=1).plot(ax=plt.gca(),color='k', marker='.',ms=8,linewidth=1.5)
plt.xlabel('Hour of Day')
plt.xlim([-0.25,23.25])
plt.ylim([0,0.65])
plt.ylabel(r'CV')
plt.legend('',frameon=False)
plt.text(0.95, 0.92, r'FC', transform=ax.transAxes)
plt.xticks(ticks= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
           labels = [0,'',2,'',4,'',6,'',8,'',10,'',12,'',14,'',16,'',18,'',20,'',22,''])

plt.tight_layout()


# Is there more variation during the day than at night?
# Analysis on the CV after binning by hour - MAYBE NOT!
dtENT = cv_h_ENT.loc[6:19]
ntENT = cv_h_ENT.loc[0:5].append(cv_h_ENT.loc[20:]) 

print('ENT')
for c in dtENT.columns:
    print(c)
    kw = stats.kruskal(dtENT[c], ntENT[c])
    print(kw)
print('Combined')
print(stats.kruskal(dtENT.mean(axis=1), ntENT.mean(axis=1)))

dtFC = cv_h_FC.loc[6:19]
ntFC = cv_h_FC.loc[0:5].append(cv_h_FC.loc[20:]) 

print('\nFC')
for c in dtFC.columns:
    print(c)
    kw = stats.kruskal(dtFC[c], ntFC[c])
    print(kw)
print('Combined')
print(stats.kruskal(dtFC.mean(axis=1), ntFC.mean(axis=1)))


# Analysis on the CV after binning by hour - MAYBE NOT!
dtENT = cv_h_ENT.loc[6:19]
ntENT = cv_h_ENT.loc[0:5].append(cv_h_ENT.loc[20:]) 

print('ENT')
for c in dtENT.columns:
    print(c)
    kw = stats.kruskal(dtENT[c], ntENT[c])
    print(kw)
print('Combined')
print(stats.kruskal(dtENT.mean(axis=1), ntENT.mean(axis=1)))

dtFC = cv_h_FC.loc[6:19]
ntFC = cv_h_FC.loc[0:5].append(cv_h_FC.loc[20:]) 

print('\nFC')
for c in dtFC.columns:
    print(c)
    kw = stats.kruskal(dtFC[c], ntFC[c])
    print(kw)
print('Combined')
print(stats.kruskal(dtFC.mean(axis=1), ntFC.mean(axis=1)))

#%% ACF Plots
events=['HSB-02','LP-13','CB-11','LP-16','CB-12','LP-18']
lag = pd.DataFrame(index=events + ['mean'])
hf['logEC']= hf['logFC']
plt.figure(figsize=(12,11))
c=1
for e in events:
    for f in ['ENT','EC']:
        plt.subplot(6,2,c)
        df = hf[hf.event==e]['log'+f]
        #plot_acf(df.dropna(), zero=False, ax=plt.gca(), marker='')
        rho = acf(df, nlags=20)
        #rho = pacf(df,nlags=20)  # Partial autocorrelation
        
        plt.stem(range(0,len(rho)), rho, linefmt='k-', markerfmt=' ', basefmt='k-')
        
        ax = plt.gca()
        
        if c in [1,2]:
            plt.title(f)
        else:
            plt.title('')
        
        if c in [1,3,5,7,9,11]:
            #plt.ylabel(r'$\rho_a$', rotation=0)
            plt.ylabel(e + ' ', rotation=0)
        #else:
            #plt.text(20.5, 0.75, e, horizontalalignment='right')
            
        plt.ylim(-.55,1.05)
        
        if c in [11,12]:
            plt.xlabel('Lag')
        
        #plt.xlim(.75,20.25)
        #plt.xticks(ticks=range(1,21), 
        #           labels =[1,'',3,'',5,'',7,'',9,'',11,'',13,'',15, '',17,'',19,''])
                            #21,'',23,''])
        
        plt.axhline(1.96/(len(df)**.5), ls='--', color='grey', alpha=0.7)
        plt.axhline(-1.96/(len(df)**.5), ls='--', color='grey', alpha=0.7)
        
        
        #ax.axes.get_lines()[0].set_color('k')  # color x axis black
        
        c+=1
        
    plt.tight_layout()
    #plt.subplots_adjust(top=0.08, bottom=0.07,left=0.049,right=0.982,hspace=0.4,wspace=0.079)
    
#%% Autocorrelation Stats [Old Plot]

# # Differenced series
# plt.figure(figsize=(12,8))
# c=1
# for e in hf.event.unique():
#     plt.subplot(3,2,c)
#     dif = hf[hf.event==e]['logENT'].diff()
#     plt.plot(dif,color='k')
#     plt.text(0.01, 0.87, e)  # Event label
#     c+=1
    
# plt.suptitle('Differenced Time Series')

# (partial) autocorrelation by event

events=['HSB-02','LP-13','CB-11','LP-16','CB-12','LP-18']
lag = pd.DataFrame(index=events + ['mean'])
for f in ['ENT','FC']:
    df_lag = pd.DataFrame(index = events, columns=[[f,f,f,f,f],['interval','N_siglags','sig_interval',
                                                            'rho_first_order','rho_second_order']])
    plt.figure(figsize=(10,6))
    c=1
    for e in events:#hf.event.unique():
        plt.subplot(3,2,c)
        df = hf[hf.event==e]['log'+f]
        #plot_acf(df.dropna(), zero=False, ax=plt.gca(), marker='')
        #rho = acf(df, nlags=40)
        rho = pacf(df, nlags=40)  # Partial autocorrelation, Unbiased Yule-Walker
        
        plt.stem(range(0,len(rho)), rho, linefmt='k-', markerfmt=' ', basefmt='k-')
        
        ax = plt.gca()
        plt.title('')
        
        if c in [1,3,5]:
            plt.ylabel(r'$\rho_a$', rotation=0)
            
        plt.ylim(-.55,1.05)
    
        
        if c in [5,6]:
            plt.xlabel('Lag')
        
        plt.xlim(.75,20.25)
        plt.xticks(ticks=range(1,21), 
                  labels =[1,'',3,'',5,'',7,'',9,'',11,'',13,'',15, '',17,'',19,''])
                          
        
        plt.axhline(1.96/(len(df)**.5), ls='--', color='grey', alpha=0.7)
        plt.axhline(-1.96/(len(df)**.5), ls='--', color='grey', alpha=0.7)
        
        plt.text(41, 0.85, e, horizontalalignment='right')
        #ax.axes.get_lines()[0].set_color('k')  # color x axis black
        
        # Number of significant lags
        l = list((rho > 1.96/len(rho)**.5).astype(int)).index(0)
        interv = abs(df.index[0].minute - df.index[1].minute)
        df_lag.loc[e][f]['N_siglags'] = l
        df_lag.loc[e][f]['interval'] = interv
        df_lag.loc[e][f]['sig_interval'] = l*interv
        df_lag.loc[e][f]['rho_first_order'] = rho[1]
        df_lag.loc[e][f]['rho_second_order'] = rho[2]
        
        c+=1
        
    #plt.tight_layout()
    ml = df_lag[df_lag[f]['N_siglags'] != 0].mean()
    ml.name = 'mean'
    df_lag = df_lag.append(ml)
    lag = pd.concat((lag,df_lag), axis=1)  # Table of significant lag data

print(lag.loc['mean'])

#%% Moving averages (window: hourly, half-hourly) - OLD
# Test on LP-13 first

# Backward moving average (to see what collecting composites would do to the management error rate)
lp13['ENT_MA2'] = lp13['logENT'].rolling(window=2,center=False).mean().shift(-1)
lp13['ENT_MA2_err'] = 1.96*lp13['logENT'].rolling(window=2,center=False).std().shift(-1)/(2**.5)
lp13['ENT_MA3'] = lp13['logENT'].rolling(window=3,center=False).mean().shift(-2)
lp13['ENT_MA3_err'] = 1.96*lp13['logENT'].rolling(window=3,center=False).std().shift(-2)/(3**.5)
lp13['ENT_MA6'] = lp13['logENT'].rolling(window=6,center=False).mean().shift(-5)
lp13['ENT_MA6_err'] = 1.96*lp13['logENT'].rolling(window=6,center=False).std().shift(-5)/(6**.5)

# How many sample pairs would result in a different management decision?
# ENT only. Need to alter for both FIB
man_err = pd.DataFrame(index=hf.event.unique(), 
                       columns=['manage_error_dif1','manage_error_dif2','manage_error_dif3','manage_error_dif6'])
for e in hf.event.unique():
    df_fib = hf[hf.event==e][['ENT','ENT_exc']]
    df_fib['ENT_exc_1']=df_fib['ENT_exc'].shift(1)
    df_fib['ENT_exc_2']=df_fib['ENT_exc'].shift(2)
    df_fib['ENT_exc_3']=df_fib['ENT_exc'].shift(3)
    df_fib['ENT_exc_6']=df_fib['ENT_exc'].shift(6)
    
    man_err.loc[e]['manage_error_dif1'] = (abs(df_fib.ENT_exc - df_fib.ENT_exc_1)).sum()
    man_err.loc[e]['manage_error_dif2'] = (abs(df_fib.ENT_exc - df_fib.ENT_exc_2)).sum()
    man_err.loc[e]['manage_error_dif3'] = (abs(df_fib.ENT_exc - df_fib.ENT_exc_3)).sum()
    man_err.loc[e]['manage_error_dif6'] = (abs(df_fib.ENT_exc - df_fib.ENT_exc_6)).sum()

# MA PLOTS 
# ax1 = lp13[['logENT','ENT_MA2','ENT_MA3','ENT_MA6']].plot(marker='.')
# ax1.axhline(y=np.log10(104))

# ax2 = lp13[lp13.index.minute==0]['logENT'].plot(marker='.')  # Test effect on a single sample minute
# lp13[lp13.index.minute==0]['ENT_MA6'].plot(ax=ax2, marker='.', color = 'k', yerr=lp13[lp13.index.minute==0]['ENT_MA6_err'])
# ax2.axhline(y=np.log10(104), color='r')

# Grab Sample vs. MA6 (with CI)
ax3 = lp13['logENT'].plot(marker='.')
#lp13['ENT_MA6'].plot(ax=ax3, yerr=lp13['ENT_MA6_err'])
lp13['ENT_MA6'].plot(ax=ax3, color='k')
(lp13['ENT_MA6'] + lp13['ENT_MA6_err']).plot(ax=ax3, ls='--',color='k')
(lp13['ENT_MA6'] - lp13['ENT_MA6_err']).plot(ax=ax3, ls='--',color='k')
ax3.axhline(y=np.log10(104),color='r')
plt.sca(ax3)
plt.legend(['Grab Samples','Backward Moving Average (6)', '95% CI'], loc="upper left", frameon=False)
plt.ylabel(r'log$_{10}$ CFU/100 ml')
plt.title('LP - 2013 Event (10 min interval) - ENT')


#%% Normalized FIB - OLD
# hf_norm = pd.DataFrame()
# for e in hf.event.unique():
#     print(e)
#     df_fib = hf[hf.event==e][['logTC','logFC','logENT']]
#     mean_conc = df_fib.mean()
#     max_conc = df_fib.max()
#     min_conc = df_fib.min()
#     conc_norm = (df_fib - min_conc)/(max_conc - min_conc)
#     #conc_norm = (df_fib - mean_conc)/(mean_conc)
    
#     conc_norm['beach'] = e
#     hf_norm = hf_norm.append(conc_norm)
    
# hf_norm['hour'] = hf_norm.index.hour
    
# hf_ent = hf_norm.groupby(['beach','hour']).mean()['logENT']
# ENT_h = hf_ent.reset_index().pivot(index='hour',columns='beach',values='logENT')
# ent_std = hf.groupby(['beach','hour']).std()['logENT']
# ent_n = hf.groupby(['beach','hour']).count()['logENT']
# ent_err = 1.96 * ent_std / (ent_n**0.5)  # 95% CI
# ent_err = ent_err.reset_index().pivot(index='hour',columns='beach',values='logENT')

# hf_fc = hf_norm.groupby(['beach','hour']).mean()['logFC']
# FC_h = hf_fc.reset_index().pivot(index='hour',columns='beach',values='logFC')
# fc_std = hf.groupby(['beach','hour']).std()['logFC']
# fc_n = hf.groupby(['beach','hour']).count()['logFC']
# fc_err = 1.96 * fc_std / (fc_n**0.5)  # 95% CI
# fc_err = fc_err.reset_index().pivot(index='hour',columns='beach',values='logFC')

# plt.figure(figsize=(12,10.5))
# plt.subplot(2,1,1)
# ax = ENT_h.plot(ax=plt.gca(), marker='.',ms=8,linewidth=1.5, yerr=ent_err)
# plt.xlabel('')
# plt.xlim([-0.25,23.25])
# plt.ylim([-0.2,1.2])
# plt.ylabel(r'log$_{10}$ CFU/100 ml')
# # plt.legend(loc="topright", ncol=1, fontsize=10, title='High Frequency', frameon=False)
# plt.legend('',frameon=False)
# plt.text(0.01, 0.92, r'log$_{10}$ENT', transform=ax.transAxes)
# plt.xticks(ticks= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
#            labels = [0,'',2,'',4,'',6,'',8,'',10,'',12,'',14,'',16,'',18,'',20,'',22,''])
# ax.tick_params(axis='x', which='minor', bottom=True)

# plt.subplot(2,1,2)
# ax2 = FC_h.plot(ax=plt.gca(), marker='.',ms=8,linewidth=1.5,yerr=fc_err)
# # plt.legend('',frameon=False)
# plt.legend(loc="bottomleft", ncol=1, fontsize=10, title='Beach (All HF Events)', frameon=False)
# plt.xlabel('Hour of Day')
# plt.xlim([-0.25,23.25])
# plt.ylim([-0.2,1.2])
# plt.ylabel(r'log$_{10}$ CFU/100 ml')
# plt.text(0.01, 0.92, r'log$_{10}$FC', transform=ax2.transAxes)
# plt.xticks(ticks= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
#            labels = [0,'',2,'',4,'',6,'',8,'',10,'',12,'',14,'',16,'',18,'',20,'',22,''])
# plt.tight_layout()
# plt.subplots_adjust(bottom=0.1, hspace=.13)



