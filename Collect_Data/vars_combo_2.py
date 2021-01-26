#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vars_combo_2.py
Created on Wed Jan  8 19:37:57 2020

@author: rtsearcy
"""

import pandas as pd
import os
from numpy import sin, cos, pi, isnan, nan

folder = '/Users/rtsearcy/Box/water_quality_modeling/thfs/traditional_nowcast/variables'
#folder = '/Users/rtsearcy/Box/water_quality_modeling/thfs/hf/variables/2018'
fib_file = 'Lovers_Point_variables_fib_2000_2020.csv'

sd = '2008-01-01'
ed = '2016-03-01'
ang = 45 # Lover's Point

#out_file = 'LP_trad_modeling_dataset_' + sd.replace('-','') + '_' + ed.replace('-','') + '.csv'
out_file = 'LP_trad_modeling_dataset_' + sd[0:4] + '_' + ed[0:4] + '.csv'

var_files = [f for f in os.listdir(folder) if f != fib_file and '.csv' in f]
df = pd.read_csv(os.path.join(folder,fib_file))
assert 'dt' in df.columns, 'No datetime column named \'dt\' found'

#if 'sample_time' in df.columns:
#    df['dt'] = pd.to_datetime(df['dt'] + ' ' + df['sample_time'], format='%Y-%m-%d %H:%M')
#else:
#    df['dt'] = pd.to_datetime(df['dt'])
df['dt'] = pd.to_datetime(df['dt'])
df.set_index('dt', inplace=True)
# Sort data into ascending time index (Earliest sample first)
df.sort_index(ascending=True, inplace=True) 
df['date_temp'] = [d.date() for d in df.index]  # for joining daily vars
#%%
for f in var_files:
    #print(f)
    df_var = pd.read_csv(os.path.join(folder,f))
    if 'date' in df_var.columns:
        df_var['dt'] = df_var['date']
        df_var.drop(['date'],axis=1, inplace=True)
    elif 'dt' not in df_var.columns:
        print('No \'dt\' column in ' + f)
        continue
    df_var['dt'] = pd.to_datetime(df_var['dt'])
    df_var.set_index('dt', inplace=True)
    df_var.sort_index(ascending=True, inplace=True)
    
    if sum([t.hour for t in df_var.index]) == 0:  # Daily vars (no hour associated)
        df_var['date_temp'] = [d.date() for d in df_var.index]
        df = df.reset_index().merge(df_var, how = 'left', on='date_temp').set_index('dt')
    else:
        df = df.merge(df_var, how = 'left', left_index=True, right_index=True)
    print(f + ' merged...')

# Direction Variables
if 'wspd' in df.columns and 'wdir' in df.columns:  # Instantaneous wind speed/direction
    df['awind'] = df['wspd'] * round(sin(((df['wdir'] - ang) / 180) * pi),1)
    df['owind'] = df['wspd'] * round(cos(((df['wdir'] - ang) / 180) * pi),1)
    
if 'wspd1' in df.columns and 'wdir1' in df.columns:  # Wind speed/direction
    df['awind1'] = df['wspd1'] * round(sin(((df['wdir1'] - ang) / 180) * pi),1)
    df['owind1'] = df['wspd1'] * round(cos(((df['wdir1'] - ang) / 180) * pi),1)

if 'wspd_L1' in df.columns and 'wdir_L1' in df.columns:  # Local wind speed/direction
    df['awind_L1'] = df['wspd_L1'] * round(sin(((df['wdir_L1'] - ang) / 180) * pi),1)
    df['owind_L1'] = df['wspd_L1'] * round(cos(((df['wdir_L1'] - ang) / 180) * pi),1)

for w in ['MWD1', 'MWD1_max', 'MWD1_min']:  # Wave direction
    if w in df.columns:
        df['MWD1_b' + w.replace('MWD1', '')] = df[w] - ang  # Direction relative to beach
        df['SIN_MWD1_b' + w.replace('MWD1', '')] = \
            round(sin(((df['MWD1_b' + w.replace('MWD1', '')]) / 180) * pi), 3)
        df.drop([w], axis=1, inplace=True)

df = df[sd:ed]
df.drop('date_temp',axis=1,inplace=True)
df_out = df
df_out.to_csv(os.path.join(folder,out_file))
