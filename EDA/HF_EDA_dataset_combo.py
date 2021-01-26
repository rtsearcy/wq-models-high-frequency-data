#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 11:30:12 2020

@author: rtsearcy

Description: Aggregate FIB and environmental data into a single dataframe

"""

import wq_modeling as wqm
import os 
import pandas as pd
import numpy as np
from numpy import sin, cos, pi, isnan, nan
from scipy import stats
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 100)

# Trad files
trad_file = '/Users/rtsearcy/Box/water_quality_modeling/thfs/traditional_nowcast/modeling_datasets/LP_trad_modeling_dataset_2008_2016_EDA.csv'
trad = wqm.load_data(trad_file)
trad['event'] = 'trad'

# Impute missing waves
#wave = pd.read_csv('/Users/rtsearcy/Box/water_quality_modeling/data/waves/raw/Monterey_Bay_Outer_raw_wave_data_20080101_20191031.csv', parse_dates=['dt'], index_col=['dt'])
wave = pd.read_csv('/Users/rtsearcy/Box/water_quality_modeling/data/waves/raw/San_Pedro_raw_wave_data_19980101_20200331.csv', parse_dates=['dt'], index_col=['dt'])
# Monterey Outer for 2008 missing data
for i in trad.index:
    for v in ['WVHT','APD','DPD','MWD','Wtemp_B']:
        if np.isnan(trad[v].loc[i]):
            wi = wave.index.get_loc(i, method='nearest')
            trad[v].loc[i] = wave[v].iloc[wi]
            
#pd.merge_asof(df,wave, left_index=True, right_index=True, direction='nearest')
           
trad = wqm.clean(trad, percent_to_drop=0.50, save_vars=[])

#Load FIB samples from agency samples
trad_sample = wqm.load_data('/Users/rtsearcy/Box/water_quality_modeling/data/fib/agency/Lovers_Point_variables_fib.csv')
trad_sample['event'] = 'trad_samples'
# THESE ARE JUST FIB DATA FROM 2000-2020

# HF files
hf_folder = '/Users/rtsearcy/Box/water_quality_modeling/thfs/hf/modeling_datasets'
hf13 = wqm.load_data(os.path.join(hf_folder, 'LP_hf_modeling_dataset_20130420.csv'))
hf16 = wqm.load_data(os.path.join(hf_folder, 'LP_hf_modeling_dataset_20160409.csv'))
hf18 = wqm.load_data(os.path.join(hf_folder, 'LP_hf_modeling_dataset_20180421.csv'))

hf13['event'] = 'HF13'
hf16['event'] = 'HF16'
hf18['event'] = 'HF18'

#df = pd.concat([hf13,hf16,hf18,trad,trad_sample])  # Combine datasets
df = trad
df['beach']='HSB'
df['date_temp'] = [d.date() for d in df.index]

#%% FIB vars
for f in ['TC','FC','ENT']:
    thresh = wqm.fib_thresh(f)
    df['log'+ f]=np.log10(df[f])
    df[f+'_exc'] = [1 if i > thresh else 0 for i in df[f]]  # Exceedance binary
    df[f+'_cat'] = 0
    df[f+'_cat'][(df[f] > 10) & (df[f] <=0.75*thresh)] = 1
    df[f+'_cat'][(df[f] > .75*thresh) & (df[f] <=1.25*thresh)] = 2
    df[f+'_cat'][(df[f] > 1.25*thresh)] = 3
    # FIB level categories (0 - Low(ABLOQ), 1 -Medium, 2 high, 3- very high)
    
    # Shift
    df['log'+ f + '1']=np.log10(df[f+'1'])
    df[f+'1_exc'] = [1 if i > thresh else 0 for i in df[f+'1']]  # Exceedance binary
    df[f+'1_cat'] = 0
    df[f+'1_cat'][(df[f + '1'] > 10) & (df[f+'1'] <=0.75*thresh)] = 1
    df[f+'1_cat'][(df[f + '1'] > .75*thresh) & (df[f+'1'] <=1.25*thresh)] = 2
    df[f+'1_cat'][(df[f + '1'] > 1.25*thresh)] = 3


#%% Time of Day    
df['hour'] = df.index.hour    # Hour of day (PST)
df['month'] = df.index.month
df['dayofyear'] = df.index.dayofyear  # Julian day of year
df['year']=df.index.year
df['month']=df.index.month

df['weekend1'] = [1 if i in [0,5,6] else 0 for i in df.index.dayofweek]

df['daytime'] = 0   # Day/Night binary variable  (day = 6 PST < time < 19 PST )
df['daytime'][(df.index.hour >=6) & (df.index.hour < 19)] = 1

df['morn_aft_night'] = 0  # Sample taken during morning, midday, or night?
df['morn_aft_night'][(df.index.hour >=6) & (df.index.hour < 12)] = 1  # Morning
df['morn_aft_night'][(df.index.hour >=12) & (df.index.hour < 19)] = 2 # Afternoon

#%% Tide variables

# Days since full moon, spring/neap binary and continuous
moon_path = '/Users/rtsearcy/Box/water_quality_modeling/data/tide/full_moon.csv'
moon = pd.read_csv(moon_path,parse_dates=['dt'], index_col=['dt'])
moon['date_temp'] = [d.date() for d in moon.index]
df = df.reset_index().merge(moon, how = 'left', on='date_temp').set_index('dt')

# 6-min data
#tide_path = '/Users/rtsearcy/Box/water_quality_modeling/data/tide/observations/raw/Monterey_water_level_20000101_20200301.csv'
tide_path = '/Users/rtsearcy/Box/water_quality_modeling/data/tide/observations/raw/Long_Beach_water_level_19980101_20200301.csv'
tide = pd.read_csv(tide_path)
tide.columns = ['dt','tide']
tide['dt'] = pd.to_datetime(tide['dt'])
tide.set_index('dt',inplace=True)

# Daily highs/lows
#thl_path = '/Users/rtsearcy/Box/water_quality_modeling/data/tide/observations/raw/Monterey_high_low_20000101_20200301.csv'
thl_path = '/Users/rtsearcy/Box/water_quality_modeling/data/tide/observations/raw/Long_Beach_high_low_19980101_20200301.csv'
thl = pd.read_csv(thl_path)
thl.columns = ['dt','hl']
thl['dt'] = pd.to_datetime(thl['dt'])
thl.set_index('dt',inplace=True)

thl['tide_stage'] = 'H'  # Set high/low tides
thl['tide_stage'][thl['hl'] < thl['hl'].shift(1)] = 'L'

tide = tide.merge(thl, how='left', left_index=True, right_index=True)
tide.drop('hl', axis=1, inplace=True)

#Tide stage -  high (1), ebb/flood(0), low (-1?
tide = tide.fillna(method='ffill',limit=10).fillna(method='bfill',limit=10)  # High/low tide +/- 1 hour from point of high/low
tide['tide_stage'].fillna(value='M', inplace=True)
tide['tide_stage'] = tide['tide_stage'].map({'L': -1, 'M': 0, 'H': 1})
tm = tide['tide'].mean()
tide['tide_gtm'] = [1 if i > tm else 0 for i in tide['tide']]  # Tide greater than the mean? Mean(2008-2020) = 0.895

#%% More tide
# Append to df
df_tide = pd.DataFrame()
for i in df.index:
    try: 
        print(i)
        year = i.year
        month = i.month #'0'*(2-len(str(i.month))) + str(i.month)
        day = i.day
        hour = i.hour
        minute = i.minute
        
        if minute % 6 > 3:
            minute = minute + (6 - (minute % 6))
            if minute == 60:
                hour += 1
                minute = 0
        else:
            minute = minute - minute % 6
        
        t = tide[(tide.index.year == year) &
                     (tide.index.month == month) & 
                     (tide.index.day == day) & 
                     (tide.index.hour == hour) & 
                     (tide.index.minute == minute)]  # Water level at sample time
        
        
        dtide_1 = float(t['tide'].values) - float(tide['tide'].loc[t.index.shift(-1, freq='H')].values)
        dtide_3 = float(t['tide'].values) - float(tide['tide'].loc[t.index.shift(-3, freq='H')].values) 
        #t = float(t.values)
    
        # obs = pd.DataFrame(index = [i])
        # obs['tide'] = t
        obs = t
        obs['dtide_1'] = dtide_1
        obs['dtide_2'] = dtide_3
        
        df_tide = df_tide.append(obs)
        
    except Exception as exc:
        print('   There was a problem: %s' % exc)
        obs = pd.DataFrame(index = [i])
        df_tide = df_tide.append(obs)
        continue
df = df.merge(df_tide, how='left', left_index=True, right_index=True)
    
#%% Met vars
met_file = '/Users/rtsearcy/Box/water_quality_modeling/data/met/CIMIS/Santa_Cruz_CIMIS_day_all_met_20080101_20200331.csv'
met_file = '/Users/rtsearcy/Box/water_quality_modeling/data/met/NCDC/John_Wayne_hourly_data_19981230_20200331.csv'
#met_file = '/Users/rtsearcy/Box/water_quality_modeling/data/met/NCDC/John_Wayne_Met_Variables_19981230_20200331.csv'
met = pd.read_csv(met_file, parse_dates=['dt'], index_col=['dt'])
# met = met[['lograin3T','lograin7T']]
# met['date_temp'] = [d.date() for d in met.index]
# met['wet3'] = [1 if i > np.log10(2.54) else 0 for i in met['lograin3T']] # Signiificant rain (>2.54 mm) in past 3 days?
# met['wet7'] = [1 if i > np.log10(2.54) else 0 for i in met['lograin7T']] # 7 days?
# df = df.reset_index().merge(met, how = 'left', on='date_temp').set_index('dt')
df = pd.merge_asof(df,met, left_index=True, right_index=True, direction='nearest')

# Wind
angle = 225 #cowell = 135, lp = 45, HSB-225
df['awind'] = df['wspd'] * round(np.sin(((df['wdir'] - angle) / 180) * np.pi),1)
df['owind'] = df['wspd'] * round(np.cos(((df['wdir'] - angle) / 180) * np.pi),1)

# Wind directior category
df['wdir_cat'] = 0  # primarily  onshore (- 45 < wdir-angle < + 45)
df['wdir_cat'][(df.wdir - angle > 180-45) & (df.wdir - angle <= 180+45)]  = 1  # primarily offshore
df['wdir_cat'][(df.wdir - angle > 90-45) & (df.wdir - angle <= 90+45)] = 2  # primarily alongshore (upbeach, northward)
df['wdir_cat'][(df.wdir - angle > 270-45) & (df.wdir - angle <= 270+45)] = 3  # primarily alongshore (downbeach, southward)

#%% Upwelling Index
uw_file = '/Users/rtsearcy/Box/water_quality_modeling/data/upwelling/CUTI_daily.csv'
uw = pd.read_csv(uw_file, parse_dates = ['date'], index_col = ['date'])
uw['date_temp'] = [d.date() for d in uw.index]
uw = uw[['date_temp','33N']]
uw.columns = ['date_temp', 'upwelling']

df = df.reset_index().merge(uw, how = 'left', on='date_temp').set_index('dt')

df.drop('date_temp', axis=1, inplace=True)