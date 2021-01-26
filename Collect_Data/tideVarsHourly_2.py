# tideVarsHourly.py - Computes hourly tidal variables from raw CO-OPS 6 min water level data for beaches with
# sample times available
# RS - 4/13/2018
# Updated: 4/2/2020 - adjusted for THFS project

# Raw data source/description: https://tidesandcurrents.noaa.gov/tide_predictions.html

# NOTE: raw csv files should have timestamps in LST

# Tide Variables: tide, dtide_1, dtide_3, tide_stage

import pandas as pd
import os
import sys
from datetime import datetime

# Import raw data csv to pd DataFrame
tide_path = '/Users/rtsearcy/Box/water_quality_modeling/data/tide/observations/raw'
tide_file = os.path.join(tide_path, 'Monterey_water_level_20080101_20200301.csv')

#fib_folder = '/Users/rtsearcy/Box/water_quality_modeling/thfs/hf/modeling_datasets/'
#fib_file = 'LP_hf_FIB_data_20180421.csv'

fib_folder = '/Users/rtsearcy/Box/water_quality_modeling/thfs/traditional_nowcast/variables/'
fib_file = 'Lovers_Point_variables_fib_2000_2020.csv'

sd = '20080101'  # Start date (account for previous day, conservatuve)
ed = '20200301'  # End date

df_raw = pd.read_csv(tide_file)
df_raw.columns = ['date', 'tide']
df_raw['date'] = pd.to_datetime(df_raw['date'])
df_raw.set_index('date', inplace=True)
df_raw = df_raw[sd:ed]  # Only samples in time range (for speed)
df_fib = pd.read_csv(os.path.join(fib_folder,fib_file))

#if 'sample_time' in df_fib.columns:
#    df_fib['dt'] = pd.to_datetime(df_fib['dt'] + ' ' + df_fib['sample_time'], format='%Y-%m-%d %H:%M')
#else:
#    df_fib['dt'] = pd.to_datetime(df_fib['dt'])
df_fib['dt'] = pd.to_datetime(df_fib['dt'])
df_fib.set_index('dt', inplace=True)
df_fib = df_fib[sd:ed]

#%%
#  Convert Date Time to timestamp, set as index
#df = pd.DataFrame(index=df_raw.resample('D').mean().index)  # Preset index to days
#df = pd.DataFrame(index=df_fib.index)
df = pd.DataFrame()

for i in df_fib.index:
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
        
        tide = df_raw[(df_raw.index.year == year) &
                     (df_raw.index.month == month) & 
                     (df_raw.index.day == day) & 
                     (df_raw.index.hour == hour) & 
                     (df_raw.index.minute == minute)]  # Water level at sample time
        
        
        dtide_1 = float(tide.values) - float(df_raw.loc[tide.index.shift(-1, freq='H')].values)
        dtide_3 = float(tide.values) - float(df_raw.loc[tide.index.shift(-3, freq='H')].values) 
        tide = float(tide.values)
    
        obs = pd.DataFrame(index = [i])
        obs['tide'] = tide
        obs['dtide_1'] = dtide_1
        obs['dtide_2'] = dtide_3
        
        df = df.append(obs)
        
    except Exception as exc:
        print('   There was a problem: %s' % exc)
        obs = pd.DataFrame(index = [i])
        df = df.append(obs)
        continue
            
# Save to file
of_name = fib_file.replace('.csv','_tide_hour.csv')
outfile = os.path.join(fib_folder, of_name)
df.index.rename('date', inplace=True)
df.to_csv(outfile)
