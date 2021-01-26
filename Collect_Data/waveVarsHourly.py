# waveVarsHourly.py 
# Updated: 4/2/2020 - adjusted for THFS project
# Takes  30-min CDIP wave data and pairs it to samples with times


import pandas as pd
import os
from datetime import datetime

# Import raw data csv to pd DataFrame
wave_path = '/Users/rtsearcy/Box/water_quality_modeling/data/waves/raw'
wave_file = os.path.join(wave_path, 'Cabrillo_Point_Nearshore_raw_wave_data_20080101_20191031.csv')

fib_folder = '/Users/rtsearcy/Box/water_quality_modeling/thfs/traditional_nowcast/variables'
fib_file = 'Lovers_Point_variables_fib_2000_2020.csv'

#fib_folder = '/Users/rtsearcy/Box/water_quality_modeling/thfs/traditional_nowcast/modeling_datasets/'
#fib_file = 'LP_trad_modeling_dataset_20000101_20200301.csv'

sd = '20080101'  # Start date (account for previous day, conservatuve)
ed = '20200301'  # End date

df_raw = pd.read_csv(wave_file)

df_raw['dt'] = pd.to_datetime(df_raw['dt'])
df_raw.set_index('dt', inplace=True)
df_raw = df_raw[sd:ed]  # Only samples in time range (for speed)

df_fib = pd.read_csv(os.path.join(fib_folder,fib_file))
#if 'sample_time' in df_fib.columns:
#    df_fib['dt'] = pd.to_datetime(df_fib['dt'] + ' ' + df_fib['sample_time'], format='%Y-%m-%d %H:%M')
#else:
#    df_fib['dt'] = pd.to_datetime(df_fib['dt'])
df_fib['dt'] = pd.to_datetime(df_fib['dt'])
df_fib.set_index('dt', inplace=True)

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
        
        wi = df_raw.index.get_loc(i, method='nearest')  # nearest index
        
#        wave_day = df_raw[(df_raw.index.year == year) &
#                     (df_raw.index.month == month) & 
#                     (df_raw.index.day == day)] 
#                     (df_raw.index.hour == hour) & 
#                     (df_raw.index.minute == minute)]  # waves at sample time
        
        
#        dtide_1 = float(tide.values) - float(df_raw.loc[tide.index.shift(-1, freq='H')].values)
#        dtide_3 = float(tide.values) - float(df_raw.loc[tide.index.shift(-3, freq='H')].values) 
#        tide = float(tide.values)
    
        obs = pd.DataFrame(index = [i])
#        obs['tide'] = tide
#        obs['dtide_1'] = dtide_1
#        obs['dtide_2'] = dtide_3
        
        obs = df_raw.iloc[wi].to_frame().T
        obs.index = [i]
        df = df.append(obs)
        
    except Exception as exc:
        print('   There was a problem: %s' % exc)
        obs = pd.DataFrame(index = [i])
        df = df.append(obs)
        continue
            
# Save to file
of_name = fib_file.replace('.csv','_wave_hour.csv')
outfile = os.path.join(fib_folder, of_name)
df.index.rename('dt', inplace=True)
df.to_csv(outfile)
