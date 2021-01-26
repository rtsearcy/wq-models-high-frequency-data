# coopsVarsDaily.py - Computes daily local variables from raw NOAA CO-OPs data
# RS - 1/18/2017
# RTS - 3/16/2018 UPDATE

# NOTE: Use getRawCOOPS.py to get raw data. 6 min raw files should have timestamps in LST.
# Note: Likely don't need to calculate same day variables...none ever showed up in models in 2017

# Variables included:
# temp_L1, Wtemp_L1, wspd_L1, wdir_L1, pres_L1  (previous day means)
# temp_L1_max, Wtemp_L1_max, wspd_L1_max, wdir_L1_max, pres_L1_max (previous day max)
# temp_L1_min, Wtemp_L1_min, wspd_L1_min, wdir_L1_min, pres_L1_min (previous day min)

import pandas as pd
import re
import os

# Inputs
outfolder = '/Users/rtsearcy/Box/water_quality_modeling/data/coops'
infolder = os.path.join(outfolder, 'raw')

sd = '20080101'  # Start date (account for previous day, conservative)
ed = '20191031'  # End date

# TO PROCESS SINGLE FILE
# (change indent)
# file = 'NorthSplit_Tidal_Data_20080101_20201231.csv'
# infile = os.path.join(infolder, file)

# TO ITERATE THROUGH ALL RAW FILES
print('NOAA CO-OPS\nRaw Directory: ' + infolder + '\nVariable Directory: ' + outfolder)
for file in os.listdir(infolder):
    if not file.endswith('.csv'):
        continue
    infile = os.path.join(infolder, file)
    station = re.sub('_\d.+', '', file).replace('_', ' ')  # get station name from filename in raw folder
    print('\nProcessing NOAA CO-OPS data for ' + station)

    df_raw = pd.read_csv(infile)
    params_dict = {
        'temp_L': 'air_temperature',
        'Wtemp_L': 'water_temperature',
        'wspd_L': 'wspd',
        'wdir_L': 'wdir',
        'pres_L': 'air_pressure'
    }
    params = ['temp_L', 'Wtemp_L', 'wspd_L', 'wdir_L', 'pres_L']
    params = [p for p in params if params_dict[p] in df_raw.columns]  # Some stations do not have all params
    df_raw.columns = ['dt'] + params
    df_raw['dt'] = pd.to_datetime(df_raw['dt'], infer_datetime_format=True)
    df_raw = df_raw.drop_duplicates(subset='dt')
    df_raw = df_raw.set_index('dt')

    df_raw = df_raw[sd:ed]  # Only samples in time range (for speed)
    df_vars = pd.DataFrame(index=df_raw.resample('D').mean().index)  # Preset index to days
    for p in params:
        df_vars[p + '1'] = round(df_raw[p].resample('D').mean().shift(1, freq='D'), 1)  # previous day mean
        df_vars[p + '1_max'] = df_raw[p].resample('D').max().shift(1, freq='D')  # previous day max
        df_vars[p + '1_min'] = df_raw[p].resample('D').min().shift(1, freq='D')  # previous day min

    # Export vars to spreadsheet
    #outname = station.replace(' ', '_') + '_COOPS_Variables_' + sd + '_' + ed + '.csv'
    outname = file.replace('.csv','_variables.csv')
    out_file = os.path.join(outfolder, outname)
    df_vars.to_csv(out_file)
    print('  NOAA CO-OPS variables for ' + station + ' written to ' + outname)

    # Summary to indicate missing data
    print('Start Date: ' + str(df_vars.index[0].date()))
    print('End Date: ' + str(df_vars.index[-1].date()))
    print('Total Days: ')
    print(str(len(df_vars)))
    df_missing = 1 - df_vars.isnull().sum()/len(df_vars)
    print('Percentage Available: ')
    print(df_missing)
