# tideVarsDaily.py - Computes daily tidal variables from raw CO-OPS 6 min water level data
# RS - 1/5/2017
# RS - 3/14/2018 - Update
# RTS - 11/11/2019

# Raw data source/description: https://tidesandcurrents.noaa.gov/tide_predictions.html

# NOTE: raw csv files should have timestamps in LST

# Tide Variables (Continuous): TideMax, TideMin, TideR (tidal range = max - min), TideMax1, TideMin1, TideR1
# Tide Variables (Binary): TideGT_# (tide greater than # m), TideLT_# (tide less than # m),
# TideGT1_#, TideLT1_# (1 signifies previous day's value)

import pandas as pd
import os
import re


def gt(df, thresh):  # Compute greater than binary var.
    a = df > thresh
    a = a.astype('int')
    return a


def lt(df, thresh):  # Compute less than binary var.
    a = df < thresh
    a = a.astype('int')
    return a


# Import raw data csv to pd DataFrame
#path = '/Users/rtsearcy/data/water_quality_modeling/forecasting/tides'
path = '/Users/rtsearcy/Box/water_quality_modeling/data/tide/observations'
raw_path = os.path.join(path, 'raw')
# To use for setting up daily run folder

sd = '20080101'  # Start date (account for previous day, conservative)
ed = '20200301'  # End date


# TO PROCESS SINGLE FILE
# (change indent)
# file = 'NorthSplit_Tidal_Data_20080101_20201231.csv'
# infile = os.path.join(infolder, file)

# TO ITERATE THROUGH ALL FILES
df_means = pd.DataFrame()
for file in os.listdir(raw_path):
    if not file.endswith('.csv'):
        continue
    infile = os.path.join(raw_path, file)
    station = re.sub('_Tide_.+', '', file).replace('_', ' ')  # find station name from filename
    print('\nProcessing tidal data for ' + station + ' ...')
    df_raw = pd.read_csv(infile)
    df_raw.columns = ['dt', 'tide']

    #  Convert Date Time to timestamp, set as index
    df_raw['dt'] = pd.to_datetime(df_raw['dt'])
    df_raw.set_index('dt', inplace=True)
    mht = float(round(df_raw.resample('D').max().mean(), 4))  # Mean high and low tides from entire infile (2000-2020)
    mlt = float(round(df_raw.resample('D').min().mean(), 4))

    df_raw = df_raw[sd:ed]  # Only samples in time range (for speed)
    df_out = pd.DataFrame(index=df_raw.resample('D').mean().index)  # Preset index to days

    #%% Tide (Tide level at 6a PST) - MAY NEED TO VARY THIS IF SAMPLE TIME IS KNOWN
    df_out['Tide5'] = df_raw[(df_raw.index.hour == 5) & (df_raw.index.minute == 0)].resample('D').mean()
    df_out['Tide6'] = df_raw[(df_raw.index.hour == 6) & (df_raw.index.minute == 0)].resample('D').mean()
    df_out['Tide7'] = df_raw[(df_raw.index.hour == 7) & (df_raw.index.minute == 0)].resample('D').mean()
    df_out['Tide8'] = df_raw[(df_raw.index.hour == 8) & (df_raw.index.minute == 0)].resample('D').mean()
    #%% Tide Stage - TODO if sample time is available

    #%% Max/Min/Range
    df_out['TideMax'] = df_raw.resample('D').max()  # Max tide
    df_out['TideMin'] = df_raw.resample('D').min()  # Min tide
    df_out['TideR'] = df_out.TideMax - df_out.TideMin  # Tidal range
    print('  Maximum: ' + str(df_out['TideMax'].max()) + ' ; Minimum: ' + str(df_out['TideMin'].min()) +
          ' ; Max Range: ' + str(df_out['TideR'].max()))

    #%% Greater/Less Than
    gt_list = [2, mht]
    for g in gt_list:
        if g != mht:
            df_out['TideGT_' + str(g).replace('.', '_')] = gt(df_out.TideMax, g)  # Was Tide today greater than 'g' ?
        else:
            df_out['TideGT_mht'] = gt(df_out.TideMax, g)
            # Was the tide greater than the mean high tide in the entire dataset?

    lt_list = [0]
    for g in lt_list:
        if g != mlt:
            df_out['TideLT_' + str(g).replace('-', '').replace('.', '_')] = lt(df_out.TideMin, g)
            # Was Tide today less than 'g' ?
        else:
            df_out['TideLT_mlt'] = lt(df_out.TideMin, g)
            # Was the tide less than the mean low tide in the entire dataset?

    #%% Prev day vars; NOTE: only works if you are SURE that days are consecutive (no gaps)
    for c in list(df_out.columns):
        if 'GT' in c:
            df_out[c.replace('GT', 'GT1')] = df_out[c].shift(1)  # Previous Day's value timestamped to today
        elif 'LT' in c:
            df_out[c.replace('LT', 'LT1')] = df_out[c].shift(1)  # Previous Day's value timestamped to today
        elif c in ['TideMax','TideMin','TideR']:
            df_out[c + '1'] = df_out[c].shift(1)  # Previous Day's value timestamped to today

    #%% Spring/Neap
    
    
    #%% Save to file
    of_name = station.replace(' ', '_') + '_Tide_Variables_' + sd + '_' + ed + '.csv'
    outfile = os.path.join(path, of_name)
    df_out.index.rename('date', inplace=True)
    df_out.to_csv(outfile)

    means = {'mean_High': mht, 'mean_Low': mlt}
    df_means = df_means.append(pd.DataFrame(means, index=[station]))
    print('Variables written to ' + outfile)

#%% Save means for each station
mf_name = 'tidal_means.csv'
mean_file = os.path.join(path, mf_name)
df_means.index.rename('station', inplace=True)
df_means.to_csv(mean_file)
print('\nTidal means written to ' + mean_file)
