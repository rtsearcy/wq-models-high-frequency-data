# getMet_NCDC.py - Download raw met data from National Climatic Data Center (NCDC).
# RTS - 3/21/2018 (Updated 10/23/2019)

# Grabs hourly METAR data for airport stations along the coast, saves to file
# Parses raw data into met variables
# Note: This is a different data source of METAR data than what is used for 
# implementation (AWS). Data matches up well, though.

import pandas as pd
import numpy as np
import requests
import os

# Inputs
outfolder = '/Users/rtsearcy/Box/water_quality_modeling/data/met/NCDC'
airport_file = os.path.join(outfolder, 'airports.csv')  # file with station metadata (see below for necessary columns)

sd = '1997-12-31'  # start date, in YYYY-MM-DD format (account for previous day)
ed = '2020-03-31'  # end date, account for 8hr UTC shift

rs = 0  # 0 - grab raw data from internet; 1 - grab raw data from flat files (saves time)

# Import Airport Stations
df_air = pd.read_csv(airport_file)
df_air.set_index('NAME', inplace=True)
air_list = ['John Wayne'] #list(df_air.index)  # or custom list on airport locations

print('Meterological Data\nDirectory: ' + outfolder )
for a in air_list:
    print('\nProcessing meteorological data for ' + a + ' (' + df_air.loc[a]['CALL'] + ')')

    raw_skip = rs
    # Grab data from NCDC
    if raw_skip == 0 | len([x for x in os.listdir(outfolder) if x.startswith(a.replace(' ', ''))]) == 0:
        USAF = str(df_air.loc[a]['USAF'])
        WBAN = str(df_air.loc[a]['WBAN'])
        if len(WBAN) != 5:
            WBAN = '0'*(5-len(WBAN)) + WBAN
        st_id = USAF + WBAN
        #murl = 'https://www.ncdc.noaa.gov/access-data-service/api/v1/data?dataset=global-hourly'
        url = 'https://www.ncei.noaa.gov/access/services/data/v1?dataset=global-hourly'
        payload = {
            'startDate': sd,
            'endDate': ed,
            'stations': st_id,
            'format': 'json',
            'includeAttributes': 'false'
            }
        print('  Searching for raw data via NCDC')
        r = requests.get(url, params=payload)
        try:
            r.raise_for_status()
            df_raw = pd.DataFrame(r.json())
            df_raw = df_raw[df_raw['REPORT_TYPE'] == 'FM-15']  # METAR format only
            print('   ' + str(len(df_raw)) + ' METAR records found')
            df_raw['dt'] = pd.to_datetime(df_raw['DATE']) - pd.to_timedelta('8 hours')  # UTC to PST (-8hr)
            df_raw.set_index('dt', inplace=True)
            sd_new = str(df_raw.index[0].date())
            ed_new = str(df_raw.index[-1].date())
            print('Min. Date - ' + sd_new + '\nMax Date - ' + ed_new)
            cols = ['NAME', 'STATION', 'CALL_SIGN', 'REM', 'TMP', 'DEW', 'SLP', 'WND', 'AA1']
            df_raw = df_raw[cols]

            # Save raw METAR data
            raw_file = a.replace(' ', '_') + '_raw_METAR_data_' + sd_new.replace('-', '') + '_' \
                + ed_new.replace('-', '') + '.csv'
            df_raw.to_csv(os.path.join(outfolder, raw_file))
            print('  METAR data saved to ' + raw_file)

        except Exception as exc:
            print('  There was a problem grabbing met data: %s' % exc)
            continue

    else:  # grab data from csv files
        try:
            raw_file = [x for x in os.listdir(outfolder) if x.startswith(a.replace(' ', '_'))][0]
            df_raw = pd.read_csv(outfolder_raw + raw_file)
        except:
            print('Could not find existing data for' + a)
            continue
        print('  Found METAR data in ' + raw_file)
        df_raw['dt'] = pd.to_datetime(df_raw['dt'])  # already in PST
        df_raw.set_index('dt', inplace=True)
        sd_new = str(df_raw.index[0].date())
        ed_new = str(df_raw.index[-1].date())
        print('  ' + str(len(df_raw)) + ' METAR records found')
        print('Min. Date - ' + sd_new + '\nMax Date - ' + ed_new + '\n')

    # Process Raw METAR Data #
    SF = 10  # scaling factor
    df_raw = df_raw.resample('H').last()  # Resample by hour, selecting last value if multiple
    # Note: timestamp now shows exactly on the hour, values are from the last METAR of that hour
    #     Ex. 8:00 - values from 8:56
    # This means that for calculation of sums and means grouped by day, the 0 hour will be included for that day
    # For rain_6h (first 6h of rain for the day, values from hour 0 - 5 should be included)

    # Temperature (degC)
    df_raw['temp'] = df_raw['TMP'][df_raw['TMP'].notnull()].apply(lambda x: x.split(',')[0])
    df_raw['temp'] = pd.to_numeric(df_raw['temp'], errors='coerce')/SF
    df_raw['temp'][df_raw['temp'] > 100] = np.nan  # Account for 99999 values
    print('Temperature parsed')

    # Dew Point Temperature (degC)
    df_raw['dtemp'] = df_raw['DEW'][df_raw['DEW'].notnull()].apply(lambda x: x.split(',')[0])
    df_raw['dtemp'] = pd.to_numeric(df_raw['dtemp'], errors='coerce') / SF
    df_raw['dtemp'][df_raw['dtemp'] > 100] = np.nan  # Account for 99999 values
    print('Dew point temperature parsed')

    # Sea Level Pressure (mbar)
    df_raw['pres'] = df_raw['SLP'][df_raw['SLP'].notnull()].apply(lambda x: x.split(',')[0])
    df_raw['pres'] = pd.to_numeric(df_raw['pres'], errors='coerce') / SF
    df_raw['pres'][df_raw['pres'] > 1500] = np.nan  # Account for 99999 values
    print('Sea level pressure parsed')

    # Wind Direction (deg) and Speed (m/s)
    df_raw['wdir'] = df_raw['WND'][df_raw['WND'].notnull()].apply(lambda x: x.split(',')[0])  # wind direction
    df_raw['wdir'] = pd.to_numeric(df_raw['wdir'], errors='coerce')
    df_raw['wdir'][df_raw['wdir'] > 360] = np.nan  # Account for 999 values
    print('Wind direction parsed')

    df_raw['wspd'] = df_raw['WND'][df_raw['WND'].notnull()].apply(lambda x: x.split(',')[3])  # wind speed
    df_raw['wspd'] = pd.to_numeric(df_raw['wspd'], errors='coerce') / SF
    df_raw['wspd'][df_raw['wspd'] > 90] = np.nan
    print('Wind speed parsed')

    # Precipitation (mm)
    df_raw['rain'] = df_raw['AA1'][df_raw['AA1'].notnull()].apply(lambda x: x.split(',')[1])
    df_raw['rain'] = pd.to_numeric(df_raw['rain'], errors='coerce') / SF
    df_raw['rain'][df_raw['rain'].isnull()] = 0
    df_raw['rain'][df_raw['rain'] > 900] = np.nan
    print('Rain parsed')

    # Parameterize met data into variables
    df_vals = df_raw[['temp', 'dtemp', 'pres', 'wspd', 'wdir', 'rain']]  # Values dataframe
    rounder = {'temp': 1,  # sig figs
               'dtemp': 1,
               'pres': 1,
               'wspd': 1,
               'wdir': 0,
               'rain': 1}

    hourly_file = a.replace(' ', '_') + '_hourly_data_' + sd_new.replace('-', '') + '_' \
                + ed_new.replace('-', '') + '.csv'
    df_vals.to_csv(os.path.join(outfolder, hourly_file))
    df_daily = pd.DataFrame(index=df_vals.resample('D').mean().index)
    df_vars = pd.DataFrame(index=df_vals.resample('D').mean().index)

    # Mean
    for c in df_vals.columns:
        if c != 'rain':
            df_daily[c] = round(df_vals[c].resample('D').mean(), rounder[c])
            df_vars[c + '1'] = df_daily[c].shift(1, freq='D')  # previous day
    # Max
    for c in df_vals.columns:
        if c not in ['wdir', 'rain']:
            df_daily[c + '_max'] = df_vals[c].resample('D').max()
            df_vars[c + '1_max'] = df_daily[c + '_max'].shift(1, freq='D')  # previous day

    # Min
    for c in df_vals.columns:
        if c not in ['wdir', 'rain']:
            df_daily[c + '_min'] = df_vals[c].resample('D').min()
            df_vars[c + '1_min'] = df_daily[c + '_min'].shift(1, freq='D')  # previous day

    rr = 4  # rain rounder
    # rain_6h - First 6 hours of rain in the day
    df_vals_6h = df_vals[(df_vals.index.hour == 0) | (df_vals.index.hour == 1) | (df_vals.index.hour == 2) | (
     df_vals.index.hour == 3) | (df_vals.index.hour == 4) | (df_vals.index.hour == 5)]
    df_daily['rain_6h'] = df_vals_6h['rain'].resample('D').sum()
    df_vars['lograin_6h'] = round(np.log10(df_daily['rain_6h']), rr)
    df_vars['lograin_6h'][np.isneginf(df_vars['lograin_6h'])] = round(np.log10(0.005), rr)

    # rain
    df_daily['rain'] = df_vals['rain'].resample('D').sum()
    for i in range(1, 8):  # rain1 - rain7, lograin1-lograin7
        df_daily['rain' + str(i)] = df_daily['rain'].shift(i, freq='D')
        df_vars['lograin' + str(i)] = round(np.log10(df_daily['rain' + str(i)]), rr)
        df_vars['lograin' + str(i)][np.isneginf(df_vars['lograin' + str(i)])] = round(np.log10(0.005), rr)

    total_list = list(range(2, 8)) + [14, 30]
    for j in total_list:  # rain2T-rain7T
        df_daily['rain' + str(j) + 'T'] = 0.0
        for k in range(j, 0, -1):
            df_daily['rain' + str(j) + 'T'] += df_daily['rain'].shift(k, freq='D')
        df_vars['lograin' + str(j) + 'T'] = round(np.log10(df_daily['rain' + str(j) + 'T']), rr)
        df_vars['lograin' + str(j) + 'T'][np.isneginf(df_vars['lograin' + str(j) + 'T'])] = round(np.log10(0.005), rr)

    # Wet Days
    df_daily['wet'] = (df_daily[['rain_6h', 'rain3T']] > 2.54).any(axis=1).astype(int)
    # Not including same day rain because not available for daily runs, and samples are typically taken early in the day
    df_vars['wet'] = (df_vars[['lograin_6h', 'lograin3T']] > 0.4048).any(axis=1).astype(int)

    # Save to file
    dailyfile = a.replace(' ', '_') + '_daily_data_' + sd_new.replace('-', '') + '_' + ed_new.replace('-', '') + '.csv'
    df_daily.index.rename('date', inplace=True)
    df_daily.to_csv(os.path.join(outfolder, dailyfile))  # Save daily file
    print('\nDaily data saved to: ' + dailyfile)

    outfile = a.replace(' ', '_') + '_Met_Variables_' + sd_new.replace('-', '') + '_' + ed_new.replace('-', '') + '.csv'
    df_vars.index.rename('date', inplace=True)
    df_vars.to_csv(os.path.join(outfolder, outfile))  # Save vars file
    print('Meteorological variables saved to: ' + outfile)
