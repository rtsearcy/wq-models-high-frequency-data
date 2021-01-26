#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wave_impute.py - Missing Wave Imputer
Created on Fri Apr 10 13:47:34 2020
 
Imputes missing wave data from a regression on correlated data from another station 

@author: rtsearcy
"""

import os
import pandas as pd
import numpy as np
from scipy import stats

folder = '/Users/rtsearcy/Box/water_quality_modeling/data/waves'
missing_file = 'Cabrillo_Point_Nearshore_Wave_Variables_20080101_20191031.csv'
fill_file = 'Monterey_Bay_Outer_Wave_Variables_20080101_20191031.csv'
outname = 'Imputed_from_MB_Outer_to_Cabrillo.csv'

df1 = pd.read_csv(os.path.join(folder, missing_file), parse_dates=['dt'], index_col = ['dt'])
df2 = pd.read_csv(os.path.join(folder, fill_file), parse_dates=['dt'], index_col = ['dt'])

df_out = pd.DataFrame()
for c in df1.columns:
    A = df1[c]
    B = df2[c]
    C = pd.concat([A,B], axis=1).dropna()
    C.columns = ['A','B']
    pcc = stats.pearsonr(C['A'],C['B'])
    model = stats.linregress(C['B'],C['A'])
    C['mod'] = model.intercept + model.slope*C['B']
    rmse = np.sqrt(((C['mod']-C['B'])**2).sum()/len(C))
    
    print(c)
    print('  PCC: ' + str(pcc))
    print('  Model:  Missing = ' + str(model.intercept) + ' + Fill * ' + str(model.slope))
    print('  R2: ' + str(model.rvalue**2))
    print('  RMSE: ' + str(rmse))
    
    df_out[c] = model.intercept + model.slope * df2[c]
    
outfile = os.path.join(folder, outname)
df_out.to_csv(outfile, index=True)
    