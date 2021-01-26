#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HF_EV_stats_plots.py

Created on Fri Jun 19 09:53:56 2020

@author: rtsearcy

Description: Bin analysis, correlations, boxplots for environmental data
"""

import pandas as pd
import os
import datetime
import numpy as np
from scipy import stats
from scipy.stats.mstats import gmean
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf, pacf
import pingouin as pg
import wq_modeling as wqm

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 100)

folder = '/Users/rtsearcy/Box/water_quality_modeling/thfs/EDA'
save_folder = '/Users/rtsearcy/Box/water_quality_modeling/thfs/EDA/summer2020'

#EV = ['rad', 'tide','WVHT','APD','Wtemp_B','atemp','wspd','wdir']
EV = ['rad', 
      'tide',
      #'dtide_1',
      #'dtide_2',
      'WVHT',
      'DPD',
      'APD',
      'Wtemp_B',
      'atemp',
      'dtemp', 
      'awind',
      'owind'] # Important enviro vars (EV)

#EV = ['rad', 'tide','WVHT','APD','Wtemp_B','atemp','wspd','wdir']
EV_dict = {'rad':r'$W/m^2$', 
      'tide':'m',
      'WVHT':'m',
      'DPD':'s',
      'APD':'s',
      'Wtemp_B':'°C',
      'atemp':'°C',
      'dtemp':'°C', 
      'awind':"m/s",
      'owind':'m/s'} # Important enviro vars (EV)

alpha = 0.05

LP = pd.read_csv(os.path.join(folder, 'all_event_data_LP.csv'), parse_dates=['dt'], index_col=['dt'])
CB = pd.read_csv(os.path.join(folder, 'all_event_data_CB.csv'), parse_dates=['dt'], index_col=['dt'])
HSB = pd.read_csv(os.path.join(folder, 'all_event_data_HSB.csv'), parse_dates=['dt'], index_col=['dt'])

df = pd.concat([LP,CB,HSB])  # all data

hf = df[df.event.isin(['LP-13','LP-16','LP-18','CB-11','CB-12','HSB-02'])] # HF events

# Daytime = between sunrise and sunset
df_log = pd.read_csv('/Users/rtsearcy/Box/water_quality_modeling/thfs/EDA/summer2020/logistics/logistics_all_events.csv', 
                     index_col=['event'])

for e in hf.event.unique():
    sr = pd.to_datetime(df_log.loc[e]['sunrise1']).time()
    ss = pd.to_datetime(df_log.loc[e]['sunset1']).time()
    hf.loc[hf.event==e,'daytime'] = [1 if (x > sr) and (x < ss) else 0 for x in hf[hf.event==e].index.time] 

# Solar Noon
#Values: (1) = hours = 9-3, (0) = 6-9 or 3-7
sn = hf[(hf.index.time >= datetime.time(9,0)) & (hf.index.time < datetime.time(15,0))]
hf['solar_noon'] = [1 if x in sn.index else 0 for x in hf.index]  # If sample was taken between 9a and 3p

# Interaction terms
hf['rad_tide'] = hf['rad']*hf['tide']
hf['WVHT_tide'] = hf['WVHT']*hf['tide']
hf['WVHT_Wtemp_B'] = hf['WVHT']*hf['Wtemp_B']
hf['Wtemp_B_tide'] = hf['tide']*hf['Wtemp_B']

lp13 = hf[hf.event=='LP-13']
lp16 = hf[hf.event=='LP-16']
lp18 = hf[hf.event=='LP-18']
lp13_16 = hf[hf.event.isin(['LP-13','LP-16'])]  # First two LP events
lp = hf[hf.event.isin(['LP-13','LP-16','LP-18'])]  # All LP events

cb11 = hf[hf.event=='CB-11']
cb12 = hf[hf.event=='CB-12']
cb = hf[hf.event.isin(['CB-11','CB-12'])]  # All CB events

hsb02 = hf[hf.event=='HSB-02']


#%% EV Distributions - Boxplots and Data Tables
p = True  # Plot?
bp_folder = os.path.join(save_folder, 'figures','EV_boxplots')

params = {
   'axes.labelsize': 11,
   'font.size': 11,
   'legend.fontsize': 11,
   'xtick.labelsize': 9,
   'ytick.labelsize': 9,
   'font.family'  : 'sans-serif',
   'font.sans-serif':'Helvetica',
   'axes.axisbelow': True,
   'xtick.top': True
   }
plt.rcParams.update(params)

df_dist = pd.DataFrame()
 
for v in EV:
    # By event
    # Plot
    if p:
        plt.figure(figsize=(6,3))
        ax = sns.boxplot(x=v,y='event',data=hf, color='white')
        
        for i,box in enumerate(ax.artists):  # Change box color
            box.set_edgecolor('black')
            box.set_facecolor('white')
    
            # iterate over whiskers and median lines
            for j in range(6*i,6*(i+1)):
                 ax.lines[j].set_color('black')
        
        plt.xlabel(v + ' (' + EV_dict[v] + ')')
        plt.ylabel('')
        plt.tight_layout()
        plt.savefig(os.path.join(bp_folder,v+'_event_boxplots.png'))
                    
   # Stats
    temp = hf.groupby('event').describe()[v]
    temp['CV'] = temp['mean'] / temp['std']
    temp['EV'] = v
    df_dist = df_dist.append(temp)

    # # By beach
    # if p:
    #     plt.figure(figsize=(6,3))
    #     ax = sns.boxplot(x=v,y='beach',data=hf, color='white')
        
    #     for i,box in enumerate(ax.artists):  # Change box color
    #         box.set_edgecolor('black')
    #         box.set_facecolor('white')
    
    #         # iterate over whiskers and median lines
    #         for j in range(6*i,6*(i+1)):
    #              ax.lines[j].set_color('black')
        
    #     plt.ylabel('')
    #     plt.tight_layout()
    
    temp = hf.groupby('beach').describe()[v]
    temp.index.name = 'event'
    temp['CV'] = temp['mean'] / temp['std']
    temp['EV'] = v
    df_dist = df_dist.append(temp)
    
# Overall
temp = hf[EV].describe().T
temp.index.name = 'EV'
temp.reset_index(inplace=True)
temp['CV'] = temp['mean'] / temp['std']
temp['event'] = 'All Data'
temp.set_index('event', inplace=True)
df_dist = df_dist.append(temp)

# Save
df_dist.to_csv(os.path.join(save_folder,'stats','EV_distribution_stats.csv'))

#%% Correlations - By Event + Lollipop Plot

# Semi-Partial correlation:
#https://www.statisticssolutions.com/what-are-zero-order-partial-and-part-correlations/
# http://faculty.cas.usf.edu/mbrannick/regression/Partial.html

p=True  # plot?
w = 0.15  # separation width
params = {
   'axes.labelsize': 11,
   'font.size': 11,
   'legend.fontsize': 11,
   'xtick.labelsize': 9,
   'ytick.labelsize': 9,
   'font.family'  : 'sans-serif',
   'font.sans-serif':'Helvetica',
   'axes.axisbelow': True,
   'xtick.top': True
   }
plt.rcParams.update(params)

event_corr = pd.DataFrame()
for b in hf.event.unique():
    temp = pd.DataFrame(index=EV,columns=[12*[b],['PCC_ENT','p_ENT','partial_ENT','p_partial_ENT','semi_ENT','p_semi_ENT',
                                                  'PCC_FC','p_FC', 'partial_FC','p_partial_FC', 'semi_FC','p_semi_FC']])
    print(b)
    
    # Correlations
    for f in ['FC','ENT']:
        
        if (b == 'HSB-02')& (f=='FC'):
            continue
        print('  ' + f)
        sig_list = []
        for v in EV:
            fib = hf[hf.event==b]['log'+f].dropna()
            evar = hf[hf.event==b][v].reindex(fib.index)
            
            # Full Corr
            pcc = stats.pearsonr(fib, evar)
            temp.loc[v][b,'PCC_'+f] = round(pcc[0],2)
            temp.loc[v][b,'p_'+f] = pcc[1]
            
            if pcc[1] <= alpha:
                sig_list.append(v)
                
            # Partial
            confound = [x for x in EV if x!=v]
            partial = pg.partial_corr(data=hf[hf.event==b], x = 'log'+f, y=v, covar=confound)
            temp.loc[v][b,'partial_'+f] = round(float(partial['r']),2)
            temp.loc[v][b,'p_partial_'+f] = float(partial['p-val'])
            
            if float(partial['p-val']) <= alpha:
                sig_list.append('*')
            
            # Semi-Partial
            confound = [x for x in EV if x!=v]
            partial = pg.partial_corr(data=hf[hf.event==b], x = 'log'+f, y=v, x_covar=confound)
            temp.loc[v][b,'semi_'+f] = round(float(partial['r']),2)
            temp.loc[v][b,'p_semi_'+f] = float(partial['p-val'])
            
            if float(partial['p-val']) <= alpha:
                sig_list.append('+')
                
        print(sig_list)
    
    event_corr = pd.concat((event_corr, temp), axis=1)
    
    # Plots
    # https://python-graph-gallery.com/184-lollipop-plot-with-2-groups/
    if p:
        plt.figure(figsize=(3,5.5))
        r = np.arange(0,len(temp))
        
        # ENT
        plt.hlines(y=r+w, xmin=temp[b,'partial_ENT'], xmax=temp[b,'PCC_ENT'], color='black', alpha=0.5, zorder=0)
        
        plt.scatter(temp[b,'PCC_ENT'], r+w, color='black', alpha=1, label='Zero-Order Correlation',zorder=1)
        plt.scatter(temp[b,'partial_ENT'], r+w, facecolors='white', edgecolors='black', alpha=1,zorder=1.5, label='Partial Correlation')
        
        # FC
        plt.hlines(y=r-w, xmin=temp[b,'partial_FC'], xmax=temp[b,'PCC_FC'], color='black', alpha=0.5, zorder=0.5)
        
        plt.scatter(temp[b,'PCC_FC'], r-w, color='black', alpha=.5, label='Zero-Order Correlation', zorder=10)
        plt.scatter(temp[b,'partial_FC'], r-w, facecolors='white', edgecolors='black', alpha=1 , label='Partial Correlation')
        
        plt.axvline(0, ls='--', color='grey', alpha=0.1)
        
        plt.title(b)
        plt.yticks(range(len(temp)),temp.index)
        plt.xlim(-1,1)
        plt.xticks(np.arange(-1,1.1,.25), ['-1.0','','-0.5','','0','','0.5','','1.0'])
        plt.ylim(-0.5,len(temp)-0.5)
        plt.grid(axis='x', color='gray', alpha=0.1)
        
        #plt.legend(frameon=False)
        plt.tight_layout()

        

event_corr.to_csv(os.path.join(save_folder,'stats','corr_by_event.csv'))

# Print table
A = event_corr.T
ptable = pd.DataFrame(index=hf.event.unique(), columns=[2*EV,len(EV)*['ENT'] + len(EV)*['FC']])
for c in A.columns:
    for f in ['ENT','FC']:
        for e in ptable.index:
            entry = str(A.loc[e][c]['PCC_'+f]) # PCC
            if A.loc[e][c]['p_'+f]<0.1:
                entry+='*'
            if A.loc[e][c]['p_'+f]<0.05:
                entry+='*'
            
            entry += '/' + str(A.loc[e][c]['partial_'+f]) # PCC
            if A.loc[e][c]['p_partial_'+f]<0.1:
                entry+='*'
            if A.loc[e][c]['p_partial_'+f]<0.05:
                entry+='*'
            ptable[c,f].loc[e] = entry

ptable.T.to_csv(os.path.join(save_folder,'stats','corr_by_event_FORMATTED.csv'))

#%% Correlations - By Beach
beach_corr = pd.DataFrame()
for b in hf.beach.unique():
    temp = pd.DataFrame(index=EV,columns=[12*[b],['PCC_ENT','p_ENT','partial_ENT','p_partial_ENT','semi_ENT','p_semi_ENT',
                                                  'PCC_FC','p_FC', 'partial_FC','p_partial_FC', 'semi_FC','p_semi_FC']])
    print(b)
    for f in ['FC','ENT']:
        if (b == 'HSB')& (f=='FC'):
            continue
        print('  ' + f)
        sig_list = []
        for v in EV:
            fib = hf[hf.beach==b]['log'+f].dropna()
            evar = hf[hf.beach==b][v].reindex(fib.index)
            
            # Full Corr
            pcc = stats.pearsonr(fib, evar)
            temp.loc[v][b,'PCC_'+f] = round(pcc[0],3)
            temp.loc[v][b,'p_'+f] = pcc[1]
            
            if pcc[1] <= alpha:
                sig_list.append(v)
                
            # Partial
            confound = [x for x in EV if x!=v]
            partial = pg.partial_corr(data=hf[hf.beach==b], x = 'log'+f, y=v, covar=confound)
            temp.loc[v][b,'partial_'+f] = round(float(partial['r']),3)
            temp.loc[v][b,'p_partial_'+f] = float(partial['p-val'])
            
            if float(partial['p-val']) <= alpha:
                sig_list.append('*')
            
            # Semi-Partial
            confound = [x for x in EV if x!=v]
            partial = pg.partial_corr(data=hf[hf.beach==b], x = 'log'+f, y=v, x_covar=confound)
            temp.loc[v][b,'semi_'+f] = round(float(partial['r']),3)
            temp.loc[v][b,'p_semi_'+f] = float(partial['p-val'])
            
            if float(partial['p-val']) <= alpha:
                sig_list.append('+')
                
        print(sig_list)
    
    beach_corr = pd.concat((beach_corr, temp), axis=1)

beach_corr.to_csv(os.path.join(save_folder,'stats','corr_by_beach.csv'))

#%% Correlations - All Data
overall_corr = pd.DataFrame(index=EV,columns=['PCC_ENT','p_ENT','partial_ENT','p_partial_ENT','semi_ENT','p_semi_ENT',
                                              'PCC_FC','p_FC', 'partial_FC','p_partial_FC', 'semi_FC','p_semi_FC'])
print('\nOverall Pearson Correlation')
for f in ['FC','ENT']:
    print(f)
    sig_list = []
    
    for v in EV:
        confound = [x for x in EV if x!=v]
        confound = ['log'+f]
        c=3
        fib = hf['log'+f].dropna()
        evar = hf[v].reindex(fib.index)
        
        # Full Corr
        pcc = stats.pearsonr(fib, evar)
        overall_corr.loc[v]['PCC_'+f] = round(pcc[0],3)
        overall_corr.loc[v]['p_'+f] = pcc[1]
        
        if pcc[1] <= alpha:
            c-=1
            
        # Partial
        partial = pg.partial_corr(data=hf, x = 'log'+f, y=v, covar=confound)
        overall_corr.loc[v]['partial_'+f] = round(float(partial['r']),3)
        overall_corr.loc[v]['p_partial_'+f] = float(partial['p-val'])
        
        if float(partial['p-val']) <= alpha:
            c-=1
        
        # Semi-Partial
        partial = pg.partial_corr(data=hf, x = 'log'+f, y=v, x_covar=confound)
        overall_corr.loc[v]['semi'+f] = round(float(partial['r']),3)
        overall_corr.loc[v]['p_semi_'+f] = float(partial['p-val'])
        
        if float(partial['p-val']) <= alpha:
            c-=1
            
        if c == 0:
            sig_list.append(v)
    print(sig_list)
            
overall_corr.to_csv(os.path.join(save_folder,'stats','corr_all_data.csv'))

#%% Correlations with low-freq variables

lf = ['dayofyear','lograin3T','lograin7T','wet3','wet7',
      'upwelling','spring_tide','days_since_full_moon']

FC = hf.groupby('event').mean()['logFC']
ENT = hf.groupby('event').mean()['logENT']
FCv = hf.groupby('event').var()['logFC']
ENTv = hf.groupby('event').var()['logENT']

for l in lf:
    print('\n'+l)
# Means
    print('mean (ENT/FC):')
    print(pg.corr(ENT,hf.groupby('event').max()[l])[['r','p-val']])
    print(pg.corr(FC,hf.groupby('event').max()[l])[['r','p-val']])
# Variances
    print('var (ENT/FC):')
    print(pg.corr(ENTv,hf.groupby('event').max()[l])[['r','p-val']])
    print(pg.corr(FCv,hf.groupby('event').max()[l])[['r','p-val']])

#%% 2
N = len(EV)
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]
 
for b in hf.beach.unique():
    
    ENT_corrs = list(hf[hf.beach==b].corr().loc['logENT'][EV])
    ENT_corrs+=ENT_corrs[:1]  # Add the first value to complete the circle
    
    FC_corrs = list(hf[hf.beach==b].corr().loc['logFC'][EV])
    FC_corrs+=FC_corrs[:1]  # Add the first value to complete the circle
    
    plt.figure(figsize=(5,5))
    
    ax = plt.subplot(111, polar=True)
    ax.set_title(b, loc='left')
 
    # First axis to be on top:
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], EV)
    
    # Draw ylabels
    ax.set_rlabel_position(5)
    plt.yticks([.25,.5,.75], ["0.25","0.5","0.75",'1.0'], color="grey", size=7)
    plt.ylim(0,0.75)

    # ENT
    ax.plot(angles, [abs(x) for x in ENT_corrs], linewidth=1, color='g', linestyle='solid', label="logENT")
    ax.fill(angles, [abs(x) for x in ENT_corrs], 'g', alpha=0.1)
    
    # FC
    ax.plot(angles, [abs(x) for x in FC_corrs], linewidth=1, color='b',alpha=.5, linestyle='solid', label="logENT")
    ax.fill(angles, [abs(x) for x in FC_corrs], 'b', alpha=0.1)

    # Markers indicating neg corr
    for i in range(N):
        if ENT_corrs[i]<0:
            ax.plot(angles[i]+0.05, abs(ENT_corrs[i]) + 0.05,'g_', ms = 7)
        if FC_corrs[i]<0:
            ax.plot(angles[i]+0.1, abs(FC_corrs[i]) + 0.05,'b_', ms = 7)
            
#%% Correlation Plot - Radar Plot (by EVENT)
# https://python-graph-gallery.com/391-radar-chart-with-several-individuals/            
            
N = len(EV)
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]
 
for b in hf.beach.unique():
    temp = hf[hf.beach==b]
    
    plt.figure(figsize=(5,10))
    
    
    # ENT subplot
    ax = plt.subplot(211, polar=True)
    ax.set_title('logENT', loc='left')
 
    # First axis to be on top:
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], EV)
    # Draw ylabels
    ax.set_rlabel_position(5)
    #plt.yticks([.25,.5,.75,1], ["0.25","0.5","0.75",'1.0']+, color="grey", size=7)
    plt.yticks([-1,-.5,0,.5,1], ["- 1.0","- 0.5"," 0","+ 0.5",'+ 1.0'], color="black",alpha=0.8, size=7)
    plt.ylim(-1,1)
    
    # FC subplot
    ax2 = plt.subplot(212, polar=True)
    ax2.set_title('logFC', loc='left')
 
    # First axis to be on top:
    ax2.set_theta_offset(np.pi / 2)
    ax2.set_theta_direction(-1)
    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], EV)
    # Draw ylabels
    ax2.set_rlabel_position(5)
    #plt.yticks([.25,.5,.75,1], ["0.25","0.5","0.75",'1.0'], color="grey", size=7)
    plt.yticks([-1,-.5,0,.5,1], ["- 1.0","- 0.5"," 0","+ 0.5",'+ 1.0'], color="black",alpha=0.8, size=7)
    plt.ylim(-1,1)
    
    # Colors
    c = ['g','b','k']
    count=0
    
    for e in temp.event.unique():
        
        ENT_corrs = list(temp[temp.event==e].corr().loc['logENT'][EV])
        ENT_corrs+=ENT_corrs[:1]  # Add the first value to complete the circle
    
        FC_corrs = list(temp[temp.event==e].corr().loc['logFC'][EV])
        FC_corrs+=FC_corrs[:1]  # Add the first value to complete the circle
    
        # ENT
        #ax.plot(angles, [abs(x) for x in ENT_corrs], color=c[count],linewidth=1, linestyle='solid', label=e)
        #ax.fill(angles, [abs(x) for x in ENT_corrs],color=c[count], alpha=0.1)
        ax.plot(angles, [x for x in ENT_corrs], color=c[count],linewidth=1, linestyle='solid', label=e)
        ax.fill(angles, [x for x in ENT_corrs],color=c[count], alpha=0.1)
        
        # FC
        # ax2.plot(angles, [abs(x) for x in FC_corrs], color=c[count], linewidth=1, alpha=.5, linestyle='solid', label=e)
        # ax2.fill(angles, [abs(x) for x in FC_corrs], color=c[count], alpha=0.1)
        ax2.plot(angles, [x for x in FC_corrs], color=c[count], linewidth=1, alpha=.5, linestyle='solid', label=e)
        ax2.fill(angles, [x for x in FC_corrs], color=c[count], alpha=0.1)
        
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))


        # # Markers indicating neg corr
        # for i in range(N):
        #     if ENT_corrs[i]<0:
        #         ax.plot(angles[i]+0.05, abs(ENT_corrs[i]) + 0.05, c[count]+'_', ms = 7)
        #     if FC_corrs[i]<0:
        #         ax2.plot(angles[i]+0.1, abs(FC_corrs[i]) + 0.05, c[count]+'_', ms = 7)
        count+=1
    plt.tight_layout()


#%% EV Bins - Rad
# Summary: Low rad significantly higher FIB for mist events (except for LP-16 and CB-12 at some thresholds)
# All beaches when events were combined, and all data

# Daytime only
# Mean of all events  - 430 W/m2
# 25% quantile of all events (Daytime Only) = 170 W/m2
# 33% quantile = 232
# Median = 360

# All samples
# Mean of all events  - 259 W/m2
# 25% quantile of all events (Daytime Only) = 0 W/m2
# 33% quantile = 0
# Median = 150

rad_thresh = 250
hf_day = hf[hf.daytime==1]  # Remove nighttime samples
hf_day['rad_GT'] = [1 if x > rad_thresh else 0 for x in  hf_day['rad']]  # For plotting

# Plot

params = {
   'axes.labelsize': 11,
   'font.size': 11,
   'legend.fontsize': 11,
   'xtick.labelsize': 9,
   'ytick.labelsize': 9,
   'font.family'  : 'sans-serif',
   'font.sans-serif':'Helvetica',
   'axes.axisbelow': True,
   'xtick.top': True
   }
plt.rcParams.update(params)

plt.figure()
plt.subplot(2,1,1)
sns.boxplot(x='beach', y='logENT', hue='rad_GT',data=hf_day)
plt.legend('',frameon=False)
plt.xlabel('')
plt.title('Rad Greater Than ' + str(rad_thresh) + ' W/m2 (Daytime Samples Only)')
plt.subplot(2,1,2)
sns.boxplot(x='beach', y='logFC', hue='rad_GT',data=hf_day)
plt.legend(frameon=False)
plt.xlabel('')


low_rad = hf_day[hf_day.rad_GT == 0]
high_rad = hf_day[hf_day.rad_GT == 1]

# By event
print('Rad\nBy Event:')
print(hf_day.groupby(['event','rad_GT']).size())
print(hf_day.groupby(['event','rad_GT']).mean()[['logENT','logFC']])
for e in hf_day.event.unique():
    print('  ' + e)
    for f in ['ENT','FC']:
        print('   ' + f)
        print(stats.mannwhitneyu(low_rad[low_rad.event==e]['log'+f],high_rad[high_rad.event==e]['log'+f]))

# By beach
print('\nBy Beach:')
print(hf_day.groupby(['beach','rad_GT']).size())
print(hf_day.groupby(['beach','rad_GT']).mean()[['logENT','logFC']])
for b in hf_day.beach.unique():
    print('  ' + b)
    for f in ['ENT','FC']:
        print('   ' + f)
        print(stats.mannwhitneyu(low_rad[low_rad.beach==b]['log'+f],high_rad[high_rad.beach==b]['log'+f]))

# Overall
print('\nAll Data')
print(hf_day.groupby('rad_GT').mean()[['logENT','logFC']])
print(stats.mannwhitneyu(low_rad['logENT'],high_rad['logENT']))
print(stats.mannwhitneyu(low_rad['logFC'],high_rad['logFC']))
# Result - Significantly higher when rad is lower


# Exceedances
print('\nExceedances')
print(hf_day[hf_day.rad_GT==0].groupby('event').sum()[['ENT_exc','FC_exc']] / 
                                                      hf_day.groupby('event').sum()[['ENT_exc','FC_exc']])

#%% EV Bins - Tide

# tide_gtm (tide greater than the mean tide at the beach)

params = {
   'axes.labelsize': 11,
   'font.size': 11,
   'legend.fontsize': 11,
   'xtick.labelsize': 9,
   'ytick.labelsize': 9,
   'font.family'  : 'sans-serif',
   'font.sans-serif':'Helvetica',
   'axes.axisbelow': True,
   'xtick.top': True
   }
plt.rcParams.update(params)
pal = ['#969696','#525252']
pal = sns.color_palette(pal)

hf['tcat'] = ['Higher' if x==1 else 'Lower' for x in hf['tide_gtm']]

plt.figure()
plt.subplot(2,1,1)
sns.boxplot(x='beach', y='logENT', hue='tcat',data=hf, palette=pal)
plt.legend('',frameon=False)
plt.xlabel('')
plt.title('Tide Greater Than Mean for Beach')
plt.subplot(2,1,2)
sns.boxplot(x='beach', y='logFC', hue='tcat',data=hf, palette=pal)
plt.legend(frameon=False)
plt.xlabel('')

plt.figure()
plt.subplot(2,1,1)
sns.boxplot(x='event', y='logENT', hue='tcat',data=hf, palette=pal)
plt.legend('',frameon=False)
plt.xlabel('')
#plt.title('Tide Greater Than Mean for Beach')
plt.subplot(2,1,2)
sns.boxplot(x='event', y='logFC', hue='tcat',data=hf, palette=pal)
plt.legend(frameon=False)
plt.xlabel('')
plt.ylabel('logEC')


low_tide = hf[hf.tide_gtm == 0]
high_tide = hf[hf.tide_gtm == 1]

# By event
print('Tide\nBy Event:')
print(hf.groupby(['event','tide_gtm']).size())
print(hf.groupby(['event','tide_gtm']).median()[['logENT','logFC']])
for e in hf.event.unique():
    print('  ' + e)
    for f in ['ENT','FC']:
        print('   ' + f)
        print(stats.mannwhitneyu(low_tide[low_tide.event==e]['log'+f],high_tide[high_tide.event==e]['log'+f]))

# By beach
print('\nBy Beach:')
print(hf.groupby(['beach','tide_gtm']).size())
print(hf.groupby(['beach','tide_gtm']).median()[['logENT','logFC']])
for b in hf.beach.unique():
    print('  ' + b)
    for f in ['ENT','FC']:
        print('   ' + f)
        print(stats.mannwhitneyu(low_tide[low_tide.beach==b]['log'+f],high_tide[high_tide.beach==b]['log'+f]))

# Overall
print('\nAll Data')
print(hf.groupby('tide_gtm').median()[['logENT','logFC']])
print(stats.mannwhitneyu(low_tide['logENT'],high_tide['logENT']))
print(stats.mannwhitneyu(low_tide['logFC'],high_tide['logFC']))

# Result - HSB and LP FIB significantly higher when tide is greater than the mean;
# No sig difference for CB (both events)


# Tide stage (Low/Slack/High)
pal = ['#cccccc','#969696','#525252']
pal = sns.color_palette(pal)

plt.figure() # by beach
plt.subplot(2,1,1)
sns.boxplot(x='beach', y='logENT', hue='tide_stage',data=hf, palette=pal)
plt.legend('',frameon=False)
plt.xlabel('')
plt.subplot(2,1,2)
sns.boxplot(x='beach', y='logFC', hue='tide_stage',data=hf, palette=pal)
plt.legend(['Low','Slack','High'],frameon=False)
plt.xlabel('')

plt.figure() # by event
plt.subplot(2,1,1)
sns.boxplot(x='event', y='logENT', hue='tide_stage',data=hf, palette=pal)
plt.legend('',frameon=False)
plt.xlabel('')
plt.subplot(2,1,2)
sns.boxplot(x='event', y='logFC', hue='tide_stage',data=hf, palette=pal)
plt.legend(['Low','Slack','High'],frameon=False)
plt.xlabel('')

# By event
print('Tide Stage:')
print('\nBy Event:')
print(hf.groupby(['event','tide_stage']).size())
print(hf.groupby(['event','tide_stage']).median()[['logENT','logFC']])
for e in hf.event.unique():
    print('  ' + e)
    for f in ['ENT','FC']:
        temp = hf[hf.event==e]
        print('   ' + f)
        print(stats.kruskal(temp[temp.tide_stage==-1]['log'+f], 
                            temp[temp.tide_stage==0]['log'+f], 
                            temp[temp.tide_stage==1]['log'+f]))


# By beach
print('\nBy Beach:')
print(hf.groupby(['beach','tide_stage']).size())
print(hf.groupby(['beach','tide_stage']).median()[['logENT','logFC']])
for b in hf.beach.unique():
    print('  ' + b)
    for f in ['ENT','FC']:
        temp = hf[hf.beach==b]
        print('   ' + f)
        print(stats.kruskal(temp[temp.tide_stage==-1]['log'+f], 
                            temp[temp.tide_stage==0]['log'+f], 
                            temp[temp.tide_stage==1]['log'+f]))

# Overall
print('\nAll Data')
print(hf.groupby('tide_stage').median()[['logENT','logFC']])
for f in ['ENT','FC']:
    temp = hf
    print(stats.kruskal(temp[temp.tide_stage==-1]['log'+f].dropna(), 
                            temp[temp.tide_stage==0]['log'+f].dropna(), 
                            temp[temp.tide_stage==1]['log'+f].dropna()))

# Result:
# When broken down by tide stage, ENT at CB does have slightly higher levels at high tide compared to slack or low
# FC at CB still has no sig difference
# For HSB and LP, FIB levels go with the tide (lowest at low, medium at slack)
    
