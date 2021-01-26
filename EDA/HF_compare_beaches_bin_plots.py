#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:41:10 2020

@author: rtsearcy

Description: Plots and some stats for EDA on the HF sampling and environmental data
"""

import pandas as pd
import os
import numpy as np
from scipy import stats
import statsmodels.api as sm
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 100)

folder = '/Users/rtsearcy/Box/water_quality_modeling/thfs/EDA'

LP = pd.read_csv(os.path.join(folder, 'all_event_data_LP.csv'), parse_dates=['dt'], index_col=['dt'])
CB = pd.read_csv(os.path.join(folder, 'all_event_data_CB.csv'), parse_dates=['dt'], index_col=['dt'])
HSB = pd.read_csv(os.path.join(folder, 'all_event_data_HSB.csv'), parse_dates=['dt'], index_col=['dt'])

df = pd.concat([LP,CB,HSB])

hf = df[df.event.isin(['LP-13','LP-16','LP-18','CB-11','CB-12','HSB-02'])]
lp13 = hf[hf.event=='LP-13']
lp16 = hf[hf.event=='LP-16']
lp18 = hf[hf.event=='LP-18']
cb11 = hf[hf.event=='CB-11']
cb12 = hf[hf.event=='CB-12']
hsb02 = hf[hf.event=='HSB-02']

trad = df[df.event.isin(['LP-RM-FIB','CB-RM','HSB-RM'])]
trad = trad['2000':'2020']

trad_vars = df[df.event.isin(['LP-RM','CB-RM','HSB-RM'])]
trad_vars = trad_vars['2000':'2020']

#%% HF FIB time series
# Date labels, ticks (note about CB sample rate) 
tide = True
plt.figure(figsize=(12,5))
c = 1
params = {
   'axes.labelsize': 9,
   'font.size': 11,
   'legend.fontsize': 10,
   'xtick.labelsize': 9,
   'ytick.labelsize': 9,
   'font.family'  : 'sans-serif',
   'font.sans-serif':'Helvetica'
   }
plt.rcParams.update(params)


#for i in ['LP-13','CB-11','LP-16','CB-12','LP-18','HSB-02']:
for i in ['HSB-02','LP-13','CB-11','LP-16','CB-12','LP-18']:
    h = hf[hf.event == i]
    plt.subplot(3,2,c)
    if i == 'HSB-02':  # Only ENT sampled at HSB-02
        ftp = ['logENT']
    else:
        ftp = ['logENT','logFC']
    ax = h[ftp].plot(ax=plt.gca(), color = ['k','grey'], alpha=0.7)
    #ax.set_xticklabels('')
    
    if tide==True:
        ax2 = h['tide'].plot(secondary_y=True, color='teal', ls=':')
        plt.ylabel('Water Level [m]', color = 'teal', rotation=270, ha='center', va='baseline', rotation_mode='anchor')
        if i == 'LP-13':  # Legend
            plt.legend(loc=(.5,.82),frameon=False)
            plt.ylim([0., 1.5])
        plt.sca(ax)
    else:
        ax.tick_params(right=True)
    
    plt.ylim([0.9, 4])
    plt.ylabel(r'log$_{10}$ CFU/100 ml')
    plt.xlabel('')
    plt.text(0.01, 0.87, i, transform=ax.transAxes)  # Event label
    
    if i == 'LP-13':  # Legend
        plt.legend(loc=(.64,.82),ncol=2,columnspacing=.5,frameon=False)
    else:
        plt.legend('',frameon=False)
    
    plt.axhline(y=np.log10(104),color='k',ls='--',linewidth=1) # SSS threshold lines
    if i != 'HSB-02':
        plt.axhline(y=np.log10(400),color='grey',ls='-.',linewidth=1)
    
    # tick, labels = plt.xticks()
    # for j in range(0,len(labels)):
    #     l = labels[j].get_text()
    #     if '00:00\n' in l:
    #         print('yes')
    #         labels[j].set_text(l.replace('00:00','test'))
            
    #plt.xticks(tick, labels)
    #ax.set_xticklabels(labels)
    
    c+=1
    
plt.tight_layout(pad=.8,h_pad=0.01,w_pad=0.01)
if tide:
    plt.subplots_adjust(wspace=0.18, hspace=0.325, top=0.98, bottom=0.108)
else:
    plt.subplots_adjust(wspace=0.1, hspace=0.25)

# top=0.981,
# bottom=0.108,
# left=0.04,
# right=0.955,
# hspace=0.325,
# wspace=0.18

#%% HF FIB - Boxplots by event
    
# Melt fib_data
hf['logEC'] = hf['logFC']
hf_melt =  pd.melt(hf, id_vars=['event'], value_vars=['logEC','logENT'], var_name='FIB',value_name='logFIB')

params = {
   'axes.labelsize': 11,
   'font.size': 11,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 9,
   'font.family'  : 'sans-serif',
   'font.sans-serif':'Helvetica'
   }
plt.rcParams.update(params)
pal = ['#969696','#525252']
pal = sns.color_palette(pal)

plt.figure(figsize=(10,4))
ax = sns.boxplot(x='event',y='logFIB',hue='FIB',data=hf_melt, palette=pal)
plt.ylabel(r'log$_{10}$ MPN/100 ml')
plt.xlabel('')
plt.legend(frameon=False)
plt.tight_layout()
plt.subplots_adjust(bottom=0.1)

    
#%% HF FIB category by event
#Categories: 0 - FIB <= 10 MPN/100ml (Low); 
#1 - 10 < FIB <= 0.75* State Standard (Medium);
#2 - 0.75* State Standard < FIB <= 1.25* State Standard (High);
#3 - FIB > 1.255* State Standard (Very High)
params = {
   'axes.labelsize': 11,
   'font.size': 11,
   'legend.fontsize': 9.5,
   'xtick.labelsize': 10,
   'ytick.labelsize': 11,
   'font.family'  : 'sans-serif',
   'font.sans-serif':'Helvetica',
   'axes.axisbelow': True
   }
plt.rcParams.update(params)

pal = ['#253494','#2c7fb8','#41b6c4','#a1dab4']  # Color Blind Friendly
pal = sns.color_palette(pal)
plt.figure(figsize=(8,8))
plt.subplot(2,2,1)
hf.groupby(['event', 'FC_cat']).size().reset_index().pivot(columns='FC_cat',index='event', values=0).dropna(how='all').plot(kind='bar', stacked=True, ax=plt.gca(), color=pal)
plt.title('High-Frequency (HF) Events')
plt.ylabel('# FC Samples')
plt.xlabel('')
plt.legend('',frameon=False)
plt.subplot(2,2,3)
hf.groupby(['event','ENT_cat']).size().reset_index().pivot(columns='ENT_cat',index='event', values=0).dropna(how='all').plot(kind='bar', stacked=True, ax=plt.gca(), color=pal)
plt.ylabel('# ENT Samples')
plt.legend('',frameon=False)
plt.xlabel('')
plt.subplot(2,2,2)
trad.groupby(['event','FC_cat']).size().reset_index().pivot(columns='FC_cat',index='event', values=0).dropna(how='all').plot(kind='bar', stacked=True, ax=plt.gca(), color=pal)
plt.title('Routine Monitoring (RM)')
plt.xlabel('')
plt.legend(labels=['Low','Medium','High','Very High'], frameon=False)
plt.subplot(2,2,4)
trad.groupby(['event','ENT_cat']).size().reset_index().pivot(columns='ENT_cat',index='event', values=0).dropna(how='all').plot(kind='bar', stacked=True, ax=plt.gca(), color=pal)
plt.xlabel('')
plt.legend('', frameon=False)
plt.tight_layout()
plt.subplots_adjust(hspace=0.3)

# Exceedances
pal = ['#969696','#525252']
pal = sns.color_palette(pal)
plt.figure(figsize=(8,8))
plt.subplot(2,2,1)
hf.groupby(['event', 'FC_exc']).size().reset_index().pivot(columns='FC_exc',index='event', values=0).dropna(how='all').plot(kind='bar', stacked=True, ax=plt.gca(), color=pal)
plt.title('High-Frequency (HF) Events')
plt.ylabel('# FC Samples')
plt.legend('',frameon=False)
plt.xlabel('')
plt.subplot(2,2,3)
hf.groupby(['event','ENT_exc']).size().reset_index().pivot(columns='ENT_exc',index='event', values=0).dropna(how='all').plot(kind='bar', stacked=True, ax=plt.gca(), color=pal)
plt.ylabel('# ENT Samples')
plt.legend('',frameon=False)
plt.xlabel('')
plt.subplot(2,2,2)
trad.groupby(['event','FC_exc']).size().reset_index().pivot(columns='FC_exc',index='event', values=0).dropna(how='all').plot(kind='bar', stacked=True, ax=plt.gca(), color=pal)
plt.title('Routine Monitoring (RM)')
plt.xlabel('')
plt.legend(labels=['Attainment','Exceedance'], frameon=False)
plt.subplot(2,2,4)
trad.groupby(['event','ENT_exc']).size().reset_index().pivot(columns='ENT_exc',index='event', values=0).dropna(how='all').plot(kind='bar', stacked=True, ax=plt.gca(), color=pal)
plt.xlabel('')
plt.legend('',frameon=False)
plt.tight_layout()
plt.subplots_adjust(hspace=0.3)

#%% RM FIB by Year
pal = ['#a6cee3','#1f78b4','#b2df8a']  # Colorblind, greyscale friendly
pal = ['#cccccc','#969696','#525252']
pal = sns.color_palette(pal)
params = {
   'axes.labelsize': 11,
   'font.size': 11,
   'legend.fontsize': 10,
   'xtick.labelsize': 11,
   'ytick.labelsize': 9,
   'font.family'  : 'sans-serif',
   'font.sans-serif':'Helvetica',
   'axes.axisbelow': True
   }
plt.rcParams.update(params)

trad['year']=trad.index.year
summer = True
trad_y = trad['2000':'2020']
if summer:
    trad_y = trad_y[trad_y.index.month.isin([4,5,6,7,8,9,10])]

## Time Series
rm_ent = trad_y.groupby(['beach','year']).mean()['logENT']
ENT_y = rm_ent.reset_index().pivot(index='year',columns='beach',values='logENT')
ent_std = trad_y.groupby(['beach','year']).std()['logENT']
ent_n = trad_y.groupby(['beach','year']).count()['logENT']
ent_err = 1.96 * ent_std / (ent_n**0.5)  # 95% CI
ent_err = ent_err.reset_index().pivot(index='year',columns='beach',values='logENT')

rm_fc = trad_y.groupby(['beach','year']).mean()['logFC']
FC_y = rm_fc.reset_index().pivot(index='year',columns='beach',values='logFC')
fc_std = trad_y.groupby(['beach','year']).std()['logFC']
fc_n = trad_y.groupby(['beach','year']).count()['logFC']
fc_err = 1.96 * fc_std / (fc_n**0.5)  # 95% CI
fc_err = fc_err.reset_index().pivot(index='year',columns='beach',values='logFC')

plt.figure(figsize=(8,6))
plt.subplot(2,1,1)
ax = ENT_y[['HSB','CB','LP']].plot(ax=plt.gca(),color=pal, marker='.', yerr=ent_err)
plt.xlabel('')
plt.ylabel(r'log$_{10}$ CFU/100 ml')
plt.legend('',frameon=False)
plt.text(0.01, 0.87, r'log$_{10}$ENT', transform=ax.transAxes)
plt.xticks(ticks= [2000,2002,2004,2006,2008,2010,2012,2014,2016,2018,2020])

plt.subplot(2,1,2)
ax2 = FC_y[['HSB','CB','LP']].plot(ax=plt.gca(), color=pal, marker='.',yerr=fc_err)
plt.legend(bbox_to_anchor=(0.5,-0.25), loc="center", ncol=3, fontsize=10, title='Beach')
plt.xlabel('')
plt.ylabel(r'log$_{10}$ CFU/100 ml')
plt.text(0.01, 0.87, r'log$_{10}$FC', transform=ax2.transAxes)
plt.xticks(ticks= [2000,2002,2004,2006,2008,2010,2012,2014,2016,2018,2020])
plt.subplots_adjust(bottom=0.135)

## Boxplots

# trad['year_4bin'] = 0 #4-year bins (0: 00-04, 1: 05-08, 2:09-12, 3: 13-16, 4: 17:20)
# trad['year_4bin'][trad.year.isin([2005,2006,2007,2008])] = 1
# trad['year_4bin'][trad.year.isin([2009,2010,2011,2012])] = 2
# trad['year_4bin'][trad.year.isin([2013,2014,2015,2016])]= 3
# trad['year_4bin'][trad.year.isin([2017,2018,2019,2020])]= 4
# y = ['2000-2004', '2005-2008', '2009-2012', '2013-2016', '2017-2020']

trad_y['year_5bin'] = 0 #4-year bins (0: 00-05, 1: 06-10, 2:11-15, 3: 16-20)
trad_y['year_5bin'][trad_y.year.isin([2006,2007,2008,2009,2010])] = 1
trad_y['year_5bin'][trad_y.year.isin([2011,2012,2013,2014,2015])] = 2
trad_y['year_5bin'][trad_y.year.isin([2016,2017,2018,2019,2020])]= 3
y = ['2000-2005', '2006-2010', '2011-2015', '2016-2020']

plt.figure(figsize=(8,6))
plt.subplot(2,1,1)
ax = sns.boxplot(x='year_5bin',y='logENT',hue='beach',data=trad_y, palette=pal)
plt.xlabel('')
plt.legend('',frameon=False)
ticks, labels = plt.xticks()
if len(labels) in [4,5]:
    for i in range(0,len(labels)):
        labels[i].set_text('')
plt.xticks(ticks,labels)
plt.subplot(2,1,2)
ax2 = sns.boxplot(x='year_5bin',y='logFC',hue='beach',data=trad_y, palette=pal)
plt.legend(bbox_to_anchor=(0.5,-0.25), loc="center", ncol=3, fontsize=10, title='Beach')
plt.xlabel('')
ticks, labels = plt.xticks()
if len(labels) in [4,5]:
    for i in range(0,len(labels)):
        labels[i].set_text(y[i])
plt.xticks(ticks,labels)
plt.tight_layout()
plt.subplots_adjust(bottom=0.135)

#%% RM FIB by month
trad['month']=trad.index.month
pal = ['#a6cee3','#1f78b4','#b2df8a']
pal = ['#cccccc','#969696','#525252']
pal = sns.color_palette(pal)

params = {
   'axes.labelsize': 11,
   'font.size': 11,
   'legend.fontsize': 11,
   'xtick.labelsize': 9,
   'ytick.labelsize': 9,
   'font.family'  : 'sans-serif',
   'font.sans-serif':'Helvetica',
   'axes.axisbelow': True
   }
plt.rcParams.update(params)

plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plt.grid(axis='y',alpha=0.5)
ax = sns.boxplot(x='month',y='logFC',hue='beach',data=trad, palette=pal)
plt.xlabel('')
plt.legend('',frameon=False)
ax.tick_params(right=True)

ticks, labels = plt.xticks()
m = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
for i in range(0,len(labels)):
    labels[i].set_text(m[i])
plt.xticks(ticks,labels)
plt.subplot(2,1,2)
ax2 = sns.boxplot(x='month',y='logENT',hue='beach',data=trad, palette=pal)
plt.grid(axis='y',alpha=0.5)
plt.legend(bbox_to_anchor=(0.5,-0.25), loc="center", ncol=3, fontsize=10, title='Beach')
plt.xlabel('')
ax2.tick_params(right=True)
ticks, labels = plt.xticks()
for i in range(0,len(labels)):
    labels[i].set_text(m[i])
plt.xticks(ticks,labels)
plt.tight_layout()
plt.subplots_adjust(bottom=0.135)

#%% RM FIB by Time of Day
pal = ['#a6cee3','#1f78b4','#b2df8a']
pal = ['#cccccc','#969696','#525252']
pal = sns.color_palette(pal)

params = {
   'axes.labelsize': 11,
   'font.size': 11,
   'legend.fontsize': 11,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'font.family'  : 'sans-serif',
   'font.sans-serif':'Helvetica',
   'axes.axisbelow': True
   }
plt.rcParams.update(params)


summer = True
trad['hour']=trad.index.hour
trad_y = trad['2000':'2020']
if summer:
    trad_y = trad_y[trad_y.index.month.isin([4,5,6,7,8,9,10])]
# Drop 
trad_h = trad_y[~trad_y['sample_time'].isnull()]  # drop samples without a sample time

## Time Series
rm_ent = trad_h.groupby(['beach','hour']).mean()['logENT']
ENT_y = rm_ent.reset_index().pivot(index='hour',columns='beach',values='logENT')
ent_std = trad_h.groupby(['beach','hour']).std()['logENT']
ent_n = trad_h.groupby(['beach','hour']).count()['logENT']
ent_err = 1.96 * ent_std / (ent_n**0.5)  # 95% CI
ent_err = ent_err.reset_index().pivot(index='hour',columns='beach',values='logENT')

rm_fc = trad_h.groupby(['beach','hour']).mean()['logFC']
FC_y = rm_fc.reset_index().pivot(index='hour',columns='beach',values='logFC')
fc_std = trad_h.groupby(['beach','hour']).std()['logFC']
fc_n = trad_h.groupby(['beach','hour']).count()['logFC']
fc_err = 1.96 * fc_std / (fc_n**0.5)  # 95% CI
fc_err = fc_err.reset_index().pivot(index='hour',columns='beach',values='logFC')

plt.figure(figsize=(6,6.5))
plt.subplot(2,1,1)
ax = ENT_y[['HSB','CB','LP']].plot(ax=plt.gca(),color=pal, marker='.',ms=8,linewidth=1.5, yerr=ent_err)
plt.xlabel('')
plt.ylabel(r'log$_{10}$ CFU/100 ml')
plt.legend(loc="topright", ncol=1, fontsize=10, title='Routine Monitoring', frameon=False)
plt.xlim([-0.25,23.25])
plt.ylim([0.4,3.75])
plt.text(0.01, 0.92, r'log$_{10}$ENT', transform=ax.transAxes)
plt.xticks(ticks= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
           labels = [0,'',2,'',4,'',6,'',8,'',10,'',12,'',14,'',16,'',18,'',20,'',22,''])

plt.subplot(2,1,2)
ax2 = FC_y[['HSB','CB','LP']].plot(ax=plt.gca(), color=pal, marker='.',ms=8,linewidth=1.5,yerr=fc_err)
plt.xlabel('Hour of Day')
plt.xlim([-0.25,23.25])
plt.ylim([0.4,3.75])
plt.ylabel(r'log$_{10}$ CFU/100 ml')
plt.text(0.01, 0.92, r'log$_{10}$FC', transform=ax2.transAxes)
plt.xticks(ticks= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
           labels = [0,'',2,'',4,'',6,'',8,'',10,'',12,'',14,'',16,'',18,'',20,'',22,''])
plt.tight_layout()
plt.legend('',frameon=False)
plt.subplots_adjust(bottom=0.1, hspace=.13)


# Boxplots
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
ax = sns.boxplot(x='hour',y='logFC',hue='beach',data=trad_h, palette=pal)
plt.xlabel('')
plt.legend(loc="topright", ncol=3, fontsize=10, title='Routine Monitoring', frameon=False)

plt.subplot(2,1,2)
ax2 = sns.boxplot(x='hour',y='logENT',hue='beach',data=trad_h, palette=pal)
plt.xlabel('Hour of Day')
plt.legend('',frameon=False)
plt.tight_layout()
plt.subplots_adjust(bottom=0.1)

#%% HF FIB by Time of Day
hf['hour']=hf.index.hour
pal = ['#a6cee3','#1f78b4','#b2df8a']
pal = ['#cccccc','#969696','#525252']
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

## Time Series [log-transformed]
hf_ent = hf.groupby(['beach','hour']).mean()['logENT']
ENT_h = hf_ent.reset_index().pivot(index='hour',columns='beach',values='logENT')
ent_std = hf.groupby(['beach','hour']).std()['logENT']
ent_n = hf.groupby(['beach','hour']).count()['logENT']
ent_err = 1.96 * ent_std / (ent_n**0.5)  # 95% CI
ent_err = ent_err.reset_index().pivot(index='hour',columns='beach',values='logENT')

hf_fc = hf.groupby(['beach','hour']).mean()['logFC']
FC_h = hf_fc.reset_index().pivot(index='hour',columns='beach',values='logFC')
fc_std = hf.groupby(['beach','hour']).std()['logFC']
fc_n = hf.groupby(['beach','hour']).count()['logFC']
fc_err = 1.96 * fc_std / (fc_n**0.5)  # 95% CI
fc_err = fc_err.reset_index().pivot(index='hour',columns='beach',values='logFC')

plt.figure(figsize=(8,6.5))
plt.subplot(2,1,1)
ax = ENT_h[['HSB','CB','LP']].plot(ax=plt.gca(),color=pal, marker='.',ms=8,linewidth=1.5, yerr=ent_err)
plt.xlabel('')
plt.xlim([-0.25,23.25])
plt.ylim([0.75,3.75])
plt.ylabel(r'log$_{10}$ CFU/100 ml')
plt.legend(loc="upper left", ncol=3, fontsize=10, frameon=False)
#plt.legend('',frameon=False)
plt.text(0.94, 0.92, r'ENT', transform=ax.transAxes)
# plt.xticks(ticks= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
#            labels = [0,'',2,'',4,'',6,'',8,'',10,'',12,'',14,'',16,'',18,'',20,'',22,''])
plt.xticks(ticks= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
           labels = ['-12','','-10','','-8','','-6','','-4','','-2','','0','','+2','','+4','','+6','','+8','','+10',''])
ax.tick_params(axis='x', which='minor', bottom=True)

plt.subplot(2,1,2)
ax2 = FC_h[['HSB','CB','LP']].plot(ax=plt.gca(), color=pal, marker='.',ms=8,linewidth=1.5,yerr=fc_err)
plt.legend('',frameon=False)
# plt.legend(loc="bottom left", ncol=3, fontsize=10, frameon=False)

plt.xlim([-0.25,23.25])
plt.ylim([0.75,3.75])
plt.ylabel(r'log$_{10}$ CFU/100 ml')
plt.text(0.95, 0.92, r'FC', transform=ax2.transAxes)
# plt.xlabel('Hour of Day')
# plt.xticks(ticks= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
#            labels = [0,'',2,'',4,'',6,'',8,'',10,'',12,'',14,'',16,'',18,'',20,'',22,''])
plt.xlabel('Hours from Solar Noon')
plt.xticks(ticks= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
           labels = ['-12','','-10','','-8','','-6','','-4','','-2','','0','','+2','','+4','','+6','','+8','','+10',''])
plt.tight_layout()
plt.subplots_adjust(bottom=0.1, hspace=.13)


# ## Boxplots
# plt.figure(figsize=(12,6))
# plt.subplot(2,1,1)
# ax = sns.boxplot(x='hour',y='logFC',hue='beach',data=hf, palette=pal)
# plt.xlabel('')
# plt.legend(loc="topright", ncol=3, fontsize=10, title='High Frequency', frameon=False)
# plt.subplot(2,1,2)
# ax2 = sns.boxplot(x='hour',y='logENT',hue='beach',data=hf, palette=pal)
# plt.legend('',frameon=False)
# plt.xlabel('Hour of Day')
# plt.tight_layout()
# plt.subplots_adjust(bottom=0.1)



#%% Exceedances
# Daytime exceedances by event
df.groupby(['beach','event','daytime','ENT_exc']).count()['tide'].dropna()
# Results: Only 1 nighttime sample during trad event; remaining night samples are for HF13 ONLY
#* tide is used for the count function because there are no missing datapoints for tide

# Morn/Afternoon/Night exceedances by event
df.groupby(['beach','event','morn_aft_night','ENT_exc']).count()['tide']
df.groupby(['beach','event','morn_aft_night','FC_exc']).count()['tide']
# Results: HF13 ENT exceedances occur when sun is down; HF18 ENT exceedances occur mostly in the afternoon
# FC exceedances during HF13 and HF 16 mostly in the afternoon
# trad_samples FC and ENT exceedances occur mostly in the morning (though there are very few night samples)

sns.catplot(x='logENT', y='morn_aft_night', row='beach', kind='box', orient='h', data=trad, palette=pal)
sns.catplot(x='logFC', y='morn_aft_night', row='beach', kind='box', orient='h', data=trad, palette=pal)

#%% Time of Day (Morn/Aft)
pal = ['#a6cee3','#1f78b4','#b2df8a']  # Colorblind, greyscale friendly
pal = ['#cccccc','#969696','#525252']
pal = sns.color_palette(pal)
params = {
   'axes.labelsize': 11,
   'font.size': 11,
   'legend.fontsize': 10,
   'xtick.labelsize': 11,
   'ytick.labelsize': 11,
   'font.family'  : 'sans-serif',
   'font.sans-serif':'Helvetica',
   'axes.axisbelow': True
   }
plt.rcParams.update(params)

summer = True
trad['hour']=trad.index.hour
trad_ma = trad
if summer:
    trad_ma = trad_ma[trad_ma.index.month.isin([4,5,6,7,8,9,10])]
# Drop 
trad_ma = trad_ma[~trad_ma['sample_time'].isnull()]  # drop samples without a sample time (about 2000 samples,most from HSB)
trad_ma = trad_ma[trad_ma.hour.isin([5,6,7,8,9,10,11,12,14,14,15,16,17,18])]
hf_ma = hf[hf.hour.isin([5,6,7,8,9,10,11,12,14,14,15,16,17,18])]

trad_ma['morn_aft'] = ['Morning' if i in([5,6,7,8,9,10,11]) else 'Afternoon' for i in trad_ma.hour]
hf_ma['morn_aft'] = ['Morning' if i in([5,6,7,8,9,10,11]) else 'Afternoon' for i in hf_ma.hour]

plt.figure(figsize=(8,8))
plt.subplot(2,2,1)
sns.boxplot(x='beach',y='logENT',hue='morn_aft', data=hf_ma, palette=pal[1:])
plt.axhline(y=np.log10(104), color='k',ls='--', alpha=0.5)
plt.title('High-Frequency (HF) Events')
plt.ylabel(r'ENT [log$_{10}$ CFU/100ml]')
plt.legend('',frameon=False)
plt.xlabel('')
plt.subplot(2,2,3)
sns.boxplot(x='beach',y='logFC',hue='morn_aft', data=hf_ma, palette=pal[1:])
plt.axhline(y=np.log10(400), color='k',ls='--', alpha=0.5)
plt.ylabel(r'FC [log$_{10}$ CFU/100ml]')
plt.xlabel('')
plt.legend('',frameon=False)
plt.subplot(2,2,2)
sns.boxplot(x='beach',y='logENT',hue='morn_aft', data=trad_ma, palette=pal[1:])
plt.axhline(y=np.log10(104), color='k',ls='--', alpha=0.5)
plt.title('Routine Monitoring (RM)')
plt.legend('',frameon=False)
plt.xlabel('')
plt.subplot(2,2,4)
sns.boxplot(x='beach',y='logFC',hue='morn_aft', data=trad_ma, palette=pal[1:])
plt.axhline(y=np.log10(400), color='k',ls='--', alpha=0.5)
plt.legend(title='',frameon=False)
plt.xlabel('')
plt.tight_layout()

# # Test if morning and afternoon means are different
# aft = df[df.morn_aft_night == 2][['logFC','logENT']]
# morn = df[df.morn_aft_night == 1][['logFC','logENT']]
# stats.mannwhitneyu(morn['logENT'], aft['logENT'])
# #Out[81]: MannwhitneyuResult(statistic=49803.0, pvalue=0.038247719474763145)
# stats.mannwhitneyu(morn['logFC'], aft['logFC'])
# #Out[82]: MannwhitneyuResult(statistic=47305.0, pvalue=0.008387984001205637)

# # Exceedences by hour of day
# df.groupby(['event','hour','ENT_exc']).count()['tide'].dropna()
# df.groupby(['event','hour','FC_exc']).count()['tide']
# # Results: Seems like exceedances rarely occur between 11a and sunset. 
# # FC during HF16 was really high at noon (6 exceedances), (note: rad dipped around noon also)


#%% Solar Rad
df['rad_gtm'] = [1 if i > 375 else 0 for i in df['rad']]  # Mean rad in dataset is 375 W/m2
df.groupby(['event','rad_gtm']).count()['tide'].dropna()
# Distributed high rad, low rad days in trad

df.groupby(['event','rad_gtm','ENT_exc']).count()['tide'].dropna()
df.groupby(['event','rad_gtm','FC_exc']).count()['tide'].dropna()
# Exceedances do tend to occur when rad is lower than the mean more often, though not in HF18

pal = ['#a6cee3','#1f78b4','#b2df8a']  # Colorblind, greyscale friendly
pal = ['#cccccc','#969696','#525252']
pal = ['#969696','#525252']
pal = sns.color_palette(pal)
params = {
   'axes.labelsize': 11,
   'font.size': 11,
   'legend.fontsize': 10,
   'xtick.labelsize': 11,
   'ytick.labelsize': 11,
   'font.family'  : 'sans-serif',
   'font.sans-serif':'Helvetica',
   'axes.axisbelow': True
   }
plt.rcParams.update(params)

var_light = df[(df.index.hour >= 6) & (df.index.hour < 19)]
# Test if Sunny and not Sunny means are different
sunny = var_light[var_light.rad_gtm == 1][['logFC','logENT']]
not_sunny = var_light[var_light.rad_gtm == 0][['logFC','logENT']]
stats.mannwhitneyu(sunny['logENT'], not_sunny['logENT'])
#MannwhitneyuResult(statistic=21191.5, pvalue=1.8544440779793416e-05)
stats.mannwhitneyu(sunny['logFC'], not_sunny['logFC'])
#Out[98]: MannwhitneyuResult(statistic=16643.0, pvalue=2.3814616646864342e-12)

# Boxplots
trad_vars['rad_gtm'] = [1 if i > 375 else 0 for i in trad_vars['rad']]
hf['rad_gtm'] = [1 if i > 375 else 0 for i in hf['rad']]

pal = ['#a6cee3','#1f78b4','#b2df8a']  # Colorblind, greyscale friendly
pal = ['#cccccc','#969696','#525252']
pal = sns.color_palette(pal)
params = {
   'axes.labelsize': 11,
   'font.size': 11,
   'legend.fontsize': 10,
   'xtick.labelsize': 11,
   'ytick.labelsize': 11,
   'font.family'  : 'sans-serif',
   'font.sans-serif':'Helvetica',
   'axes.axisbelow': True
   }
plt.rcParams.update(params)

summer = True
trad_rad = trad_vars
if summer:
    trad_rad = trad_rad[trad_rad.index.month.isin([4,5,6,7,8,9,10])]
trad_rad = trad_rad[(trad_rad.index.hour >= 6) & (trad_rad.index.hour < 19)]
hf_rad = hf[(hf.index.hour >= 6) & (hf.index.hour < 19)]

plt.figure(figsize=(8,8))
plt.subplot(2,2,1)
sns.boxplot(x='event',y='logENT',hue='rad_gtm', data=hf_rad, dodge=True, palette=pal[1:])
plt.axhline(y=np.log10(104), color='k',ls='--', alpha=0.5)
plt.title('High-Frequency (HF) Events')
plt.ylabel(r'ENT [log$_{10}$ CFU/100ml]')
plt.legend('',frameon=False)
plt.xlabel('')
plt.subplot(2,2,3)
sns.boxplot(x='event',y='logFC',hue='rad_gtm', data=hf_rad, dodge=True, palette=pal[1:])
plt.axhline(y=np.log10(400), color='k',ls='--', alpha=0.5)
plt.ylabel(r'FC [log$_{10}$ CFU/100ml]')
plt.xlabel('')
plt.legend('',frameon=False)
plt.subplot(2,2,2)
sns.boxplot(x='beach',y='logENT',hue='rad_gtm', data=trad_rad, palette=pal[1:])
plt.axhline(y=np.log10(104), color='k',ls='--', alpha=0.5)
plt.title('Routine Monitoring (RM)')
plt.legend(labels=['Low Rad','High Rad'],markerscale=4,frameon=False)
plt.xlabel('')
plt.subplot(2,2,4)
sns.boxplot(x='beach',y='logFC',hue='rad_gtm', data=trad_rad, palette=pal[1:])
plt.axhline(y=np.log10(400), color='k',ls='--', alpha=0.5)
plt.legend('',frameon=False)
plt.xlabel('')
plt.tight_layout()


#%% Tide - Spring/Neap
df.groupby(['event','spring_tide']).count()['tide'].dropna()
#Results: HF13 and HF18 (neap tide), HF16 spring tide, even distribution in trad

#Exceedances by spring/neap tide cycle
trad.groupby(['event','spring_tide','ENT_exc']).count()['tide'].dropna()
    
spring = trad[trad.spring_tide==1]
neap = trad[trad.spring_tide==0]
stats.mannwhitneyu(spring['logFC'], neap['logFC'])
stats.mannwhitneyu(spring['logENT'], neap['logENT'])

#Results: ENT: Exceedances during HF13 and HF18 only (neap days), FC: occured during both spring (HF16) and neap (HF13)
#but for trad, twice as many ENT and FC exc in spring tide than neap

pal = ['#a6cee3','#1f78b4','#b2df8a']  # Colorblind, greyscale friendly
pal = ['#cccccc','#969696','#525252']
pal = sns.color_palette(pal)
params = {
   'axes.labelsize': 11,
   'font.size': 11,
   'legend.fontsize': 10,
   'xtick.labelsize': 11,
   'ytick.labelsize': 11,
   'font.family'  : 'sans-serif',
   'font.sans-serif':'Helvetica',
   'axes.axisbelow': True
   }
plt.rcParams.update(params)

summer = True
trad_sp = trad
if summer:
    trad_sp = trad_sp[trad_sp.index.month.isin([4,5,6,7,8,9,10])]

plt.figure(figsize=(8,8))
plt.subplot(2,2,1)
sns.boxplot(x='event',y='logENT',hue='spring_tide', data=hf, dodge=False, palette=pal[1:])
plt.axhline(y=np.log10(104), color='k',ls='--', alpha=0.5)
plt.title('High-Frequency (HF) Events')
plt.ylabel(r'ENT [log$_{10}$ CFU/100ml]')
plt.legend('',frameon=False)
plt.xlabel('')
plt.subplot(2,2,3)
sns.boxplot(x='event',y='logFC',hue='spring_tide', data=hf, dodge=False, palette=pal[1:])
plt.axhline(y=np.log10(400), color='k',ls='--', alpha=0.5)
plt.ylabel(r'FC [log$_{10}$ CFU/100ml]')
plt.xlabel('')
plt.legend('',frameon=False)
plt.subplot(2,2,2)
sns.boxplot(x='beach',y='logENT',hue='spring_tide', data=trad_sp, palette=pal[1:])
plt.axhline(y=np.log10(104), color='k',ls='--', alpha=0.5)
plt.title('Routine Monitoring (RM)')
plt.legend(labels=['Neap Tide','Spring Tide'],markerscale=4,frameon=False)
plt.xlabel('')
plt.subplot(2,2,4)
sns.boxplot(x='beach',y='logFC',hue='spring_tide', data=trad_sp, palette=pal[1:])
plt.axhline(y=np.log10(400), color='k',ls='--', alpha=0.5)
plt.legend('',frameon=False)
plt.xlabel('')
plt.tight_layout()

#%% Tide greater than mean
df.groupby(['event','tide_gtm','ENT_exc']).count()['tide'].dropna()
df.groupby(['event','tide_gtm','FC_exc']).count()['tide'].dropna()
#Results: exceedances significantly more abundant when tide is about 0.88 m (mean 2008-2016)

tgm = trad[(trad.tide_gtm==1) & (trad.beach=='LP')]
tlm = trad[(trad.tide_gtm==0) & (trad.beach=='LP')]
stats.mannwhitneyu(tgm['logFC'], tlm['logFC'])
stats.mannwhitneyu(tgm['logENT'], tlm['logENT'])

pal = ['#a6cee3','#1f78b4','#b2df8a']  # Colorblind, greyscale friendly
pal = ['#cccccc','#969696','#525252']
pal = sns.color_palette(pal)
params = {
   'axes.labelsize': 11,
   'font.size': 11,
   'legend.fontsize': 10,
   'xtick.labelsize': 11,
   'ytick.labelsize': 11,
   'font.family'  : 'sans-serif',
   'font.sans-serif':'Helvetica',
   'axes.axisbelow': True
   }
plt.rcParams.update(params)

summer = True
trad_sp = trad
if summer:
    trad_sp = trad_sp[trad_sp.index.month.isin([4,5,6,7,8,9,10])]

plt.figure(figsize=(8,8))
plt.subplot(2,2,1)
sns.boxplot(x='event',y='logENT',hue='tide_gtm', data=hf, palette=pal[1:])
plt.axhline(y=np.log10(104), color='k',ls='--', alpha=0.5)
plt.title('High-Frequency (HF) Events')
plt.ylabel(r'ENT [log$_{10}$ CFU/100ml]')
plt.legend('',frameon=False)
plt.xlabel('')
plt.subplot(2,2,3)
sns.boxplot(x='event',y='logFC',hue='tide_gtm', data=hf, palette=pal[1:])
plt.axhline(y=np.log10(400), color='k',ls='--', alpha=0.5)
plt.ylabel(r'FC [log$_{10}$ CFU/100ml]')
plt.xlabel('')
plt.legend('',frameon=False)
plt.subplot(2,2,2)
sns.boxplot(x='beach',y='logENT',hue='tide_gtm', data=trad_sp, palette=pal[1:])
plt.axhline(y=np.log10(104), color='k',ls='--', alpha=0.5)
plt.title('Routine Monitoring (RM)')
plt.legend(labels=['tide < mean','tide > mean'],markerscale=4,frameon=False)
plt.xlabel('')
plt.subplot(2,2,4)
sns.boxplot(x='beach',y='logFC',hue='tide_gtm', data=trad_sp, palette=pal[1:])
plt.axhline(y=np.log10(400), color='k',ls='--', alpha=0.5)
plt.legend('',frameon=False)
plt.xlabel('')
plt.tight_layout()


#%% tide stage (low, ebb/flood, high)
df.groupby(['event','tide_stage','ENT_exc']).count()['tide'].dropna()
df.groupby(['event','tide_stage','FC_exc']).count()['tide'].dropna()

pal = ['#a6cee3','#1f78b4','#b2df8a']  # Colorblind, greyscale friendly
pal = ['#cccccc','#969696','#525252']
pal = sns.color_palette(pal)
params = {
   'axes.labelsize': 11,
   'font.size': 11,
   'legend.fontsize': 10,
   'xtick.labelsize': 11,
   'ytick.labelsize': 11,
   'font.family'  : 'sans-serif',
   'font.sans-serif':'Helvetica',
   'axes.axisbelow': True
   }
plt.rcParams.update(params)

summer = True
trad_sp = trad
if summer:
    trad_sp = trad_sp[trad_sp.index.month.isin([4,5,6,7,8,9,10])]

plt.figure(figsize=(10,7))
plt.subplot(2,2,1)
sns.boxplot(x='event',y='logENT',hue='tide_stage', data=hf, palette=pal)
plt.axhline(y=np.log10(104), color='k',ls='--', alpha=0.5)
plt.title('High-Frequency (HF) Events')
plt.ylabel(r'ENT [log$_{10}$ CFU/100ml]')
plt.legend('',frameon=False)
plt.xlabel('')
plt.subplot(2,2,3)
sns.boxplot(x='event',y='logFC',hue='tide_stage', data=hf, palette=pal)
plt.axhline(y=np.log10(400), color='k',ls='--', alpha=0.5)
plt.ylabel(r'FC [log$_{10}$ CFU/100ml]')
plt.xlabel('')
plt.legend('',frameon=False)
plt.subplot(2,2,2)
sns.boxplot(x='beach',y='logENT',hue='tide_stage', data=trad_sp, palette=pal)
plt.axhline(y=np.log10(104), color='k',ls='--', alpha=0.5)
plt.title('Routine Monitoring (RM)')
plt.legend(labels=['Low Tide','Ebb/Flood','High Tide'],markerscale=4,frameon=False)
plt.xlabel('')
plt.subplot(2,2,4)
sns.boxplot(x='beach',y='logFC',hue='tide_stage', data=trad_sp, palette=pal)
plt.axhline(y=np.log10(400), color='k',ls='--', alpha=0.5)
plt.legend('',frameon=False)
plt.xlabel('')
plt.tight_layout()

#%% Wind direction
df.groupby(['event','wdir_cat','ENT_exc']).count()['tide'].dropna()
df.groupby(['event','wdir_cat','FC_exc']).count()['tide'].dropna()

#Result:
# Wind is mostly alongshore (downbeach) [Cat 3]
# Alongshore wind was important for exceedances during and HF and trad events
# Though maybe less discrepent if accounting for the number of samples in each wdir_cat bin
# Exceedances do seem to occur also when wind is offshore (wdir_cat == 0)


# #%% Swarmplots of variable distributions by HF event
# var = {
#        'tide':r'Tide $[m]$',
#        'rad':r'Solar Insolation $[W/m^2]$',
#        'hour':'Hour of Day',
#        'wspd':r'Wind Speed $[m/s]$',
#        'Wtemp_B':r'Water Temperature $[°C]$',
#        'WVHT': r'Significant Wave Height $[m]$'
#        }

# # Map legend
# plt.rcParams.update({'font.size': 10})
# pal = ['#253494','#2c7fb8' ,'#41b6c4','#a1dab4']  # Color Blind Friendly
# #pal = ['#cccccc','#525252'] # Color Blind eFriendly
# pal = sns.color_palette(pal)

# df_var = hf

# for f in ['FC','ENT']:
#     df_var[f+'map'] =df_var[f+'_cat'].map({0:'Low',1:'Medium',2:'High',3:'Very High'})  # Cat
#     #df_var[f+'map'] =df_var[f+'_exc'].map({0: 'Below SSS', 1: 'Above SSS'})   # Exc       
#     plt.figure(figsize=(12,8))
#     c=1
#     for v in var:
#         plt.subplot(2,len(var)/2,c)
#         sns.swarmplot(x='event',y=v,hue=f+'map',data=df_var,size=3, palette=pal)
#         plt.ylabel('')
#         plt.xlabel('')
#         plt.legend('',frameon=False)
#         plt.title(var[v])
#         if c==5:
#             plt.legend(bbox_to_anchor=(0.5,-0.25), loc="center", ncol=4, fontsize=10, title=f + ' Category')
#         c+=1
#     plt.suptitle('Environmental Variables and ' + f + ' Distributions by Event', fontsize=12)

#     plt.tight_layout()
#     plt.subplots_adjust(top=0.934,
#     bottom=0.13,
#     left=0.036,
#     right=0.986,
#     hspace=0.18,
#     wspace=0.18)

#%% Swarmplots of variable distributions by HF event and RM per beach [INDIVIDUAL]
var = {
       #'tide':{'label':r'Tide [m]', 'ymin':-0.7,'ymax':2.3},
       #'rad': {'label':r'Solar Insolation [W/m$^2$]', 'ymin':-50,'ymax':1050},
       #'hour':{'label':'Hour of Day','ymin':-0.5,'ymax':23.5},
       #'wspd':{'label': r'Wind Speed [m/s]','ymin':-0.5,'ymax':7},
       #'Wtemp_B':{'label': r'Water Temperature [°C]', 'ymin':10,'ymax':20},
       #'WVHT': {'label':r'Significant Wave Height [m]', 'ymin':-0.5,'ymax':5}
       }

# Map legend
params = {
   'axes.labelsize': 11,
   'font.size': 11,
   'legend.fontsize': 11,
   'xtick.labelsize': 9,
   'ytick.labelsize': 9,
   'font.family'  : 'sans-serif',
   'font.sans-serif':'Helvetica',
   'axes.axisbelow': True
   }
plt.rcParams.update(params)
pal = ['#253494','#2c7fb8' ,'#41b6c4','#a1dab4']  # Color Blind Friendly
#pal = ['#cccccc','#525252'] # Color Blind eFriendly
pal = sns.color_palette(pal)

df_plot = df
df_plot['hour'] = df_plot.index.hour
df_plot = df_plot[~df_plot['sample_time'].isnull()] 


for f in ['FC','ENT']:
        df_plot[f+'map'] =df_plot[f+'_cat'].map({0:'Low',1:'Medium',2:'High',3:'Very High'})  # Cat
        #df_var[f+'map'] =df_var[f+'_exc'].map({0: 'Below SSS', 1: 'Above SSS'})   # Exc

for v in var:
    for f in ['ENT']:
        plt.figure(figsize=(12,5))
        c=1
        for b in ['CB','LP','HSB']:
            dp = df_plot[df_plot.beach==b]
            plt.subplot(1,3,c)
            ax = sns.swarmplot(x='event',y=v,hue=f+'map',data=dp,size=4, palette=pal)
            plt.ylabel('')
            if c==1:
                plt.ylabel(var[v]['label'])
            plt.xlabel('')
            plt.ylim(var[v]['ymin'],var[v]['ymax'])
            ax.tick_params(right=True)
            plt.legend('',frameon=False)
            if c==2:
                plt.legend(bbox_to_anchor=(0.5,-0.15), loc="center", ncol=4, fontsize=10, title=f + ' Category')
            c+=1

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.15)

#%%#%% Swarmplots by EVENT
var = {
       'tide':{'label':r'Tide [m]', 'ymin':-0.7,'ymax':2.3},
       'rad': {'label':r'Solar Insolation [W/m$^2$]', 'ymin':-50,'ymax':1050},
       'hour':{'label':'Hour of Day','ymin':-0.5,'ymax':23.5},
       'wspd':{'label': r'Wind Speed [m/s]','ymin':-0.5,'ymax':7},
       'Wtemp_B':{'label': r'Water Temperature [°C]', 'ymin':10,'ymax':16},
       'WVHT': {'label':r'Significant Wave Height [m]', 'ymin':0,'ymax':2.5}
       }

# Map legend
params = {
   'axes.labelsize': 10,
   'font.size': 10,
   'legend.fontsize': 11,
   'xtick.labelsize': 9,
   'ytick.labelsize': 9,
   'font.family'  : 'sans-serif',
   'font.sans-serif':'Helvetica',
   'axes.axisbelow': True
   }
plt.rcParams.update(params)
pal = ['#253494','#2c7fb8' ,'#41b6c4','#a1dab4']  # Color Blind Friendly
#pal = ['#cccccc','#525252'] # Color Blind eFriendly
pal = sns.color_palette(pal)

df_plot = df[df.event.isin(['LP-13','LP-16','LP-18'])]

for f in ['FC','ENT']:
        df_plot[f+'map'] =df_plot[f+'_cat'].map({0:'Low',1:'Medium',2:'High',3:'Very High'})  # Cat
        #df_var[f+'map'] =df_var[f+'_exc'].map({0: 'Below SSS', 1: 'Above SSS'})   # Exc


for f in ['ENT']:
    plt.figure(figsize=(9,12))
    c=1
    for v in var:
        plt.subplot(3,3,c)
        ax = sns.swarmplot(x='event',y=v,hue=f+'map',data=df_plot,size=4, palette=pal)
        plt.ylabel('')
        plt.title(var[v]['label'])
        plt.xlabel('')
        plt.ylim(var[v]['ymin'],var[v]['ymax'])
        ax.tick_params(right=True)
        plt.legend('',frameon=False)
        c+=1
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15)
#%%Rainfall
# Number of wet days
trad.groupby(['event','wet3']).count()['tide'].dropna()
#wet3: 0 for HF events, 33/353 for trad (2008-2016)
df.groupby(['event','wet7']).count()['tide'].dropna()
#wet7: HF18 only, 78/353 for trad

#Exceedances by wet3 (Wet weather in previous 3 days)
hf.groupby(['beach','wet3','ENT_exc']).count()['tide'].dropna()
df.groupby(['event','wet3','FC_exc']).count()['tide'].dropna()
#Results: All HF events on dry days (by wet 3 standards)
# Majority of trad sample exceedances occur on dry day (could be skewed by season)

#Exceedances by wet7 (Wet weather in previous 7 days)
df.groupby(['event','wet7','ENT_exc']).count()['tide'].dropna()
df.groupby(['event','wet7','FC_exc']).count()['tide'].dropna()
# Results: HF18 is a wet day by wet7 standards
# Trad ENT total exceedences more distributed between wet and dry days, but not FC

# For both wet3 and wet7, a larger percentage of the sample taken on wet days were exceedances
# (though most exceedances and samples were dry days)
# Rain might not be super important at LP

plt.figure()
plt.subplot(1,2,1)
sns.boxplot(x='wet3', y='logENT', data=trad, palette='RdBu')
sns.swarmplot(x='wet3',y='logENT',data=trad, color='k',size=4)
plt.xlabel('')
ticks, lab_temp = plt.xticks()
plt.xticks(ticks=ticks, labels=['Dry','Wet'])
plt.axhline(y=np.log10(104), color='k',ls='--')
plt.subplot(1,2,2)
sns.boxplot(x='wet3', y='logFC', data=trad, palette='RdBu')
sns.swarmplot(x='wet3',y='logFC',data=trad, color='k',size=4)
plt.axhline(y=np.log10(400), color='k',ls='--')
plt.xlabel('')
plt.suptitle('FIB by Wet/Dry Days (3 Day Threshold)\nRoutine Monitoring (2008-2016)')
plt.xticks(ticks=ticks, labels=['Dry','Wet'])

plt.figure()
plt.subplot(1,2,1)
sns.boxplot(x='wet7', y='logENT', data=trad, palette='RdBu')
sns.swarmplot(x='wet7',y='logENT',data=trad, color='k',size=4)
plt.xlabel('')
ticks, lab_temp = plt.xticks()
plt.xticks(ticks=ticks, labels=['Dry','Wet'])
plt.axhline(y=np.log10(104), color='k',ls='--')
plt.subplot(1,2,2)
sns.boxplot(x='wet7', y='logFC', data=trad, palette='RdBu')
sns.swarmplot(x='wet7',y='logFC',data=trad, color='k',size=4)
plt.axhline(y=np.log10(400), color='k',ls='--')
plt.xlabel('')
plt.suptitle('FIB by Wet/Dry Days (7 Day Threshold)\nRoutine Monitoring (2008-2016)')
plt.xticks(ticks=ticks, labels=['Dry','Wet'])
