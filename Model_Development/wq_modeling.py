#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wq_modeling.py

Created on Thu Dec 19 10:03:41 2019
@author: rtsearcy

Package of functions to create statistical water quality nowcast models from FIB and 
environmental data. Functions include:
    
    - fib_thresh: FIB exceedance thresholds
    - load_data: Load modeling datasets
    - clean: Cleans modeling datasets by imputing missing values**, and removing missing 
      rows and columns
    - parse: Parse data into training and test subsets
    - pred_eval: Evaluates predictions statistics 
    - select_vars: Selects best variables for modeling from dataset
    - current_method: Computes performance metrics for the current method
    - multicollinearity_check: Check VIF of model variables, drop if any above threshold
    - check_corr: Checks high correlation between modeling variables against FIB
    - fit: Fit model on training set
    - tune: tunes regression models to a certain set of performance standards
    - test: TBD
    - save: Saves model fits, performance, coefficients

TODO List:
TODO - clean: impute, extreme values
TODO - fit: add more model types (ann, RF) or create seperate functions
TODO - test: add testing function


"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, roc_curve, r2_score
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import os
import sys

# %% DATA
default_no_model = ['sample_time', 'TC', 'FC', 'ENT', 'TC1', 'FC1', 'ENT1', 
            'TC_exc', 'FC_exc', 'ENT_exc', 'TC1_exc','FC1_exc', 'ENT1_exc', 
            'logTC1', 'logFC1', 'logENT1', # previous sample typically not included
            'wet','rad_1h', 'rad_2h', 'rad_3h', 'rad_4h', 
            'MWD','MWD1','MWD1_b','MWD1_b_max', 'MWD1_b_min']

# Model Performance criteria
default_perf_criteria = {  
    'sens_min': 0.3,  # Model sensitivity must be at least this
    'sens_plus_cm': 0.1,  # Model sensitivity must also be at least this much greater than the current method
    'spec_min': 0.85  # Model specificity must be at least this
}


# %% FIB thresh
def fib_thresh(f):
    '''
    FIB exceedance thresholds for TC, FC/E. coli, ENT
    
    Parameters:
        - f = 'TC', 'FC', or 'ENT'
        
    Output:
        - integer of the exceedance threshold
        
    '''
    
    assert f in ['TC','FC','ENT'], 'FIB type must be either TC, FC, or ENT'
    if f == 'TC':
        return 10000
    elif f == 'FC':
        return 400
    elif f == 'ENT':
        return 104
        
    
# %% Load Data
def load_data(file, years=[1991], season='a'):
    '''
    Load modeling dataset and returns a Pandas DataFrame.
    - Set datetime index
    - Remove undesired modeling years
    - Prints FIB statistics in the dataset
    
    Parameters:
        - file = full file path with modeling dataset. Must:
                - BE A .CSV FILE
                - Contain a datetime column named 'dt' for the index
                - Contain FIB data (TC,FC, and/or ENT samples)
                
        - years = list of length 1 or 2 containing range of integer years to be loaded
                - No need to define for High Frequency sampling (i.e. one day - less than 
                  one year of data)
        - season = 'a', 's', or 'w', if data is to be modeled for a specific season
                - 'a' - All data/do not split by season       
                - 's' - Summer data only (Apr-Oct)
                - 'w' - Winter data only (Nov-Mar)
                
    Output:
        - df = Pandas DataFrame containing modeling dataset with a sorted datetime index
        
    '''
    
    assert type(years) == list, 'years paramater must be a list of size 1 or 2'
    assert years[0] in range(1991,2100), 'years parameter list must contain an integer year greater than 1991'
    if len(years) == 2:
        assert years[1] in range(1991,2100) and years[1] >= years[0], 'second years parameter must be an integer year >= to the first years parameter'
    assert season in ['a','s','w'], 'season paramter must be either \'a\', \'s\', or \'w\''
    
    df = pd.read_csv(file)
    assert 'dt' in df.columns, 'No datetime column named \'dt\' found'
    df['dt'] = pd.to_datetime(df['dt'])
    df.set_index('dt', inplace=True)
    # Sort data into ascending time index (Earliest sample first)
    df.sort_index(ascending=True, inplace=True) 
    
    # Remove years not in desired range
    df = df[df.index.year >= years[0]] # start year
    if len(years) == 2:
        df = df[df.index.year <= years[1]]
    
    # If seasonal, remove other season data
    if season == 's':
        df = df[(df.index.month >= 4) & (df.index.month < 11)]
    elif season == 'w':
        df = df[(df.index.month <= 3) | (df.index.month >= 11)]
        
    # TODO Remove rows with missing data/IMPUTE OR CREATE CLEAN FUNCTION
    
    # FIB Statistics and Plots
    print('\n- - | FIB Statistics | - -\n')
    print('Dataset: ' + file)
    print('\nStart Year: ' + str(df.index.year[0]) + '; End Year: ' + str(df.index.year[-1]))        
    if season == 's':
        print('Season: Summer')
    elif season == 'w':
        print('Season: Winter')
    else: print('Season: All Data')
    print('Number of Samples: ' + str(df.shape[0]))
    
    fib = []
    for f in ['TC','FC','ENT']:
        fib.append(f) if f in df.columns else print(f + ' not in dataset')
    assert len(fib) > 1, '- - No FIB data in this dataset --'
    print(df[fib].describe())
    
    # Previous FIB / Exceedances / Log10 Transform
    print('\nExceedances:')
    # fib_thresh = {'TC': 10000, 'FC': 400, 'ENT': 104}
    for f in fib:
        if f + '1' not in df.columns: # Previous FIB sample
            df[f + '1'] = df[f].shift(1)
        if f + '_exc' not in df.columns: # Exceedance variables
            df[f + '_exc'] = (df[f] > fib_thresh(f)).astype(int)
            df[f + '1_exc'] = (df[f+'1'] > fib_thresh(f)).astype(int)
        print(f + ' : ' + str(sum(df[f + '_exc'])))
        if 'log' + f not in df.columns: # log10 transformed FIB variables
            df['log' + f] = np.log10(df[f])
            df['log' + f + '1'] = np.log10(df[f + '1'])
            
    print('\nNumber of Columns: ' + str(df.shape[1]))
    
    return df


#%% Clean
def clean(df, percent_to_drop=0.05, save_vars=[]):
    '''
    Cleans modeling datasets by imputing missing values, and removing missing 
    rows and columns
    
    Parameters:
        - df = Input dataframe (result from load_data)
        
        - percent_to_drop = Fraction of total rows allowed of missing values in a variable
          before it is dropped
          
        - save_vars = List of variables to keep in modeling dataset despite missing values
        
    Output:
        - df_out = Cleaned Dataframe without missing values (Ready for modeling)
    '''
    assert type(save_vars) == list, 'save_vars paramater must be a list containing variable names'
    assert type(percent_to_drop) == float and 0 <= percent_to_drop < 1.0, 'percent_to_drop must be a fraction value'
    
    print('\n\n- - | Dataset Cleaning | - -\n')
    # Extreme Values
    # Check for errors in 'dirty' data (range)
#    for c in df_dirty.columns:
#        if c in df_range.index:  # Out of range = NAN
#            df_dirty[c][~df_dirty[c].between(df_range.loc[c]['lower'], df_range.loc[c]['upper'])] = nan
    
    # Missing Values
    print('- Missing Values -')
    if (df.isna().sum() > 0).sum() > 0:  # If there are missing values
        num_allow = int(len(df) * percent_to_drop) # num_allow - number of allowable missing points before drop
        print('Number of missing data points allowed before variable drop: ' + str(num_allow))
        miss_list = [x for x in df.columns if df[x].isna().sum() > num_allow] # Variables with missing data
        miss_list = [x for x in miss_list if all(f not in x for f in ['TC','FC','ENT'])] # Except FIB vars
        if len(save_vars) > 0:
            miss_list = [x for x in miss_list if x not in save_vars] # Vars to keep
        new_cols = [x for x in df.columns if x not in miss_list]
        df = df[new_cols]  # Drop 'miss_list' variables from dirty dataset
        if len(miss_list) > 0:
            print('Dropped variables (' + str(len(miss_list))+ '):\n')
            print(miss_list)
        else: print('* No variables dropped *')
    
        # Drop remaining rows with missing values
        df_out = df.dropna()  # Drop rows with missing values
        print('\nDropped rows: ' + str(len(df) - len(df_out)))
    else:
        print('* No missing values in dataset *')
        df_out = df
    print('\n- Clean dataset -\nRows: ' + str(len(df_out)))
    print('Columns: ' + str(len(df_out.columns)))
    return df_out


#%% Parse
def parse(df, fib='ENT', parse_type='c', test_percentage=0.3, save_dir=None):
    '''
    Parse dataset into training and test subset to be used for model fitting
    and evaluation.
    
    Parameters:
        - df = DataFrame containing modeling dataset. Must run load_dataset function first
        
        - season = 'a', 's', or 'w', if data is to be modeled for a specific season
                - 'a' - All data/do not split by season       
                - 's' - Summer data only (Apr-Oct)
                - 'w' - Winter data only (Nov-Mar)
        
        - fib = FIB type to be modeled (TC, FC, or ENT). Default ENT
        
        - parse_type = 'c' or 'j' (Chronological or Jackknife methods, respectively)
            - 'c': Splits dataset chronologically (i.e. the last xx years/% of data into 
              the test set; the remaining into the training set)
            - 'j': Splits dataset using a random xx% of data for the test subset, and the
              remaining for the training subset
              
        - test_percentage = Percent of dataset that will go into the test subset
            - If parse_type = 'c', then the nearest whole year/season will be used
            
        - save_dir = name of the directory for the training and testing subsets to be saved
            - Will create a new directory if it doesn't exist
            
    Output:
        - y_train, X_train, y_test, X_test = Training and test subsets
            - y - Pandas Series, X - Pandas Dataframes
            - y and X DFs have matching indices
        
    '''
    
    # assert statements
    assert fib in ['TC','FC','ENT'], 'FIB type must be either TC, FC, or ENT'
    assert parse_type in ['c','j'], 'parse_type must be either \'c\' or \'j\''
    assert type(test_percentage) == float and 0 <= test_percentage < 1.0, 'test_percentage must be a fraction value'
    
    # Remove other FIB variables
    print('\n\n- - | Parsing Dataset | - -')
    other_fib = [x for x in ['TC','FC','ENT'] if x != fib]
    cols = df.columns
    for i in range(0, len(other_fib)):
        cols = [x for x in cols if other_fib[i] not in x]
    df = df[cols]  
    # TODO - check for previous FIB and logFIB vars, add if not there
    
    print('FIB: ' + fib)
    df.sort_index(ascending=True, inplace=True) # Sort data into ascending time index
    # Split
    if parse_type == 'c':
        if len(np.unique(df.index.year)) == 1: #If High freq. data or less than one year
            num_test_sam = int(len(df)*test_percentage)
            test_data = df.iloc[-num_test_sam:]  # Select last % of samples for test set
        else:
            test_yr_e = str(max(df.index.year))  # End year
            num_test_yrs = round(test_percentage * (max(df.index.year) - min(df.index.year)))
            if any(x in np.unique(df.index.month)for x in [1,2,3,11,12]) & any(x not in np.unique(df.index.month)for x in [4,5,6,7,8,9,10]):
                test_yr_s = str(int(test_yr_e) - num_test_yrs)  # Start year
                temp_test = df[test_yr_s:test_yr_e]
                test_data = temp_test[~((temp_test.index.month.isin([1, 2, 3])) &
                                        (temp_test.index.year.isin([test_yr_s])))].sort_index(ascending=True)
            # Ensure winter seasons (which cross over years) are bundled together   
            else:
                test_yr_s = str(int(test_yr_e) - num_test_yrs + 1)  # Start year
                test_data = df[test_yr_s:test_yr_e].sort_index(ascending=True)  
        #Set remaining data to training subset    
        train_data = df[~df.index.isin(test_data.index)].sort_index(ascending=True)
    
        y_test = test_data['log' + fib]
        X_test = test_data.drop('log' + fib, axis=1)
        y_train = train_data['log' + fib]
        X_train = train_data.drop('log' + fib, axis=1)
        print('Parse Method: Chronological')
        print('   Training Set: ' + str(min(y_train.index.year)) + ' - '
              + str(max(y_train.index.year)))
        print('   Test Set: ' + str(min(y_test.index.year)) + ' - '
              + str(max(y_test.index.year)))
    
    else:
        y = df['log' + fib]  # Separate into mother dependent and independent datasets
        X = df.drop('log' + fib, axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage,
                                                            random_state=0)
        print('Parse Method: Jackknife')
        print('   Test Set Percentage: ' + str(test_percentage*100) + '%')
    
    # Account for NA samples
    y_test = y_test.dropna()
    X_test = X_test.reindex(y_test.index)
    y_train = y_train.dropna()
    X_train = X_train.reindex(y_train.index)
    
    # Check Exceedances
    train_exc = X_train[fib + '_exc'].sum() # Assune 'FIB_exc' vars previously calculated
    test_exc = X_test[fib + '_exc'].sum()
    print('\nTraining (calibration) subset:\n' + '  Samples - ' + str(len(X_train))  
    + '\n  Exc. - ' + str(train_exc) + ' (' + str(round(100*train_exc/len(X_train),1)) + '%)')
    print('\nTest (validation) subset:\n' + '  Samples - ' + str(len(X_test)) 
    + '\n  Exc. - ' + str(test_exc)+ ' (' + str(round(100*test_exc/len(X_test),1)) + '%)')
    
    print('\nNumber of Variables: ' + str(X_train.shape[1]))
    
    # Save train and test sets seperately
    if save_dir != None:
        try:
            os.makedirs(save_dir, exist_ok=True)  # Create dir if doesn't exist
            train_fname = 'training_subset_' + fib + '_' + parse_type + '_' + str(min(df.index.year)) + '_' + str(max(df.index.year)) + '.csv' # Training set filename
            test_fname = train_fname.replace('training', 'test') # Test set filename
            test_data = y_test.to_frame().merge(X_test, left_index=True, right_index=True)
            train_data = y_train.to_frame().merge(X_train, left_index=True, right_index=True)
            train_data.to_csv(os.path.join(save_dir, train_fname))
            test_data.to_csv(os.path.join(save_dir, test_fname))
            print('\nTraining and test subsets saved to: \n' + save_dir)
            
        except Exception as exc:
            print('\nERROR (There was a problem saving the parsed files: %s)' % exc)
            # continue

    return y_train, X_train, y_test, X_test
    

# %% Pred eval
def pred_eval(true, predicted, thresh=0.5, tune=False):  # Evaluate Model Predictions
    '''
    Evaluates model sensitivity, specificity, and accuracy for a given set of predictions
    
    Parameters:
        - true = Pandas or Numpy Series of true values
        
        - predicted = Pandas or Numpy Series of model predictions
        
        - thresh = threshold above which a positive outcome is predicted (i.e. FIB 
          exceedance)
        
        - tune = True if model tuning (see below function)
        
    Output:
        - out = Dictionary of performance statistics
            - Sensitivity (True Positive Rate)
            - Specificity (True Negative Rate)
            - Accuracy (Total correctly Predicted)
            - Samples
            - Exceedances
    '''
#    if true.dtype == 'float':
#        true = (true > thresh).astype(int)  # Convert to binary
#    if predicted.dtype == 'float':
#        predicted = (predicted > thresh).astype(int)
        
#    true = (true > thresh).astype(int)  # Convert to binary
#    predicted = (predicted > thresh).astype(int)

#    cm = confusion_matrix(true, predicted)  
#    # Lists number of true positives, true negatives,false pos,and false negs.
#    sens = cm[1, 1] / (cm[1, 1] + cm[1, 0])  # sensitivity - TP / TP + FN
#    spec = cm[0, 0] / (cm[0, 1] + cm[0, 0])  # specificity - TN / TN + FP
#    acc = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[1, 1] + cm[0, 1] + cm[1, 0])
    
    samples = len(true)  # number of samples
    exc = (true>thresh).sum()  # number of exceedances
    
    tp = np.sum((true > thresh) & (predicted > thresh))  # True positives
    tn = np.sum((true < thresh) & (predicted < thresh))  # True negatives
    fp = np.sum((true < thresh) & (predicted > thresh))  # False positives
    fn = np.sum((true > thresh) & (predicted < thresh))  # False negative

    sens = tp / (tp + fn)  # Sensitivity
    spec = tn / (tn + fp)  # Specificity
    acc = (tn + tp) / samples  # Accuracy

    if tune == False:
        out = {'Sensitivity': round(sens, 3), 'Specificity': round(spec, 3), 
               'Accuracy': round(acc, 3), 'Samples': samples, 'Exceedances': exc}
    else:
        out = [round(sens, 3), round(spec, 3)]

    return out


# %% CHECK CORRELATION
def check_corr(dep, ind, thresh=0.5):
    '''
    Check if confounding variables have correlations > thresh, and drop the one with 
    least correlation to the dependnet variable
    
    Parameters:
        - dep - Pandas Series of the dependant variable
        
        - ind - Dataset (Pandas DF) containing modeling variables to be checked against the dependent
          variables
        
        - thresh - Threshold for Pearson Correlation Coefficient between two variables
          above which the least correlated variable to FIB will be dropped
          
    Output:
        - DataFrame with the best correlated variables included
    
    '''
    print('\nChecking variable correlations against threshold (PCC > ' + str(thresh) + '): ')
    c = ind.corr()  # Pearson correlation coefs.
    to_drop = []

    for ii in c.columns:  # iterate through all variables in correlation matrix except dependant variable
        temp = c.loc[ii]
        temp = temp[temp.abs() > thresh]  # .5 removed a lot of variables
        temp = temp.drop(ii, errors='ignore')  # Remove the variable itself
        i_corr = dep.corr(ind[ii])
        if len(temp) > 0:
            for j in temp.index:
                j_corr = dep.corr(ind[j])
                if ii not in to_drop and abs(i_corr) < abs(j_corr):  # Drop variable if its corr. with logFIB is lower
                    to_drop.append(ii)

    print('  Variables dropped - ' + str(len(to_drop)))
    print(to_drop)
    ind = ind.drop(to_drop, axis=1, errors='ignore')  # Drop variables
    print('Remaining variables - ' + str(len(ind.columns) - 1))
    print(ind.columns.values)
    return ind


#%% CHECK FOR MULTICOLLINEARITY
def multicollinearity_check(X, thr=5):  
    '''
    Check VIF of model variables, drop if any above threshold
    
    Parameters:
        - X = Variable dataset
        
        - thr = threshold VIF maximum
        
    Output:
        - Dataset with no multicolinear variables
        
    '''
    variables = list(X.columns)
    print('\nChecking multicollinearity of ' + str(len(variables)) + ' variables for VIF:')
    if len(variables) > 1:
        vif_model = LinearRegression()
        v = [1 / (1 - (r2_score(X[ix], vif_model.fit(X[variables].drop(ix, axis=1), X[ix]).
                                predict(X[variables].drop(ix, axis=1))))) for ix in variables]
        maxloc = v.index(max(v))  # Drop max VIF var if above 'thr'
        if max(v) > thr:
            print(' Dropped: ' + X[variables].columns[maxloc] + ' (VIF - ' + str(round(max(v), 3)) + ')')
            variables.pop(maxloc)  # remove variable with maximum VIF
        else:
            print(' VIFs for all variables less than ' + str(thr))
        X = X[[i for i in variables]]
        return X
    else:
        return X


# %% Select Vars
def select_vars(y, X, method=None, no_model=default_no_model, corr_thresh=0.5, vif=5):
    '''
    Selects best variables for modeling from dataset.
    
    Parameters:
        - y = Dependant variable
        
        - X = Independant variables
        
        - method = Variable selection method
            - 'lasso' - Lasso Regressions - insignificant variables automatically 
               assigned 0 coefficient
            - 'rfe' - Recursive Feature Elimination - Selects best features
            - 'forest' - Random Forest
        
        - no_model = variables to exclude from modeling prior to analysis
        
        - corr_thresh = Threshold for Pearson Correlation Coefficient between two variables
          above which the least correlated variable to FIB will be dropped
            - If 'False' or 0 -> Correlation analysis will not be performed
            
        - vif = Maximum VIF for multicollinearity check
        
    Output:
        - Dataset with the best variables to use for modeling
    '''
    assert type(no_model) == list, 'no_model parameter must be a list of variables to exclude from modeling'
    print('\n\n- - | Selecting Variables | - -')
    print('\nOriginal # of Variables - ' + str(len(X.columns)))
    
    # Drop variables NOT to be modeled
    if len(no_model) > 0:
        print('\nAutomatically dropped variables - ' + str(len(no_model)))
        print(no_model)
        to_model = [x for x in X.columns if x not in no_model]  # Drop excluded variables
        X = X[to_model]
    
    # Check similarly correlated vars to FIB 
    if corr_thresh > 0:
        X = check_corr(y, X, thresh=corr_thresh)  
        
    # Select variables
    print('\nVariable Selection Method: ' + method.upper() + '\n')
    multi=True
    c=0
    while multi:
        if method == 'lasso':  # LASSO
            lm = LassoCV(cv=5, normalize=True).fit(X, y)
            new_vars = list(X.columns[list(np.where(lm.coef_)[0])]) # vars Lasso kept
            assert len(new_vars) > 0, 'Lasso regression failed to select any variables'
            X = X[new_vars]
        elif (method == 'forest') & (c==0): # Random Forest
        # Ref: Jones et al 2013 - Hydrometeorological variables predict fecal indicator bacteria densities in freshwater: data-driven methods for variable selection
            # Only run once
            rf = RandomForestRegressor(n_estimators=500, 
                                       oob_score=True, 
                                       random_state=0, 
                                       #max_samples=.75,
                                       max_features=.75
                                       )
            #Xs = X
            # Scale by mean and std dev
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            
            rf.fit(Xs,y)
            temp = permutation_importance(rf,Xs,y,random_state=0, n_repeats=10)['importances_mean']
            temp = pd.Series(data=temp, index=X.columns).sort_values(ascending=False)
            print('  Mean Importance: ' + str(round(temp.mean(),3)))
            #new_vars = list(temp.index[0:10])
            new_vars = list(temp[temp>1.5*temp.mean()].index)  # Select the variables > 1.25 of th emean importance
            assert len(new_vars) > 0, 'Random Forest Regression failed to select any variables'
            c+=1
            X = X[new_vars]
            print('  Out of Bag R-sq: ' + str(round(rf.oob_score_,3)))
                          
        elif method == 'rfe':  # Recursive Feature Elimination w. Linear Regression
            # Credit: https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
            low_score=1*10**6
            nof=0           
            for n in range(1,10):
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
                lm = LinearRegression(normalize=True)
                rfe = RFE(lm, n)
                X_train_rfe = rfe.fit_transform(X_tr,y_tr)
                X_test_rfe = rfe.transform(X_te)
                lm.fit(X_train_rfe,y_tr)
                score = ((((lm.predict(X_test_rfe) - y_te)**2).sum())**0.5) / len(y_te)  # RMSE
                #score = lm.score(X_test_rfe,y_te)  # R2 of prediction
                #print(str(score))
                if(score<low_score):
                    low_score = score
                    nof = n
            print("Optimum number of features: %d" %nof)
            print("R2 with %d features: %f" % (nof, low_score))
            
            lm = LinearRegression(normalize=True)
            S = RFE(lm,nof).fit(X, y)
            new_vars = list(X.columns[list(np.where(S.support_)[0])])
            X = X[new_vars]
             
        # Check VIF
        X_multi = multicollinearity_check(X, thr=vif)
        if len(X_multi.columns) == len(X.columns):
            multi = False
        else:
            X = X_multi
    
    print('\nFinal Variables Selected - ' + str(len(X.columns)))
    print(X.columns.values)    
    return X
    

# %% CURRENT METHOD
def current_method(df, fib='ENT'):
    '''
    Computes prediction metrics for the current method / persistence method:
        FIB = FIB1 [Predicted FIB = FIB from previous sample]
        
    Parameters:
        - df = DataFrame containing FIB data (must have current and previous sample 
          variables)
        
        - fib = 'TC', 'FC', or 'ENT'
        
    Output:
        - Dictionary containing the performance metrics for the current method of the
          dataset. Returns NoneType if the function cannot find FIB data
    '''
    if all(f in df.columns for f in [fib, fib + '1']):
        return pred_eval(df[fib], df[fib+'1'], thresh=fib_thresh(fib))
    elif all(f in df.columns for f in ['log' + fib, 'log' + fib + '1']):
        return pred_eval(df['log'+fib], df['log'+fib+'1'], thresh=np.log10(fib_thresh(fib)))
    elif all(f in df.columns for f in [fib + '_exc', fib + '1_exc']):
        return pred_eval(df[fib+'_exc'], df[fib+'1_exc'])
    else:
        print('Cannot compute the current method performance for this dataset')
        return


#%% FIT MODEL
def fit(y_train, X_train, model_type='mlr'):
    '''
    Fits predictive FIB model to dataset
    
    Parameters:
        - y_train = Pandas Series of log10-transformed FIB values
        
        - X_train = Pandas DataFrame of modeling variables, including logFIB1 (previous FIB)
        
        - model_type = Type of model to be fit
            - 'mlr': Multivariate linear regression
            - 'lasso': Lasso regression (used for variable selection**)
            - 'blr': Binary logistic regression
            - 'rf': Random forest
            - 'nn': Neural network
            
        - cm = Evaluate current method (Boolean)
    
    Output:
        - model = XX
        
        - df_perf = DataFrame() of model performance (sensitivity, specificity) for model
          and current method, if chosen
        
    '''
    assert model_type in ['mlr','blr','rf','nn'], 'model_type must be one of: \'mlr\',\'blr\',\'rf\',\'nn\''
    # Find FIB from y_train name or in X_train vars
    f = [f for f in ['TC','FC','ENT'] if f in y_train.name or any(f in x for x in X_train.columns)][0]
    
    print('\n\n- - | Fitting ' + f + ' Model (' + model_type.upper() + ') | - -')
    
    # Performance Dataframe Init (for printing)
    cols_perf = ['Sensitivity', 'Specificity', 'Accuracy', 'Exceedances', 'Samples']
    df_perf = pd.DataFrame(columns = ['Model'] + cols_perf)
    df_perf = df_perf.set_index('Model') 
    
    # TODO - Need to keep track of variables 
    
    if model_type == 'mlr':
    # Linear Regression
        t = np.log10(fib_thresh(f))
        model = LinearRegression()
        model.fit(X_train,y_train)
        coef = model.coef_
        intercept = model.intercept_
    
    elif model_type == 'blr':
        t = 0.5
        y_train = (y_train > np.log10(fib_thresh(f))).astype(int)  # To binary
        model = LogisticRegression(random_state=0, C=0.1, solver='lbfgs')
        model.fit(X_train, y_train)
        coef = model.coef_[0]
        intercept = model.intercept_[0]
        
    #TODO Logistic Regression
    
    #TODO Random Forests
    
    #TODO Neural Networks
    
    # Fit
    if model_type in ['mlr','blr']:
        print('\nModel Fit:')
        print('Variable - Coefficient')
        print('- - - - - - - - - - - - - -')
        for x in range(0, len(X_train.columns)):
            print(X_train.columns[x] + ' -   '  + str(round(coef[x],4)))
        print('Intercept -   ' + str(round(intercept,4)))
    
    # Model Performance
    
    model_perf = pred_eval(y_train, model.predict(X_train), thresh=t)
    df_perf = df_perf.append(pd.DataFrame(model_perf, index=[model_type.upper()]),sort=False)
    df_perf = df_perf[cols_perf]
    
    print('\nModel Performance:\n')
    print(df_perf)
    
    return model, df_perf
    
    
#%% TUNE MODEL (MLR, BLR)
def tune(y, X, model, cm_perf, perf_criteria=default_perf_criteria):
    '''
    Tune MLR or BLR model (using the calibration dataset) to acheive 
    prediction performnance standards. Tuned models are referred to as MLR-T and BLR-T
    
    Parameters:
        - y = Pandas Series of log10-transformed FIB values (Calibration)
        
        - X = Calibration modeling dataset
        
        - model = Fit sklearn model (LinearRegression or LogisticRegression only)
        
        - cm_perf = Dictionary of Current Method performance metrics in the Calibration
          dataset
        
        - perf_criteria = Dictionary of Sensitivity and Specificity metrics to 
          tune model to
          
    Output:
        - tuning_parameter = Multiplier to attach to predictions for MLR-T/BLR-T output
    '''
    # Find FIB from y_train name or in X_train vars
    f = [f for f in ['TC','FC','ENT'] if f in y.name or any(f in x for x in X.columns)][0]
    t = np.log10(fib_thresh(f))  # exceedance threshold
    
    if 'LogisticRegression' in str(type(model)):
        model_type = 'blr'
        print('\n- - | Tuning ' + f + ' Model (' + model_type.upper() + ') | - -')
        y = (y > t).astype(int)
        y_pred = model.predict_proba(X)[:, 1]  # Prob. of a post in calibration
        tune_st = round(max(y_pred),4)  # tuning start index
        tune_range = np.arange(tune_st, 0, -0.0001)
        sens_spec = np.array([pred_eval(y, (y_pred >= j).astype(int), thresh=0.5, tune=True) for j in tune_range])
        
    elif 'LinearRegression' in str(type(model)):
        model_type = 'mlr'
        print('\n- - | Tuning ' + f + ' Model (' + model_type.upper() + ') | - -')
        y_pred = model.predict(X)  # Prediction
        tune_range = np.arange(0.7, 2.25, 0.001)
        sens_spec = np.array([pred_eval(y, (y_pred * j), thresh=t, tune=True) for j in tune_range])
        
    T = np.column_stack((sens_spec, tune_range))
    # Find all tuning factors (PM) that enable model to meet performance criteria
    meets_criteria_t = (T[:, 0] > perf_criteria['sens_min']) & \
                       (T[:, 0] > perf_criteria['sens_plus_cm'] + cm_perf['Sensitivity']) & \
                       (T[:, 1] >= min(perf_criteria['spec_min'], cm_perf['Specificity']))
    
    U = T[meets_criteria_t]

    if U.size == 0:
        print('\n* * * No tuning available that passes model performance criteria for training set * * *\n')
        return np.nan
    else:
        tuning_parameter = round(U[0, -1],4)
        print('Tuning Parameter = ' + str(tuning_parameter))
        return tuning_parameter  # Select most conservative passing model (minimum sensitivity in order to max specificity)
    

#%% TEST MODEL
    
#%% SAVE MODEL, DATASETS, PERFORMANCE
def save_model(folder, name, model = None, df_coef = None, df_perf = None):
    '''
    Saves model pickle file, model coefficients, and/or model performance
    
    Parameters:
        - folder = Save directory
        
        - name = Desired name for modeling files (string)
        
        - model = Model fit
        
        - df_coef = Model variables and coefficients (if applicable)
        
        - df_perf = DataFrame of model performance
        
    '''
    print('Save Directory: ' + folder)
    # Save Model Fit
    if model is not None:
        if 'LinearRegression' in str(type(model)):
            model_type = 'MLR-T'
        elif 'LogisticRegression' in str(type(model)):
            model_type = 'BLR-T'
        model.coef_ = model.coef_[model.coef_ != 0]  # .reshape(1, -1)  # Drop zero-coefficients
        model_file = 'model_' + name.replace(' ', '_') + '_' + model_type + '.pkl'
        joblib.dump(model, os.path.join(folder, model_file))
        print('  Model fit saved to: ' + model_file )
        # use joblib.load to load this file in the model runs script
    
    # Save Coefficients
    if df_coef is not None:
        df_coef = df_coef[abs(df_coef) > 0]  # Drop zero coefficients
        coef_file = 'coefficients_' + name.replace(' ', '_') + '_' + model_type + '.csv'
        df_coef.to_csv(os.path.join(folder, coef_file))
        print('  Model coefficients saved to: ' + coef_file )
    
    # Save Performance
    if df_perf is not None:
        perf_file = 'performance_' + name.replace(' ', '_') + '_' + model_type + '.csv'
        df_perf.to_csv(os.path.join(folder, perf_file), float_format='%.3f')
        print('Model performance saved to: ' + perf_file )
