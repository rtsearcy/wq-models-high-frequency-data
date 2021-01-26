# wq-models-high-frequency-data

Develop predictive models for water quality at your beach using high-frequency (HF) sample data

The scripts in the repository are what I used when aggregating data, exploring trends, and developing and testing models for Searcy and Boehm, 2021. "A Day at the Beach: Enabling Coastal Water Quality Prediction with High-Frequency Sampling and Data-Driven Models" (DOI: https://doi.org/10.1021/acs.est.0c06742).

These should be useful to help get you started in designing your own predictive water quality modeling system. Eventually, this repository will contain a complete package that will enable users to build and test models from scratch.

For questions, please reach out to me at rtsearcy@stanford.edu

Best of luck,

Ryan Searcy
January 2021

- - - 
## PYTHON SCRIPTS
*** signifies important scripts

### COLLECT DATA
Scripts to grab and process data from:

- CDIP (cdip.ucsd.edu, wave data)
- NOAA CO-OPS (https://tidesandcurrents.noaa.gov, tide and met data)
- NCDC (https://www.ncdc.noaa.gov/, met data)
- CIMIS (https://cimis.water.ca.gov/, met data)

### EXPLORATORY DATA ANALYSIS (EDA)
1. HF_EDA_dataset_combo.py
Aggregate FIB and environmental data into a single dataframe

2. HF_compare_beaches_bin_plots.py ***
Plots and some stats for EDA on the HF sampling and environmental data

3. HF_EV_stats_plots.py
Bin analysis, correlations, boxplots for environmental data

4. HF_EDA_FIB_stats.py ***
Statistics related to the FIB variability for all HF events

### MODEL DEVELOPMENT
1. wq_modeling.py ***
Package of functions to create statistical water quality nowcast models from FIB and 
environmental data.

2. HF_models.py
Functions that tests all model types on an input test case

3. HF_predictive_models.py
Script to develop and test individual models

4. HF_model_all.py  ***
Script to iterate through all HF test cases and develop and test models

### MODEL TESTING
1. HF_analyze_all_models.py ***
Script to aggregate modeling results to summarize

2. HF_model_predictions_obs.py
Script to plot predictions and observations for specific models

