# -*- coding: utf-8 -*-
"""
Project SDA: 
Do RoBERTa and VADER sentiment models on r/Bitcoin help to explain Bitcoin retuns?

Part 6 - Fitting OLS and Random Forest Models


"""
# In case it is not installed yet, uncomment:
# pip install graphviz


import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt
import graphviz


from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor
from statsmodels.compat import lzip
from sklearn.tree import export_graphviz
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller

# This code consists of the following parts :
# 1) Load and merge Return Data and Sentiment Data and subsequently, define 
#    the number of train and test observation
# 2) Define a function for the OLS model and apply it  using VADER scores,
#    RoBERTa scores and S&P500 returns 
# 3) Use a Granger causality test to determine whether lagged values of 
#    Sentiment Scores help explain BTC returns.
# 4) Define a function to run Random Forest models, again using VADER scores,
#    RoBERTa scores and S&P500 returns



# 1) Load and merge the Data

return_data = pd.read_csv('Data/BTC_merged.csv', index_col=0)
sent20 = pd.read_csv('Data/VADER_ROBERTA_Daily_Results_2020_01_01__2020_12_31_aggregated.csv', 
                     index_col=0)
sent21_1 = pd.read_csv('Data/VADER_ROBERTA_Daily_Results_2021_01_01__2021_02_04_aggregated.csv',
                       index_col=0)
sent21_2 = pd.read_csv('Data/VADER_ROBERTA_Daily_Results_2021_02_07__2021_05_31_aggregated.csv',
                       index_col=0)
sent21_3 = pd.read_csv('Data/VADER_ROBERTA_Daily_Results_2021_06_01__2021_12_31_aggregated.csv',
                       index_col=0)
sent22_1 = pd.read_csv('Data/VADER_ROBERTA_Daily_Results_2022_01_01__2022_10_31_aggregated.csv',
                       index_col=0)
sent22_2 = pd.read_csv('DataVADER_ROBERTA_Daily_Results_2022_11_01__2022_11_19_aggregated.csv',
                       index_col=0)

sentiment  = sent20.append([sent21_1,sent21_2,sent21_3,sent22_1,sent22_2])

del(sent20,sent21_1,sent21_2,sent21_3,sent22_1,sent22_2)


data = pd.merge(return_data, sentiment, left_index=True, right_index=True)
data.head()

# Create an 80-20 Train-test split: total number of observations 717,
# Train: 574 Test: 143

train_obs_count = 574

# 2) Define OLS model


def OLS_model(factors, title):
    # Define matrix of explanatory variables and vector of Bitcoin returns
    X = data.loc[:,factors]
    y = data.loc[:,['BTC Returns']]
    
    # Train and test samples
    X_train = X[:train_obs_count]
    X_test = X[train_obs_count]
    
    y_train = y[:train_obs_count]
    y_test = y[train_obs_count:]
    
    X_train = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train).fit()
    
    vif = pd.DataFrame()
    vif['VIF Train'] = [variance_inflation_factor(X_train.values, i) for i in range(len(X.columns))]
    
    x = sm.add_constant(X_test)
    fitted = model.predict(x) 
     
    print_model = model.summary()
    print(print_model)

    vif['VIF Test'] = [variance_inflation_factor(x.values, i) for i in range(len(X.columns))]
    print(vif)
    # Print Variance Inflation Factors for Train and Tests samples to detect
    # potential multicollinearity.
    
    
    # Use Breusch-Pagan Test to check for heteroskedasticity of residuals
    names = ['Lagrange multiplier statistic', 'p-value','f-value', 'f p-value']
    bp_test = sms.het_breuschpagan(model.resid, model.model.exog)
    print('Breusch-Pagan Test:')
    print(lzip(names, bp_test))
    

    # Create a plot showing fitted returns and realized returns
    plt.figure(figsize=(9,6))
    y_test.name=('Realized BTC log returns')
    y_test.plot(color='blue')
    fitted.name=('Fitted BTC log returns')
    fitted.plot(color='red')
    plt.legend(bbox_to_anchor =(0.5,-0.2),loc='lower center', ncol=2, 
               frameon=False)
    plt.xticks(fontsize=7)
    plt.title(title, fontsize=10)
    plt.savefig('Plot/'+title+'.png', transparent=True)
    plt.close()

    
    # Calculate the mean squared error
    errors = model.resid
    errors_squ = np.square(errors)
    print('Mean Squared Error in Percent:', round(np.mean(errors_squ)*100,4))
    
    # Mean squared error test-sample
    errors_test = np.array(fitted)-np.array(y_test)
    errors_test_squ = np.square(errors_test)
    print('Mean Squared Out of Sample Error in Percent:', 
          round(np.mean(errors_test_squ)*100,4))
    
    # Heteroskedasticity-robust model
    print('OLS with heteroskedasticity-robust standard errors:')
    robust = model.get_robustcov_results(cov_type='HC1')
    robust_model = robust.summary()
    print(robust_model)
    

OLS_model(['vader_score'], 'OLS Test Sample Performance using VADER sentiment')
OLS_model(['roberta_score'], 'OLS Test Sample Performance using ROBERTA sentiment')

# Include additionally S&P500 Returns 
OLS_model(['vader_score','SP500 Returns'], 574, 
          'OLS Test Sample Performance using S&P500 log returns and VADER sentiment')
OLS_model(['roberta_score','SP500 Returns'], 574,
          'OLS Test Sample Performance using S&P500 log returns and ROBERTA sentiment')
          
          
          
# 3) Use a Granger causality test to determine whether lagged values of 
#    Sentiment Scores help explain BTC returns.

# We know from our previous script that the time series of log BTC returns 
# is stationary.
# Check whether sentiment is stationary.

def adf_test(X):
    result = adfuller(X)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key,value))

# VADER score
adf_test(data['vader_score'])
# ADF Statistic: -1.480793
# Null hypothesis of unit root cannot be rejected on a 10% significance level.

# Create variable sentiment change: First Difference of VADER score
data['vader_diff'] = data['vader_score'].diff()
# Check for stationarity after first differencing
first_diff_vader = data['vader_diff']
first_diff_vader.drop(index=first_diff_vader.index[0], axis=0, inplace=True)
adf_test(first_diff_vader)
# ADF Statistic: -9.590333
# Null hypothesis of unit root can be rejected on 1% significance level.


# ROBERTA score
adf_test(data['roberta_score'])
# ADF Statistic: -8.336483
# Null hypothesis of unit root can be rejected on a 1% significance level.


# Is Sentiment useful for predicting BTC log return at all? 
# Use Granger causality test 

# Test on a maximum number of 7 lags (a week)
# As only ROBERTA score is stationary, perform test on 
granger_roberta = grangercausalitytests(data[['BTC Returns', 'roberta_score']],
                                        7)

# Perform Granger causality test for VADER on stationary difference series
vader_set = data.merge(first_diff_vader, right_index=True, left_index=True)
granger_vader = grangercausalitytests(vader_set[['BTC Returns','vader_diff_x']],
                                      7)

# The null hypothesis of the lags not explaining BTC log returns cannot be 
# rejected for any lags of both differenced VADER score and ROBERTA on 
# conventional significance levels. 


# 4) Define function to run Random Forest Models

def Random_Forest(factors, title, decision_tree):
    X = data.loc[:,factors]
    y = data.loc[:,['BTC Returns']]
    
    X_train = X[:train_obs_count]
    X_test = X[train_obs_count:]
    
    y_train = y[:train_obs_count]
    y_test = y[train_obs_count:]
    
    
    dates = pd.to_datetime(y_test.index)
    
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)
    
  
    rf = RandomForestRegressor(n_estimators = 70, random_state = 42)
    rf.fit(X_train, y_train)
    fitted = rf.predict(X_test)
    
    y_test = pd.Series(y_test, index=dates)
    fitted = pd.Series(fitted, index=dates)
    
    plt.figure(figsize=(9,6))
    y_test.name=('Realized BTC log returns')
    y_test.plot(color='blue')
    fitted.name=('Fitted BTC log returns')
    fitted.plot(color='red')
    plt.legend(bbox_to_anchor =(0.5,-0.25),loc='lower center', ncol=2, fontsize=12,
               frameon=False)
    plt.title(title, fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('Plot/'+title+'.png', transparent=True)
    plt.close()
    
    first_decision_tree = export_graphviz(rf.estimators_[0], 
                           feature_names=factors,
                           class_names='BTC_Returns', 
                           filled=True, impurity=True, 
                           rounded=True)

    graph = graphviz.Source('Plot/'+first_decision_tree, format='png')
    graph.render(decision_tree)
    
    # Calculate the mean squared out-of-sample errors
    errors = fitted - y_test
    errors_squ = np.square(errors)
    print('Mean Squared Out of Sample Error in Percent:', 
          round(np.mean(errors_squ)*100,4))


Random_Forest(['vader_score'], 'Random Forest using VADER sentiment',
              'First Decision Tree VADER Random Forest')
Random_Forest(['roberta_score'], 'Random Forest using ROBERTA sentiment',
              'First Decision Tree ROBERTA Random Forest')

Random_Forest(['vader_score', 'SP500 Returns'], 
              'Random Forest using S&P500 log returns and VADER sentiment',
              'First Decision Tree S&P500 log returns and VADER Random Forest')

Random_Forest(['roberta_score', 'SP500 Returns'], 
              'Random Forest using S&P500 log returns and ROBERTA sentiment',
              'First Decision Tree S&P500 log returns and ROBERTA Random Forest')

