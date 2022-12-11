# -*- coding: utf-8 -*-
"""

Project SDA: 
Do RoBERTa and VADER sentiment models on r/Bitcoin help to explain Bitcoin retuns?

Part 3 - BTC  and S&P500 data preprocessing

"""


#packages

# Uncomment if not yet installed:
# pip install yfinance 
# pip install statsmodels


import pandas as pd
import numpy as np
import yfinance as yf


# This code has 4 segments:
    # 1) Download data 
    # 2) Prepare BTC data 
    # 3) Prepare S&P500 data
    # 4) Merge returns

# 1) Download data

# Keep last observation of 2019 for return calculation
BTC_Daily = yf.download('BTC-USD', start='2019-12-31',end='2022-11-20')
# Save data 
BTC_Daily.to_csv('Data/BTC_Daily_uncleansed_2020_2022.csv')  

# S&P 500 Daily Data
SP500_Daily = yf.download('^GSPC', start='2019-12-31', end='2022-11-20')
SP500_Daily.to_csv('Data/SP500_uncleansed_2020_2022.csv')


# 2) Prepare BTC data
BTC_Daily = pd.read_csv('Data/BTC_Daily_uncleansed_2020_2022.csv', index_col=0)
# Have a look at the data
BTC_Daily.head()
BTC_Daily.describe()

# Shape and Data Types
print(BTC_Daily.shape)
print(BTC_Daily.dtypes)

# Count is equal to 1055 for all variables which is equal to the number of observation in our dataset.
# To make sure check for missing values.
BTC_Daily.isnull().sum().sum()
# Zero missing values in DataFrame BTC_Daily

# Search for accidentally duplicated dates
duplicates = BTC_Daily.duplicated()
print(sum(duplicates))
# Zero duplicated timestamps, each date is unique in dataset
del(duplicates)

# Calculate log returns based on closing prices
BTC_Daily['BTC Returns'] = (np.log(BTC_Daily['Close'])-np.log(BTC_Daily['Close'].shift(1)))

# Time Format of Date index
BTC_Daily.index = pd.to_datetime(BTC_Daily.index,format='%Y-%m-%d')

# Drop the last observation of 2019:
BTC_Daily = BTC_Daily[BTC_Daily.index!='2019-12-31']

# Check for any remaining missing return values
BTC_Daily['BTC Returns'].isnull().sum()

# Save the whole cleansed data set
BTC_Daily.to_csv('Data/BTC_Daily.csv')



# 3) Prepare S&P 500 data

s_p = pd.read_csv('Data/SP500_uncleansed_2020_2022.csv', index_col=0)

s_p.describe()
s_p.isnull().sum().sum()
# Zero missing values in DataFrame BTC_Daily

# Search for duplicated dates
duplicates = s_p.duplicated()
print(sum(duplicates))
# Zero duplicated timestamps, each date is unique in dataset
del(duplicates)

# Calculate log returns based on closing prices
s_p['SP500 Returns'] = (np.log(s_p['Close'])-np.log(s_p['Close'].shift(1)))

# Time Format 
s_p.index = pd.to_datetime(s_p.index,format='%Y-%m-%d')

# Drop the last observation of 2019:
s_p = s_p[s_p.index!='2019-12-31']

# Also check here for missing return values
s_p['SP500 Returns'].isnull().sum()
# 0


# 4) Inner merge of return by Date

# Merge BTC and S&P500 Return Data
BTC_Daily.index = BTC_Daily.index.date 
# Create inner merge, S&P500 Data only available on working days
merged = pd.merge(BTC_Daily, s_p['SP500 Returns'], left_index=True, right_index=True)
# Save merged data set
merged.to_csv('Data/BTC_merged.csv')
