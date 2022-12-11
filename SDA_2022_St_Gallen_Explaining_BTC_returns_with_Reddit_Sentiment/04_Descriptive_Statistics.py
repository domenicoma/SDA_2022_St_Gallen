# -*- coding: utf-8 -*-
"""

Do RoBERTa and VADER sentiment models on r/Bitcoin help to explain Bitcoin retuns?

Part 4 - Descriptive Statistics

"""



# packages
import pandas as pd
from pandas.plotting import table
import numpy as np
import seaborn as sns

from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from sklearn.preprocessing import OneHotEncoder
from scipy.stats import pearsonr

from PIL import Image
from wordcloud import WordCloud, STOPWORDS



# 1) Create a table with descriptive statistics of log returns
# 2) Correlation S&P500 Log Returns and BTC Log Returns
# 3) Plot time series
# 3) Correlation S&P500 Log Returns and BTC Log Returns
# 4) Augmented Dickey Fuller Test
# 5) Autocorrelation of log return time series
# 6) Have a look at Reddit sentiments
# 7) Wordcloud



# 1) Create a table with descriptive statistics of log returns

# Read merged data set of Bitcoin and SP500 Returns. Keep in mind that 
# due to the inner merge, only observations for trading days of SP500 
# are considered, as these data set is also used for later analyses.

data = pd.read_csv("Data/BTC_merged.csv", index_col=0)

# Merge descriptive statistics of both returns
descr_stat = round(data['BTC Returns'].describe(),4)
descr_stat2 = round(data['SP500 Returns'].describe(),4)
descr_stat = pd.merge(descr_stat,descr_stat2, left_index=True, right_index=True)
del(descr_stat2)


# Save descriptive statistics table of Returns
fig, ax = plt.subplots()
ax.xaxis.set_visible(False) 
ax.yaxis.set_visible(False)  
ax.set_frame_on(False)
plt.title('Descriptive Statistics of Log-Returns Time Series')  
tab = table(ax, descr_stat, loc='upper right', colWidths=[0.4]*len(descr_stat.columns))  
tab.auto_set_font_size(False) 
tab.set_fontsize(10) 
plt.savefig('Plots/Returns_Described.png')


# 2) Correlation S&P500 Log Returns and BTC Log Returns

# Have a look at scatterplot between BTC returns and SP500 returns

plt.figure(figsize=(10,7))
sns.regplot(x=data['BTC Returns'], y=data['SP500 Returns'], 
            fit_reg=False, scatter_kws={"color":"darkblue","alpha":0.3,"s":40})
plt.title('Log Returns of Bitcoin against S&P500 Log Returns', fontsize=14)
plt.xlabel('BTC Log Returns', fontsize=14)
plt.ylabel('S&P500 Log Returns', fontsize=14)
plt.savefig('Plots/Scatterplot BTC and S&P500 returns.png', transparent=True)

# Compute Pearson's correlation coefficient for S&P 500 returns and BTC Returns
correlation,_ = pearsonr(data['BTC Returns'], data['SP500 Returns'])
print('Pearsons correlation: %.3f' % correlation)


# 3) Plot time series 


# Plot the two different return series over time
# Use for the plots the complete BTC time series available to compare it better
# to the sentiment plots that will be created in the next script.

data_BTC = pd.read_csv('Data/BTC_Daily.csv', index_col=0) 

def ts_plot(data, column,title,label,col,filename):
    data.index = pd.to_datetime(data.index)
    fig,ax=plt.subplots()
    plt.title(title)
    x=data.index
    y=data.iloc[:,[column]]
    plt.ylabel(label)
    ax.plot(x,y,color=col)
    plt.xticks(fontsize=8)
    plt.savefig(filename, transparent=True)

ts_plot(data_BTC, 6,'Bitcoin', 'Log Returns','navy',
        'Plots/Bitcoin_Return_Timeseries.png')

ts_plot(data, 7,'S&P 500', 'Log Returns','darkred',
        'Plots/SP500_Return_Timeseries.png')

# Plot time series of BTC closing prices and BTC Volume
ts_plot(data_BTC, 3, 'Btcoin', 'Closing Price', 'navy',
        'Plots/Bitcoin_Closing_Timeseries.png')

ts_plot(data_BTC, 5, 'Bitcoin', 'Volume', 'navy',
        'Plots/Bitcoin_Volume_Timeseries.png')



# Apart from remarkable outliers during the onset of Covid, the return series
# appear to be more or less stationary.

# 4) Augmented Dickey Fuller Test

# Use Augmented Dickey Fuller Test to test series for stationarity on merged 
# data set that will be used for analyses.

def adf_test(X):
    result = adfuller(X)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key,value))

# Bitcoin
adf_test(data['BTC Returns'])
# ADF Statistic: -6.804316

# S&P 500
adf_test(data['SP500 Returns'])
# ADF Statistic: -7.957932

# In both cases, null hypothesis of unit root can be rejected on a 1% significance level.



# 5) Autocorrelation of log return time series
# Have a look at autocorrelation and partial autocorrelation for the BTC log return series.

fig, axes = plt.subplots(1,2,figsize=(16,6))
plt.suptitle("BTC Log-Returns", size=16)
plot_acf(data['BTC Returns'].tolist(), lags=50, ax=axes[0])
plot_pacf(data['BTC Returns'].tolist(), lags=50, ax=axes[1])
plt.savefig('Plots/BTC_ACF_PACF.png', transparent=True)
# As expected, no significant serial correlation pattern can be detected.



# 6) Have a look at Reddit sentiments

# Have a look at the sentiment scores VADER and ROBERTA:
sentim20 = pd.read_csv('Data/VADER_ROBERTA_raw_Results_2020_01_01__2020_12_31.csv')
sentim21_1 = pd.read_csv('Data/VADER_ROBERTA_raw_Results_2021_01_01__2021_02_04.csv')
sentim21_2 = pd.read_csv('Data/VADER_ROBERTA_raw_Results_2021_02_07__2021_05_31.csv')
sentim21_3 = pd.read_csv('Data/VADER_ROBERTA_raw_Results_2021_06_01__2021_12_31.csv')
sentim22_1 = pd.read_csv('Data/VADER_ROBERTA_raw_Results_2022_01_01__2022_10_31.csv')
sentim22_2 = pd.read_csv('Data/VADER_ROBERTA_raw_Results_2022_11_01__2022_11_19.csv')

sentim = sentim20.append([sentim21_1,sentim21_2,sentim21_3,sentim22_1,sentim22_2])
del(sentim20,sentim21_1,sentim21_2,sentim21_3,sentim22_1,sentim22_2)
# Change format of date column
sentim['Dates'] = pd.to_datetime(sentim['Dates'])

# Using One-Hot Encoder to create dummy variables of sentiment scores
encoder = OneHotEncoder(handle_unknown='ignore')
def sentiment_count(sentiment,columns):
    encoder_df = pd.DataFrame(encoder.fit_transform(sentim[[sentiment]]).toarray())
    sentiment_df = sentim.join(encoder_df)
    sentiment_df = sentiment_df.rename(columns=columns)
    # Calculate sum for each date: determine number of comments for each sentiment dummy
    sentiment_df = sentiment_df.groupby('Dates').agg('sum')
    return(sentiment_df)

# Prepare Plots of daily Sentiment Score numbers
vader_cols = {0:'-1',1:'0',2:'1'}
vader_count = sentiment_count('vader_score',vader_cols)

roberta_cols = {0:'-1',1:'1'}
roberta_count = sentiment_count('roberta_score',roberta_cols)


# Create the plots
plt.figure(figsize=(10, 6))
plt.title('VADER Sentiment Count across time', fontsize=15)
plt.stackplot(vader_count.index,vader_count['-1'], vader_count['0'],
              vader_count['1'], 
              labels=['Negative Sentiment','Neutral Sentiment',
                      'Positive Sentiment'],
              colors=['darkred','orange','green'])
plt.ylabel('Daily count of comments with respective sentiment', fontsize=13)
plt.legend(bbox_to_anchor =(0.5,-0.15),loc='lower center', ncol=3)
plt.savefig('Plots/VADER Sentiment Count across time.png', transparent=True)


plt.figure(figsize=(10, 6))
plt.title('ROBERTA Sentiment Count across time', fontsize=15)
plt.stackplot(roberta_count.index,roberta_count['-1'], roberta_count['1'], 
              labels=['Negative Sentiment','Positive Sentiment'],
              colors=['darkred','green'])
plt.ylabel('Daily count of comments with respective sentiment', fontsize=13)
plt.legend(bbox_to_anchor =(0.5,-0.15),loc='lower center', ncol=2)
plt.savefig('Plots/ROBERTA Sentiment Count across time.png', transparent=True)



# Have a look at aggregated daily ROBERTA and VADER scores

# Descriptive statistics of sentiment scores
descr_stat = round(roberta_count['roberta_score'].describe(),4)
descr_stat2 = round(vader_count['vader_score'].describe(),4)
descr_stat = pd.merge(descr_stat,descr_stat2, left_index=True, right_index=True)
del(descr_stat2)


# Save descriptive statistics table of Returns
fig, ax = plt.subplots()
ax.xaxis.set_visible(False) 
ax.yaxis.set_visible(False)  
ax.set_frame_on(False)
plt.title('Descriptive Statistics of Daily aggregated Sentiment scores')  
tab = table(ax, descr_stat, loc='upper right', colWidths=[0.4]*len(descr_stat.columns))  
tab.auto_set_font_size(False) 
tab.set_fontsize(10) 
plt.savefig('Plots/Sentiment_Described.png')



# 7) Word Cloud 
# Create a word cloud from Reddit Titles 
titles = sentim['title']

mask = np.array(Image.open('reddit.jpg'))
# plot the WordCloud image                        
plt.figure( figsize=(12,10) )
wordcloud = WordCloud(stopwords=set(STOPWORDS),max_words=10000, 
                      mask=mask, colormap='YlOrRd',
                      background_color='maroon',
                      contour_color='white', contour_width=1).generate(str(titles))
# Generate plot
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig('Plots/Wordcloud_Reddit_Titles.png')
