# -*- coding: utf-8 -*-
"""
Project SDA: 
Do RoBERTa and VADER sentiment models on r/Bitcoin help to explain Bitcoin retuns?

Part 5 - Using Hidden Markov Model on Aggregated Sentiment


"""

import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm


# This Code loads Aggregated Daily Data for RoBERTa and VADER scores.
# After loading a function to determine three Hidden Markov States is defined
# and a plot is created for both sentiment scores. 


# 1) Load the Sentiment Data

return_data = pd.read_csv('BTC_merged.csv', index_col=0)
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
sent22_2 = pd.read_csv('Data/VADER_ROBERTA_Daily_Results_2022_11_01__2022_11_19_aggregated.csv',
                       index_col=0)

data  = sent20.append([sent21_1,sent21_2,sent21_3,sent22_1,sent22_2])

del(sent20,sent21_1,sent21_2,sent21_3,sent22_1,sent22_2)


# Create a Date column for the plot
data['Date'] = pd.to_datetime(data.index)


# 2) Define a function to derive the three HMM states and plot the time series
def hmm_states(Sentiment, Sentiment_Label, Title, Savefigure):
    y = data[[Sentiment]].values
    model = hmm.GaussianHMM(n_components = 3, covariance_type = "diag",
                            n_iter = 50, random_state = 42)
    model.fit(y)
    Z = model.predict(y)
    # Assign the three states to observations of aggregated Daily Sentiment Scores
    states = pd.unique(Z)
    
    plt.figure(figsize = (15,10))
    # Assign different colors to the three states
    for i in states:
        want = (Z==i)
        x = data['Date'].iloc[want]
        y = data[Sentiment].iloc[want]
        plt.plot(x, y, '.', markersize=15)
    plt.legend(states, bbox_to_anchor = (0.5,-0.15), loc='lower center',
               ncol=3, fontsize=16, frameon=False)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel(Sentiment_Label, fontsize=18)
    plt.title(Title, fontsize=20)
    plt.savefig(Savefigure, transparent=True)


hmm_states('vader_score', 'Vader Score', 
           'Time Series of VADER Scores with Hidden Markov States', 
           'Plots/HMM_VADER.png')

hmm_states('roberta_score', 'Roberta Score',
           'Time Series of ROBERTA Scores with Hidden Markov States',
           'Plots/HMM_ROBERTA.png')
