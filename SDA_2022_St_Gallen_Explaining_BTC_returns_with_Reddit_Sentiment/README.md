[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **SDA_2022_St_Gallen_Explaining_BTC_returns_with_Reddit_Sentiment** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml

Name of Quantlet: 'SDA_2022_St_Gallen_Explaining_BTC_returns_with_Reddit_Sentiment'

Published in: 'SDA_2022_St_Gallen'

Description: 'This Quantlet uses Reddit Titles on the subreddit r/Bitcoin from the start of 2020 until 19th November of 2022 as input for VADER and RoBERTa sentiment score calculation. Subsequently using OLS and Random Forest models aims at determining whether the sentiment scores help explain Bitcoin log returns.'

Keywords: 'BTC, Bitcoin, sentiment analysis, VADER, RoBERTa, Reddit, random forest, HMM, Hidden Markov Model'

Author: 'Josef Gitterle, Nobin Kachirayil, Andreas Karlsson, Alina Schmidt'

Submitted: '11 December 2022'

Input: 'Daily historical Bitcoin and S&P500 log returns, Daily aggregated VADER and RoBERTa scores based on previously web-scraped raw Reddit Titles from 01-01-2020 until 19-11-2022.'

Output: 'Fitted log returns of BTC resulting from OLS and from Random Forest models as well as descriptive statistics of Sentiment and BTC returns'

```

![Picture1](1.%20Random%20Forest%20using%20S&P500%20log%20returns%20and%20ROBERTA%20Sentiment.jpg)

![Picture2](2.%20Random%20Forest%20using%20S&P500%20log%20returns%20and%20VADER%20Sentiment.jpg)