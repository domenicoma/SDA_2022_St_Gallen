# -*- coding: utf-8 -*-
"""
Project SDA: 
Do RoBERTa and VADER sentiment models on r/Bitcoin help to explain Bitcoin retuns?

Part 2 - Reddit Sentiment Analysis
"""



# packages
import pandas as pd

# For VADER
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')

# For Roberta
# pip install transformers
# conda install -c pytorch pytorch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

# This code has 7 segments:
    # 1) Clean downloaded Reddit Posts
    # 2) Sentiment Analysis - Polarity Score with VADER and RoBERTa
        # 2a) VADER
        # 2b) RoBERTa
        # 2c) Polarity Scores
    # 3) VADER and RoBERTa Scores
    # 4) Calculate Daily Averages
    # 5) Saving Results
    # 6) Repeat Step 1)-5) for remaining time periods
        # 6a) 01.01.2021 - 31.12.2021
        # 6b) 01.01.2022 - 31.10.2022
        # 6c) 01.11.2022 - 19.11.2022
    # 7) Merge all Datasets

#-------------------------------------------------------------------------------------------

"""
Dataset 2a: 
# 01.01.2020 - 31.12.2020
"""
# 1) Clean downloaded Reddit Posts

## Read in Data
df = pd.read_csv('Data/Reddit_Submissions_2020_1_1__2020_12_31.csv')
test = df[['id','author','title','num_comments','created_utc','selftext']]
df.head()
print(df.shape)
## Clean Data - Drop NAs and add 'TextSentiment'  column = 'title'+'self-text'
df[df['title'].isnull()] # Finding Title NAs (to check if NAs in title also correspond to NAs in 'selftext')
df = df.dropna(subset=['title']) # Dropping all NAs
df.loc[(df['selftext']=='[removed]') |(df['selftext'].isnull()), 'selftext'] = '' # replacing [removed] and NAs through empty string
df['TextSentiment'] = df['title'] + df['selftext']

## Check Code on reduced sample
##df = df.head(5000)
##print(df.shape)

# 2) Sentiment Analysis

## Since we use two different Sentiment Analyzers, we also present their packags / codes seperatly below

# 2a) VADER
## We will use the NLTK's SentimentIntensityAnalyzer to get the neg/neu/pos scores of the text
## IMPORTANT: This approach does not account for relationship between words
sia = SentimentIntensityAnalyzer()

# 2b) RoBERTa
## Use a model trained of a large corpus of data
## Transformer model accounts for the words but also the context related to other words
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
tokenizer.vocab_size
def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, truncation=True, max_length=511, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
          'roberta_neg' : scores [0],
          'roberta_neu' : scores [1],
          'roberta_pos' : scores [2]
    }
    return scores_dict

# 2c) Polarity score
## Run the polarity score on the entire dataset
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        title = row['TextSentiment']
        myid = row['id']
        vader_result = sia.polarity_scores(title) # VADER score
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(title) # RoBERTa score
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f'Broke for id {myid}')
results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'id'})
results_df = results_df.merge(df, how='left')

# 3) VADER and RoBERTa Scores

## Addition of a Score for Vader Compound Results
## Setting threshold of score categorization according to https://levelup.gitconnected.com/reddit-sentiment-analysis-with-python-c13062b862f6
results_df['vader_score'] = 0
results_df.loc[results_df['vader_compound'] > 0.10, 'vader_score'] = 1 
results_df.loc[results_df['vader_compound'] < -0.10, 'vader_score'] = -1 
## Calculating compound score from Roberta results
results_df['roberta_score'] = 0
results_df.loc[results_df['roberta_pos'] > results_df['roberta_neg'], 'roberta_score'] = 1
results_df.loc[results_df['roberta_pos'] < results_df['roberta_neg'], 'roberta_score'] = -1

# 4) Daily Averages

## Calculating daily 'vader_score
results_df_daily = results_df.groupby('Dates')['vader_score'].sum()
## Calculating daily roberta_score
results_df_daily_roberta = results_df.groupby('Dates')['roberta_score'].sum()
## Combining daily vader and roberta scores
results_df_daily_vader_roberta = pd.concat([results_df_daily, results_df_daily_roberta], axis=1, join="inner")

# 5) Saving Results

## Aggregated (daily) VADER and ROBERTA results 
results_df_daily_vader_roberta.to_csv('Data/VADER_ROBERTA_Daily_Results_2020_01_01__2020_12_31_aggregated.csv')
## Non-aggregated (raw) VADER and ROBERTA results
results_df.to_csv('Data/VADER_ROBERTA_raw_Results_2020_01_01__2020_12_31.csv')



"""
Part 6a: Repeat Previous Steps for other Datasets
# 01.01.2021 - 31.12.2021 
"""

# 1) Clean downloaded Reddit Posts

## Read in Data
df_2021 = pd.read_csv('Reddit_Submissions_2021_1_1__2021_12_31.csv')
df_2021.head()
print(df_2021.shape)
## Clean Data - Drop NAs + 'TextSentiment'
df_2021[df_2021['title'].isnull()] # Finding Title NAs
df_2021 = df_2021.dropna(subset=['title']) # Dropping all NAs
df_2021.loc[(df_2021['selftext']=='[removed]') |(df_2021['selftext'].isnull()), 'selftext'] = '' # replacing [removed] and NAs through empty string
df_2021['TextSentiment'] = df_2021['title'] + df_2021['selftext']

## As df in 2021 is too big, we further split it into 3 smaller chunks for three time periods (01.01 - 02.04 | 02.05 - 05.31 | 06.01 - 12.31)
df_2021_1 = df_2021.loc[(df_2021['Dates'] >='2021-06-01')] # [0:68719]
df_2021_2 = df_2021.loc[(df_2021['Dates'] < '2021-06-01') & (df_2021['Dates'] >= '2021-02-05')]  # [68720:126515]
df_2021_3 = df_2021.loc[(df_2021['Dates'] < '2021-02-05')]  # [126516:194094]


# 2) Sentiment Analysis

# 2c) Polarity score
res_2021_1 = {}
res_2021_2 = {}
res_2021_3 = {}
## We canot run the whole year (all three subsets) in a loop
## If the loop gets interrupted, none of the already looped entries gets stored

# df_2021_1 (01.06.2021 - 31.12.2021)
df = df_2021_1
for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            title = row['TextSentiment']
            myid = row['id']
            vader_result = sia.polarity_scores(title) # Vader score
            vader_result_rename = {}
            for key, value in vader_result.items():
                vader_result_rename[f"vader_{key}"] = value
            roberta_result = polarity_scores_roberta(title)
            both = {**vader_result_rename, **roberta_result}
            date = row['Dates']
            if date >= '2021-06-01':
                res_2021_1[myid] = both
            elif date < '2021-06-01' and date >= '2021-02-05':
                res_2021_2[myid] = both
            else:
                res_2021_3[myid] = both
            print(i)
        except RuntimeError:
            print(f'Broke for id {myid}')
results_df_2021_1 = pd.DataFrame(res_2021_1).T
results_df_2021_1 = results_df_2021_1.reset_index().rename(columns={'index': 'id'})
results_df_2021_1 = results_df_2021_1.merge(df_2021_1, how='left')
    # 5) VADER and RoBERTA scores 
results_df_2021_1['vader_score'] = 0
results_df_2021_1.loc[results_df_2021_1['vader_compound'] > 0.10, 'vader_score'] = 1 
results_df_2021_1.loc[results_df_2021_1['vader_compound'] < -0.10, 'vader_score'] = -1 
results_df_2021_1['roberta_score'] = 0
results_df_2021_1.loc[results_df_2021_1['roberta_pos'] > results_df_2021_1['roberta_neg'], 'roberta_score'] = 1
results_df_2021_1.loc[results_df_2021_1['roberta_pos'] < results_df_2021_1['roberta_neg'], 'roberta_score'] = -1
    # 6) Daily Averages
results_df_daily_2021_1 = results_df_2021_1['daily_vader_score'] = results_df_2021_1.groupby('Dates')['vader_score'].sum()
results_df_daily_roberta_2021_1 = results_df_2021_1['daily_roberta_score'] = results_df_2021_1.groupby('Dates')['roberta_score'].sum()
results_df_daily_vader_roberta_2021_1 = pd.concat([results_df_daily_2021_1, results_df_daily_roberta_2021_1], axis=1, join="inner")
    # 7) Saving Results
results_df_daily_vader_roberta_2021_1.to_csv('Data/VADER_ROBERTA_Daily_Results_2021_06_01__2021_12_31_aggregated.csv')
results_df_2021_1.to_csv('Data/VADER_ROBERTA_raw_Results_2021_06_01__2021_12_31.csv')

# df_2021_2 (02.05.2021 - 31.05.2021)
df = df_2021_2
for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            title = row['TextSentiment']
            myid = row['id']
            vader_result = sia.polarity_scores(title) # Vader score
            vader_result_rename = {}
            for key, value in vader_result.items():
                vader_result_rename[f"vader_{key}"] = value
            roberta_result = polarity_scores_roberta(title)
            both = {**vader_result_rename, **roberta_result}
            date = row['Dates']
            if date >= '2021-06-01':
                res_2021_1[myid] = both
            elif date < '2021-06-01' and date >= '2021-02-05':
                res_2021_2[myid] = both
            else:
                res_2021_3[myid] = both
            print(i)
        except RuntimeError:
            print(f'Broke for id {myid}')
results_df_2021_2 = pd.DataFrame(res_2021_2).T
results_df_2021_2 = results_df_2021_2.reset_index().rename(columns={'index': 'id'})
results_df_2021_2 = results_df_2021_2.merge(df_2021_2, how='left')
    # 3) VADER and RoBERTA scores 
results_df_2021_2['vader_score'] = 0
results_df_2021_2.loc[results_df_2021_2['vader_compound'] > 0.10, 'vader_score'] = 1 
results_df_2021_2.loc[results_df_2021_2['vader_compound'] < -0.10, 'vader_score'] = -1 
results_df_2021_2['roberta_score'] = 0
results_df_2021_2.loc[results_df_2021_2['roberta_pos'] > results_df_2021_2['roberta_neg'], 'roberta_score'] = 1
results_df_2021_2.loc[results_df_2021_2['roberta_pos'] < results_df_2021_2['roberta_neg'], 'roberta_score'] = -1
    # 4) Daily Averages
results_df_daily_2021_2 = results_df_2021_2['daily_vader_score'] = results_df_2021_2.groupby('Dates')['vader_score'].sum()
results_df_daily_roberta_2021_2 = results_df_2021_2['daily_roberta_score'] = results_df_2021_2.groupby('Dates')['roberta_score'].sum()
results_df_daily_vader_roberta_2021_2 = pd.concat([results_df_daily_2021_2, results_df_daily_roberta_2021_2], axis=1, join="inner")
    # 5) Saving Results
results_df_daily_vader_roberta_2021_2.to_csv('Data/VADER_ROBERTA_Daily_Results_2021_05_02__2021_05_31_aggregated.csv')
results_df_2021_2.to_csv('Data/VADER_ROBERTA_raw_Results_2021_05_02__2021_05_31.csv')

# df_2021_3 (01.01.2021 - 01.05.2021)
df = df_2021_3
for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            title = row['TextSentiment']
            myid = row['id']
            vader_result = sia.polarity_scores(title) # Vader score
            vader_result_rename = {}
            for key, value in vader_result.items():
                vader_result_rename[f"vader_{key}"] = value
            roberta_result = polarity_scores_roberta(title)
            both = {**vader_result_rename, **roberta_result}
            date = row['Dates']
            if date >= '2021-06-01':
                res_2021_1[myid] = both
            elif date < '2021-06-01' and date >= '2021-02-05':
                res_2021_2[myid] = both
            else:
                res_2021_3[myid] = both
            print(i)
        except RuntimeError:
            print(f'Broke for id {myid}')
results_df_2021_3 = pd.DataFrame(res_2021_3).T
results_df_2021_3 = results_df_2021_3.reset_index().rename(columns={'index': 'id'})
results_df_2021_3 = results_df_2021_3.merge(df_2021_3, how='left')
    # 3) VADER and RoBERTA scores 
results_df_2021_3['vader_score'] = 0
results_df_2021_3.loc[results_df_2021_3['vader_compound'] > 0.10, 'vader_score'] = 1 
results_df_2021_3.loc[results_df_2021_3['vader_compound'] < -0.10, 'vader_score'] = -1 
results_df_2021_3['roberta_score'] = 0
results_df_2021_3.loc[results_df_2021_3['roberta_pos'] > results_df_2021_3['roberta_neg'], 'roberta_score'] = 1
results_df_2021_3.loc[results_df_2021_3['roberta_pos'] < results_df_2021_3['roberta_neg'], 'roberta_score'] = -1
    # 4) Daily Averages
results_df_daily_2021_3 = results_df_2021_3['daily_vader_score'] = results_df_2021_3.groupby('Dates')['vader_score'].sum()
results_df_daily_roberta_2021_3 = results_df_2021_3['daily_roberta_score'] = results_df_2021_3.groupby('Dates')['roberta_score'].sum()
results_df_daily_vader_roberta_2021_3 = pd.concat([results_df_daily_2021_3, results_df_daily_roberta_2021_3], axis=1, join="inner")
     # 5) Saving Results
results_df_daily_vader_roberta_2021_3.to_csv('Data/VADER_ROBERTA_Daily_Results_2021_01_01__2021_05_01_aggregated.csv')
results_df_2021_3.to_csv('Data/VADER_ROBERTA_raw_Results_2021_01_01__2021_05_01.csv')


"""
Part 6b: Repeat Previous Steps for other Datasets
# 01.01.2022 - 31.10.2022 
"""
# 1) Clean downloaded Reddit Posts

## Read in Data
df_2022 = pd.read_csv('Reddit_Submissions_2022_1_1__2022_10_31.csv')
df_2022.head()
print(df_2022.shape)
## Clean Data - Drop NAs and add 'TextSentiment'  column = 'title'+'self-text'
df_2022[df_2022['title'].isnull()] # Finding Title NAs (to check if NAs in title also correspond to NAs in 'selftext')
df_2022 = df_2022.dropna(subset=['title']) # Dropping all NAs
df_2022.loc[(df_2022['selftext']=='[removed]') |(df_2022['selftext'].isnull()), 'selftext'] = '' # replacing [removed] and NAs through empty string
df_2022['TextSentiment'] = df_2022['title'] + df_2022['selftext']


# 2) Sentiment Analysis

# 2c) Polarity score
## Run the polarity score on the entire dataset
res = {}
for i, row in tqdm(df_2022.iterrows(), total=len(df_2022)):
    try:
        title = row['TextSentiment']
        myid = row['id']
        vader_result = sia.polarity_scores(title) # VADER score
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(title) # RoBERTa score
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f'Broke for id {myid}')
results_df_2022 = pd.DataFrame(res).T
results_df_2022 = results_df_2022.reset_index().rename(columns={'index': 'id'})
results_df_2022 = results_df_2022.merge(df_2022, how='left')

# 3) VADER and RoBERTa Scores

## Addition of a Score for Vader Compound Results
## Setting threshold of score categorization according to https://levelup.gitconnected.com/reddit-sentiment-analysis-with-python-c13062b862f6
results_df_2022['vader_score'] = 0
results_df_2022.loc[results_df_2022['vader_compound'] > 0.10, 'vader_score'] = 1 
results_df_2022.loc[results_df_2022['vader_compound'] < -0.10, 'vader_score'] = -1 
## Calculating compound score from Roberta results
results_df_2022['roberta_score'] = 0
results_df_2022.loc[results_df_2022['roberta_pos'] > results_df_2022['roberta_neg'], 'roberta_score'] = 1
results_df_2022.loc[results_df_2022['roberta_pos'] < results_df_2022['roberta_neg'], 'roberta_score'] = -1

# 4) Daily Averages

## Calculating daily 'vader_score
results_df_2022_daily = results_df_2022.groupby('Dates')['vader_score'].sum()
## Calculating daily roberta_score
results_df_2022_daily_roberta = results_df_2022.groupby('Dates')['roberta_score'].sum()
## Combining daily vader and roberta scores
results_df_2022_daily_vader_roberta = pd.concat([results_df_2022_daily, results_df_2022_daily_roberta], axis=1, join="inner")

# 5) Saving Results

## Aggregated (daily) VADER and ROBERTA results 
results_df_2022_daily_vader_roberta.to_csv('Data/VADER_ROBERTA_Daily_Results_2022_01_01__2022_10_31_aggregated.csv')
## Non-aggregated (raw) VADER and ROBERTA results
results_df_2022.to_csv('Data/VADER_ROBERTA_raw_Results_2022_01_01__2022_10_31.csv')

"""
Part 6c: Repeat Previous Steps for other Datasets
# 01.11.2022 - 19.11.2022 
"""
# 1) Clean downloaded Reddit Posts

## Read in Data
df_Nov22 = pd.read_csv('Reddit_Submissions_2022_11_1__2022_11_19.csv')
df_Nov22.head()
print(df_Nov22.shape)
## Clean Data - Drop NAs and add 'TextSentiment'  column = 'title'+'self-text'
df_Nov22[df_Nov22['title'].isnull()] # Finding Title NAs (to check if NAs in title also correspond to NAs in 'selftext')
df_Nov22 = df_Nov22.dropna(subset=['title']) # Dropping all NAs
df_Nov22.loc[(df_Nov22['selftext']=='[removed]') |(df_Nov22['selftext'].isnull()), 'selftext'] = '' # replacing [removed] and NAs through empty string
df_Nov22['TextSentiment'] = df_Nov22['title'] + df_Nov22['selftext']


# 2) Sentiment Analysis

# 2c) Polarity score
## Run the polarity score on the entire dataset
res = {}
for i, row in tqdm(df_Nov22.iterrows(), total=len(df_Nov22)):
    try:
        title = row['TextSentiment']
        myid = row['id']
        vader_result = sia.polarity_scores(title) # VADER score
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(title) # RoBERTa score
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f'Broke for id {myid}')
results_df_Nov22 = pd.DataFrame(res).T
results_df_Nov22 = results_df_Nov22.reset_index().rename(columns={'index': 'id'})
results_df_Nov22 = results_df_Nov22.merge(df_Nov22, how='left')

# 3) VADER and RoBERTa Scores

## Addition of a Score for Vader Compound Results
## Setting threshold of score categorization according to https://levelup.gitconnected.com/reddit-sentiment-analysis-with-python-c13062b862f6
results_df_Nov22['vader_score'] = 0
results_df_Nov22.loc[results_df_Nov22['vader_compound'] > 0.10, 'vader_score'] = 1 
results_df_Nov22.loc[results_df_Nov22['vader_compound'] < -0.10, 'vader_score'] = -1 
## Calculating compound score from Roberta results
results_df_Nov22['roberta_score'] = 0
results_df_Nov22.loc[results_df_Nov22['roberta_pos'] > results_df_Nov22['roberta_neg'], 'roberta_score'] = 1
results_df_Nov22.loc[results_df_Nov22['roberta_pos'] < results_df_Nov22['roberta_neg'], 'roberta_score'] = -1

# 4) Daily Averages

## Calculating daily 'vader_score
results_df_Nov22_daily = results_df_Nov22.groupby('Dates')['vader_score'].sum()
## Calculating daily roberta_score
results_df_Nov22_daily_roberta = results_df_Nov22.groupby('Dates')['roberta_score'].sum()
## Combining daily vader and roberta scores
results_df_Nov22_daily_vader_roberta = pd.concat([results_df_Nov22_daily, results_df_Nov22_daily_roberta], axis=1, join="inner")

# 5) Saving Results

## Aggregated (daily) VADER and ROBERTA results 
results_df_Nov22_daily_vader_roberta.to_csv('Data/VADER_ROBERTA_Daily_Results_2022_11_01__2022_11_19_aggregated.csv')
## Non-aggregated (raw) VADER and ROBERTA results
results_df_Nov22.to_csv('Data/VADER_ROBERTA_raw_Results_2022_11_01__2022_11_19.csv')

"""
Part 7: Merge Datasets
"""
# Load Datasets
df_2020 = pd.read_csv('Data/VADER_ROBERTA_Daily_Results_2020_01_01__2020_12_31_aggregated.csv')
df_2020_raw = pd.read_csv('Data/VADER_ROBERTA_raw_Results_2020_01_01__2020_12_31.csv')
df_2021_1 = pd.read_csv('Data/VADER_ROBERTA_Daily_Results_2021_01_01__2021_02_04_aggregated.csv')
df_2021_1_raw = pd.read_csv('Data/VADER_ROBERTA_raw_Results_2021_01_01__2021_02_04.csv')
df_2021_2 = pd.read_csv('Data/VADER_ROBERTA_Daily_Results_2021_02_07__2021_05_31_aggregated.csv')
df_2021_2_raw = pd.read_csv('Data/VADER_ROBERTA_raw_Results_2021_02_07__2021_05_31.csv')
df_2021_3 = pd.read_csv('Data/VADER_ROBERTA_Daily_Results_2021_06_01__2021_12_31_aggregated.csv')
df_2021_3_raw = pd.read_csv('Data/VADER_ROBERTA_raw_Results_2021_06_01__2021_12_31.csv')
df_2022_1 = pd.read_csv('Data/VADER_ROBERTA_Daily_Results_2022_01_01__2022_10_31_aggregated.csv')
df_2022_1_raw = pd.read_csv('Data/VADER_ROBERTA_raw_Results_2022_01_01__2022_10_31.csv')
df_2022_2 = pd.read_csv('Data/VADER_ROBERTA_Daily_Results_2022_11_01__2022_11_19_aggregated.csv')
df_2022_2_raw = pd.read_csv('Data/VADER_ROBERTA_raw_Results_2022_11_01__2022_11_19.csv')

# Combine Datasets

## Check if all are complete
len(df_2020) # 366 years -> complete
len(df_2021_1) + len(df_2021_2) + len(df_2021_3) # 348 days -> 17 days missing in 2021 due to webscraping error [02.05/06 + 03.01 / 03.06 / 03.18-26 / 04.10-13]
len(df_2022_1) + len(df_2022_2) # 365 days -> complete in 2022

## Aggregated Data (daily)
dflist = [df_2020, df_2021_1, df_2021_2, df_2021_3, df_2022_1, df_2022_2]
df = pd.concat(dflist)
## Dataset overview
df.head()
print(df.shape)

## Non-aggregated Data
dflist_raw = [df_2020_raw, df_2021_1_raw, df_2021_2_raw, df_2021_3_raw, df_2022_1_raw, df_2022_2_raw]
df_raw = pd.concat(dflist_raw).iloc[:,1:] # as we don't need first column
df_raw.head()
print(df_raw.shape)

# Saving Datasets
df.to_csv('Data/VADER_ROBERTA_Daily_Results_2020_01_01__2022_11_19_aggregated.csv')
df_raw.to_csv('Data/VADER_ROBERTA_raw_Results_2020_01_01__2022_11_19.csv')
