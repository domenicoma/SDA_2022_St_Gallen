# -*- coding: utf-8 -*-
"""
Project SDA: 
Do RoBERTa and VADER sentiment models on r/Bitcoin help to explain Bitcoin retuns?

Part 1 - Reddit Webscraping
"""



# Uncomment if not yet installed:
# pip install pmaw
# pip install praw
import pandas as pd
import praw
from pmaw import PushshiftAPI # To get Access to Reddit API


# This code has 3 segments:
    # 1) Setup Reddit API
    # 2) Download Reddit Posts
        # 2a) 01.01.2020 - 31.12.2020
        # 2b) 01.01.2021 - 31.12.2021
        # 2c) 01.01.2022 - 31.10.2022
        # 2d) 01.11.2022 - 19.11.2022
    # 3) Clean downloaded Reddit Posts

#-------------------------------------------------------------------------------------------
# 1) Setup Reddit API

# Defining API
reddit = praw.Reddit(client_id='P6KPhw8sca_vI6ycPZn2sg',
                     client_secret='cO-QEQThI07Q3Wtsct4pt_oILQuFxg',
                     user_agent='API_Test')
api = PushshiftAPI()

# 2) Download Reddit Posts

## Overall observation period is 01.01.2020 - 19.11.2022
## We split the download of Reddit Posts into 4 chunks to avoid too large datasets (and long waiting times)
## The splits are according to years, with an exception for the last split in 2022 (Step 2d) 
## The exception in 2022 occusrs as we initially planned to stop our observation period by 31.10.2022 but after the FTX-Collapse we decided to extend our dataset up to 19.11.2022
## This led to another download of Reddit posts from 01.11.2022 - 19.11.2022 in step 2d)

# 2a) 01.01.2020 - 31.12.2020
start_epoch=int(pd.to_datetime('2020-1-1').timestamp())
stop_epoch=int(pd.to_datetime('2021-1-1').timestamp())
subreddit ='Bitcoin'
post_list = list(api.search_submissions(after=start_epoch,
                                        before = stop_epoch,
                                        subreddit=subreddit,
                                        filter=['id',
                                                'author',
                                                'title',
                                                'selftext', 
                                                'num_comments',
                                                'upvote_ratio',
                                                'link_flair_text',
                                                'created_utc',
                                                'subreddit'],
                                        ))
df_every_post = pd.DataFrame(post_list).sort_values(by = ['created_utc'], ascending=False).reset_index().drop(['index'], axis=1) # sort the values according to data, reset index and delete newly generated 'index' column
df_every_post['Dates'] = pd.to_datetime(df_every_post['created_utc'],unit='s').dt.date
df_every_post['Time'] = pd.to_datetime(df_every_post['created_utc'],unit='s').dt.time
df_every_post
df_every_post.to_csv(r'Data/Reddit_Submissions_2020_1_1__2020_12_31.csv', index=False)


# 2b) 01.01.2021 - 31.12.2021
start_epoch=int(pd.to_datetime('2021-1-1').timestamp())
stop_epoch=int(pd.to_datetime('2022-1-1').timestamp())
subreddit ='Bitcoin'
post_list = list(api.search_submissions(after=start_epoch,
                                        before = stop_epoch,
                                        subreddit=subreddit,
                                        filter=['id',
                                                'author',
                                                'title',
                                                'selftext', 
                                                'num_comments',
                                                'upvote_ratio',
                                                'link_flair_text',
                                                'created_utc',
                                                'subreddit'],
                                        ))
df_every_post = pd.DataFrame(post_list).sort_values(by = ['created_utc'], ascending=False).reset_index().drop(['index'], axis=1)
df_every_post['Dates'] = pd.to_datetime(df_every_post['created_utc'],unit='s').dt.date
df_every_post['Time'] = pd.to_datetime(df_every_post['created_utc'],unit='s').dt.time
df_every_post
df_every_post.to_csv(r'Data/Reddit_Submissions_2021_1_1__2021_12_31.csv', index=False)


# 2c) 01.01.2022 - 31.10.2022
start_epoch=int(pd.to_datetime('2022-1-1').timestamp())
stop_epoch=int(pd.to_datetime('2022-11-1').timestamp())
subreddit ='Bitcoin'
post_list = list(api.search_submissions(after=start_epoch,
                                        before = stop_epoch,
                                        subreddit=subreddit,
                                        filter=['id',
                                                'author',
                                                'title',
                                                'selftext', 
                                                'num_comments',
                                                'upvote_ratio',
                                                'link_flair_text',
                                                'created_utc',
                                                'subreddit'],
                                        ))
df_every_post = pd.DataFrame(post_list).sort_values(by = ['created_utc'], ascending=False).reset_index().drop(['index'], axis=1) # sort the values according to data, reset index and delete newly generated 'index' column
df_every_post['Dates'] = pd.to_datetime(df_every_post['created_utc'],unit='s').dt.date
df_every_post['Time'] = pd.to_datetime(df_every_post['created_utc'],unit='s').dt.time
df_every_post
df_every_post.to_csv(r'Data/Reddit_Submissions_2022_1_1__2022_10_31.csv', index=False)


# 2d) 01.11.2022 - 19.11.2022
start_epoch=int(pd.to_datetime('2022-11-1').timestamp())
stop_epoch=int(pd.to_datetime('2022-11-20').timestamp())
subreddit ='Bitcoin'
post_list = list(api.search_submissions(after=start_epoch,
                                        before = stop_epoch,
                                        subreddit=subreddit,
                                        filter=['id',
                                                'author',
                                                'title',
                                                'selftext', 
                                                'num_comments',
                                                'upvote_ratio',
                                                'link_flair_text',
                                                'created_utc',
                                                'subreddit'],
                                        ))
df_every_post = pd.DataFrame(post_list).sort_values(by = ['created_utc'], ascending=False).reset_index().drop(['index'], axis=1)
df_every_post['Dates'] = pd.to_datetime(df_every_post['created_utc'],unit='s').dt.date
df_every_post['Time'] = pd.to_datetime(df_every_post['created_utc'],unit='s').dt.time
df_every_post
df_every_post.to_csv(r'Data/Reddit_Submissions_2022_11_1__2022_11_19.csv', index=False)
