'''
PART 1
Script to read in and clean data. Returns dataframe "videos" as
training/ test data and "df" as the data for predictions
'''

import pandas as pd; import os
import csv; import numpy as np
import re; import warnings
warnings.filterwarnings('ignore')

path = '/Users/andiedonovan/myProjects/Youtube_Python_Project/AndiesBranch/'
os.chdir(path) # change directory

df = pd.read_csv('data/data.csv', delimiter="@@@", skiprows=2, encoding='utf-8', engine='python')

# training data
okgo = pd.read_csv('data/OKGO.csv', delimiter=";", skiprows=2, encoding='latin-1', engine='python') # read in the data
trump = pd.read_csv('data/trump.csv', delimiter=",", skiprows=2, encoding='utf-8', error_bad_lines=False, engine='python')
swift = pd.read_csv('data/TaylorSwift.csv', delimiter=",", skiprows=2, nrows=180, encoding='utf-8', engine='python')
royal = pd.read_csv('data/RoyalWedding.csv', delimiter=",", skiprows=2, nrows=61, encoding='utf-8', engine='python')
paul = pd.read_csv('data/LoganPaul.csv', delimiter=",", skiprows=2, nrows=200, encoding='utf-8', engine='python')
blogs = pd.read_csv('data/Kagel.csv', delimiter=",", skiprows=2, encoding='latin-1', engine='python') # read in the data
tweets = pd.read_csv('data/twitter.csv', delimiter=",", skiprows=2, encoding='latin-1', engine='python') # read in the data

# clean dataframes
tweets = tweets.drop(['Topic', 'TweetId', "TweetDate"], axis = 1).dropna()
df.columns = ["comment", "label"]
df = df[["label", "comment"]]

def fix_cols(DF):
    DF = DF.iloc[:,:2]
    DF.columns = ["label", "comment"]
    return DF

okgo = fix_cols(okgo); trump = fix_cols(trump)
swift = fix_cols(swift); royal = fix_cols(royal)
paul = fix_cols(paul); tweets = fix_cols(tweets)
blogs = fix_cols(blogs); df = fix_cols(df)


# fix up tweets dataset
tweets.label = tweets.label.replace({'positive': '1.0', 'negative':'-1.0', 'neutral': '0.0', 'irrelevant': '0.0'}, regex=True)
tweets['label'] = pd.to_numeric(tweets['label'], errors='coerce')

# make one large dataframe from YouTube videos
videos = pd.concat([okgo, trump, swift, royal, paul], ignore_index=True)
videos = fix_cols(videos) # repetitive but for safe measures

# convert to string type
'''df["comment"]= df["comment"].astype(str)
videos["comment"]= videos["comment"].astype(str)
full["comment"]= full["comment"].astype(str)
'''

# make one large dataframe from all datasets including twitter and blogs
full = pd.concat([okgo, trump, swift, royal, paul, blogs, tweets], ignore_index=False)

df["comment"] = df["comment"].astype(str)
videos["comment"] = videos["comment"].astype(str)
full["comment"] = full["comment"].astype(str)

# removing non alphanumeric characters
def cleanerFn(b):
    for row in range(0,len(b)):
        line = b.iloc[row, 1]
        b.iloc[row,1] = re.sub("[^a-zA-Z]", " ", line)
        b.iloc[row,1] = b.iloc[row,1].lower()
    return b

df = cleanerFn(df)
videos = cleanerFn(videos)


filepath1 = path + "data/cleanedVideos.csv" # path for videos dataset
filepath2 = path + "data/cleanedUserData.csv" # path for user data

videos.to_csv(filepath1, sep=',', encoding='utf-8') # write to file for videos
df.to_csv(filepath2, sep=',', encoding='utf-8') # write to file for user data
