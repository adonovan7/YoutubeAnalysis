# Basics
import pandas as pd; import os
import csv; import numpy as np
import re; import warnings
warnings.filterwarnings('ignore')

import sklearn # machine learning
from sklearn.model_selection import train_test_split # splitting up
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer

from NLP import data, df

videos = data.copy()

videos.head(3)
# splitting up data

# normal
x_train_videos, x_test_videos, y_train_videos, y_test_videos = train_test_split(
    videos["com_stem_str"], videos["label"], test_size=0.33, random_state=42)

# bigrams
x_train_videos_bi_tk, x_test_videos_bi_tk, y_train_videos_bi, y_test_videos_bi = train_test_split(
    videos["comment"], videos["label"], test_size=0.33, random_state=42)

# train on all videos, predict on user data
X_train = data["com_stem_str"]
Y_train = data["label"]
X_user = df["com_stem_str"]

videos = videos.dropna(axis=0)
videos["comment"] = videos["comment"].str.lower()

## transformations

# initialize transformers
tfidf = TfidfVectorizer()
CV = CountVectorizer(ngram_range=(1, 1))
bi_CV = CountVectorizer(ngram_range=(1, 2))

xtrain = tfidf.fit_transform(X_train) # transform and fit training data
xuser = tfidf.transform(X_user) # transform user selected comments to predict on
