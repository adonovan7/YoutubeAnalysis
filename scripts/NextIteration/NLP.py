'''
PART 2
Script to read in data cleaned by Cleaner.py. Data is then pushed through NLP
function
'''

import pandas as pd; import os
import csv; import numpy as np
import re; import warnings
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

import Cleaner

df = Cleaner.df
videos = Cleaner.videos

sw = stopwords.words('english')
nltk.download('stopwords')
ps = PorterStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()

df['com_token']=df['comment'].str.lower().str.split()

def nlpFunction(DF):
    DF['com_token'] = DF['comment'].str.lower().str.split()
    DF['com_remv'] = DF['com_token'].apply(lambda x: [y for y in x if y not in sw])
    DF["com_lemma"] = DF['com_remv'].apply(lambda x : [lemmatizer.lemmatize(y) for y in x]) # lemmatization
    DF['com_stem'] = DF['com_lemma'].apply(lambda x : [ps.stem(y) for y in x]) # stemming
    DF["com_full"] = DF["com_stem"].apply(' '.join)
    DF["com_tagged"] = DF['comment'].apply(lambda x : [nltk.pos_tag(y) for y in x]) #word tagging
    DF["com_stem_str"] = DF["com_stem"].apply(', '.join)
    return DF

df = nlpFunction(df)
data= nlpFunction(videos)
data = data.dropna()
