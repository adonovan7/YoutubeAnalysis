# coding: utf-8

import pandas as pd
import os
import csv
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.naive_bayes import MultinomialNB 
from sklearn.linear_model import LogisticRegression
from sklearn import svm # support vector machine
from sklearn import metrics # for accuracy/ precision
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier # Stochastic Gradient Descent
from sklearn.neighbors import KNeighborsClassifier # k-NN ensemble method
from sklearn.ensemble import RandomForestClassifier 

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import PorterStemmer
import sklearn # machine learning
from sklearn.model_selection import train_test_split # splitting up data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

path = '/Users/andiedonovan/myProjects/Youtube_Python_Project/AndiesBranch/'
os.chdir(path) # change directory

# load in data 
#os.chdir('/Users/andiedonovan/myProjects/Youtube_Python_Project/AndiesBranch/data/') # change directory
okgo = pd.read_csv('data/OKGOcomments.csv', delimiter=";", skiprows=2, encoding='latin-1', engine='python') # read in the data
blogs = pd.read_csv('data/Kagel_social_media_blogs.csv', delimiter="@@@", skiprows=2, encoding='latin-1', engine='python') # read in the data
tweets = pd.read_csv('data/full-corpus.csv', delimiter=",", skiprows=2, encoding='latin-1', engine='python') # read in the data
df = pd.read_csv('data/data.csv', delimiter="@@@", skiprows=2, encoding='utf-8', engine='python') # read in the user's data

# clean dataframes 
tweets = tweets.drop(['Topic', 'TweetId', "TweetDate"], axis = 1).dropna()
tweets.columns = ["label", "comment"]
tweets.label = tweets.label.replace({'positive': '1.0', 'negative':'-1.0', 'neutral': '0.0', 'irrelevant': '0.0'}, regex=True)
tweets['label'] = pd.to_numeric(tweets['label'], errors='coerce')
blogs.columns = ["label", "comment"]
blogs['label'] = pd.to_numeric(blogs['label'], errors='coerce')
okgo.columns = [
  'label','comment','a','b']
okgo = okgo.drop(['a', 'b'], axis = 1).dropna() # drop columns 3 and 4 and missing values
data = pd.concat([okgo, blogs, tweets], ignore_index=False)
df.columns = ["comment", "label"]

############
for row in range(len(data)):
    line = data.iloc[row,1]
    data.iloc[row,1] = re.sub("[^a-zA-Z]", " ", line)

df_copy = df

sw = stopwords.words('english')
ps = PorterStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()
tfidf = TfidfVectorizer()

df["comment"]= df["comment"].astype(str) 

for row in range(len(df)):
        line = df.loc[row, "comment"]
        #line = data.iloc[row,0]
        df.loc[row,"comment"] = re.sub("[^a-zA-Z]", " ", line)

labels = data['label']

def nlpFunction(a):
    a['com_token']=a['comment'].str.lower().str.split()
    a['com_remv']=a['com_token'].apply(lambda x: [y for y in x if y not in sw])
    a["com_lemma"] = a['com_remv'].apply(lambda x : [lemmatizer.lemmatize(y) for y in x]) # lemmatization
    a['com_stem']=a['com_lemma'].apply(lambda x : [ps.stem(y) for y in x]) # stemming
    return a

df = nlpFunction(df)
data = nlpFunction(data)

data['label'] = labels
df["com_stem_str"] = df["com_stem"].apply(', '.join)
data["com_stem_str"] = data["com_stem"].apply(', '.join)
data = data.loc[0:len(data),["label", "comment"]]
data = data.dropna()

# training and testing
X_train, X_test, Y_train, Y_test = train_test_split(
                                    data["com_stem_str"], data["label"], 
                                    test_size=0.25, 
                                    random_state=42)

# transforming data
tfidf = TfidfVectorizer()
xtrain = tfidf.fit_transform(X_train) # transform and fit training data
xtest = tfidf.transform(X_test) # transform test data from fitted transformer
data_trans= tfidf.transform(data["com_stem_str"]) # transform entire dataset for cross validation
df_trans = tfidf.transform(df["com_stem_str"])

# running models
rs = 10
lr = LogisticRegression(solver='sag', max_iter=100, random_state=rs, multi_class="multinomial")
mnb = MultinomialNB()
svm = svm.SVC()
rf = RandomForestClassifier(n_estimators=10, random_state=rs)
knn = KNeighborsClassifier()

models = ['lr', 'mnb', 'svm', 'rf', 'knn']
labels = ['label_' + str(models[i]) for i in range(0,len(models))]
predictions = [str(models[i])+"_predict" for i in range(0,len(models))]
d = {}

initModels = [lr, mnb, svm, rf, knn]

for i in range(0,5):
    initModels[i].fit(xtrain, Y_train)
    d[predictions[i]] = initModels[i].predict(df_trans)

Table = pd.DataFrame(columns=['comment', 'label_lr', 'label_mnb', 'label_svm', 'label_rf', 'label_knn'])
for i in range(0, len(models)):
    Table[labels[i]] = d[predictions[i]]

Table['comment'] = df_copy['comment']

Ratios = pd.DataFrame(columns=['label_lr', 'label_mnb', 'label_svm', 'label_rf', 'label_knn'], 
    index=range(0,3))

for i in range(0, len(models)):
    Table[labels[i]] = d[predictions[i]]

Table['comment'] = df_copy['comment']


def RatioFinder(model): 
    pos = Table[Table[model]== 1.0]
    neg = Table[Table[model]== -1.0]
    neu = Table[Table[model]== 0.0]

    pos_len = len(pos); neg_len = len(neg); neu_len = len(neu)

    total = pos_len + neg_len + neu_len
    
    neg_ratio = round(neg_len / float(total), 2) * 100
    pos_ratio = round(pos_len / float(total), 2) * 100
    neu_ratio = round(neu_len / float(total), 2) * 100
    
    ratios = [pos_ratio, neu_ratio, neg_ratio]
    
    return ratios

for i in range(0,3):
        for j in range(0,5):
            Ratios.iloc[i,j] = RatioFinder(labels[j])[i]


all_models = pd.DataFrame(columns=['average'], index=range(0,3))
all_models["average"]= df.mean(axis=1)