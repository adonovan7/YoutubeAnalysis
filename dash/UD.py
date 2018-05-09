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
# nltk.download('stopwords')
from nltk.stem import PorterStemmer
import sklearn # machine learning
from sklearn.model_selection import train_test_split # splitting up data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

path = '/Users/andiedonovan/myProjects/Youtube_Python_Project/AndiesBranch/'

os.chdir(path) # change directory
train_data = pd.read_csv('data/OKGOcomments.csv', delimiter=";", skiprows=2, encoding='latin-1', engine='python') # read in the data
df = pd.read_csv('data/data.csv', delimiter="@@@", skiprows=2, encoding='utf-8', engine='python') # read in the user's data
df.columns = ["comment", "label"]

train_data.columns = [
  'label',
  'comment','a','b'
]
train_data = train_data.drop(['a', 'b'], 1).dropna()
for row in range(len(train_data)):
    line = train_data.iloc[row,1]
    train_data.iloc[row,1] = re.sub("[^a-zA-Z]", " ", line)

df2 = df

sw = stopwords.words('english')
ps = PorterStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()
tfidf = TfidfVectorizer()

df["comment"]= df["comment"].astype(str) 

for row in range(len(df)):
        line = df.loc[row, "comment"]
        #line = data.iloc[row,0]
        df.loc[row,"comment"] = re.sub("[^a-zA-Z]", " ", line)

labels = train_data['label']

def nlpFunction(a):
    a['com_token']=a['comment'].str.lower().str.split()
    a['com_remv']=a['com_token'].apply(lambda x: [y for y in x if y not in sw])
    a["com_lemma"] = a['com_remv']         .apply(lambda x : [lemmatizer.lemmatize(y) for y in x]) # lemmatization
    a['com_stem']=a['com_lemma']         .apply(lambda x : [ps.stem(y) for y in x]) # stemming
    return df

df = nlpFunction(df)

train_data = nlpFunction(train_data)
train_data['label'] = labels

df["com_stem_str"] = df["com_stem"].apply(', '.join)
train_data["com_stem_str"] = train_data["com_stem"].apply(', '.join)

data = train_data.loc[0:len(train_data),["label", "comment"]]
data = train_data.dropna()

X_train, X_test, Y_train, Y_test = train_test_split(
                                    data["com_stem_str"], data["label"], 
                                    test_size=0.25, 
                                    random_state=42)


tfidf = TfidfVectorizer()
xtrain = tfidf.fit_transform(X_train) # transform and fit training data
xtest = tfidf.transform(X_test) # transform test data from fitted transformer

data_trans= tfidf.transform(data["com_stem_str"]) # transform entire dataset for cross validation

df_trans = tfidf.transform(df["com_stem_str"])

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

Table['comment'] = df2['comment']

Ratios = pd.DataFrame(columns=['label_lr', 'label_mnb', 'label_svm', 'label_rf', 'label_knn'], 
    index=range(0,3))

for i in range(0, len(models)):
    Table[labels[i]] = d[predictions[i]]

Table['comment'] = df2['comment']


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
