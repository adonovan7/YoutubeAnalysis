
# coding: utf-8

import pandas as pd
import os
import csv
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')


os.chdir('/Users/andiedonovan/myProjects/Youtube_Python_Project/AndiesBranch/data/') # change directory
train_data = pd.read_csv('OKGOcomments.csv', delimiter=";", skiprows=2, encoding='latin-1', engine='python') # read in the data
df = pd.read_csv('data.csv', delimiter="@@@", skiprows=2, encoding='utf-8', engine='python') # read in the user's data
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

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import PorterStemmer
import sklearn # machine learning
from sklearn.model_selection import train_test_split # splitting up data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

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

X_train, X_test, Y_train, Y_test = train_test_split(
                                    train_data["com_stem_str"], train_data["label"], 
                                    test_size=0.25, 
                                    random_state=42)


tfidf = TfidfVectorizer()
xtrain = tfidf.fit_transform(X_train) # transform and fit training data
xtest = tfidf.transform(X_test) # transform test data from fitted transformer

data_trans= tfidf.transform(train_data["com_stem_str"]) # transform entire dataset for cross validation

df_trans = tfidf.transform(df["com_stem_str"])


from sklearn.naive_bayes import MultinomialNB 
from sklearn.linear_model import LogisticRegression
from sklearn import svm # support vector machine
from sklearn import metrics # for accuracy/ precision
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier # Stochastic Gradient Descent

lr = LogisticRegression(solver='sag', max_iter=100, random_state=42, multi_class="multinomial") # set multinomial setting for multiclass data

lr.fit(xtrain, Y_train)

lr_predict = lr.predict(df_trans)


Table = pd.DataFrame(columns=['label','comment'])
Table['label'] = lr_predict
Table['comment'] = df2['comment']

Table['label']= Table['label'].apply(str)
mapping = {'0.0':"Neutral", '1.0':"Positive", '-1.0':"Negative"}
Table = Table.replace({'label': mapping})

Table.groupby('label').count()

pos = len(Table[Table['label']=="Positive"])
neg = len(Table[Table['label']=="Negative"])
neu = len(Table[Table['label']=="Neutral"])

total = pos + neg + neu

neg_ratio = round(neg / float(total), 2) * 100
pos_ratio = round(pos / float(total), 2) * 100
neu_ratio = round(neu / float(total), 2) * 100

print(" \n ")
print("The comments from your video were %d percent positive, %d percent negative, and %d percent neutral..." %(pos_ratio, neg_ratio, neu_ratio))

#print("The data was {} percent positive, {} percent negative, and {} percent neutral...".format(pos_ratio, neg_ratio, neu_ratio))

