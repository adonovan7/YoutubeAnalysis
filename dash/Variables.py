import pandas as pd
import os
import csv
import numpy as np
import re
import flask
#from pylab import *
#from wordcloud import WordCloud

os.chdir('/Users/andiedonovan/myProjects/Youtube_Python_Project/AndiesBranch/data/') # change directory
df = pd.read_csv('OKGOcomments.csv', delimiter=";", skiprows=2, encoding='latin-1', engine='python') # read in the data

df.columns = [
  'label',
  'comment','a','b'
]
df = df.drop(['a', 'b'], axis = 1).dropna() 

for row in range(len(df)):
    line = df.iloc[row,1]
    df.iloc[row,1] = re.sub("[^a-zA-Z]", " ", line)

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')
nltk.download('wordnet')

df['com_token']=df['comment'].str.lower().str.split()

from nltk.corpus import stopwords

nltk.download('stopwords')
sw = stopwords.words('english')

df['com_remv']=df['com_token'].apply(lambda x: [y for y in x if y not in sw])

from nltk.stem import PorterStemmer

ps = PorterStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()

df["com_lemma"] = df['com_remv'] \
    .apply(lambda x : [lemmatizer.lemmatize(y) for y in x]) # lemmatization

df['com_stem']=df['com_lemma'] \
    .apply(lambda x : [ps.stem(y) for y in x]) # stemming

import sklearn # machine learning
from sklearn.model_selection import train_test_split # splitting up data

df["com_stem_str"] = df["com_stem"].apply(', '.join)


X_train, X_test, Y_train, Y_test = train_test_split(
                                    df["com_stem_str"], df["label"], 
                                    test_size=0.25, 
                                    random_state=42)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfVectorizer()
xtrain = tfidf.fit_transform(X_train)
xtest = tfidf.transform(X_test)

from sklearn.naive_bayes import MultinomialNB 

mnb = MultinomialNB()
mnb.fit(xtrain, Y_train) # fit the model on the training data word counts and training data lables
mnb_predict = mnb.predict(xtest) # make our y predictions (labels) on the comment test data

from sklearn import metrics # for accuracy/ precision
mnb_predict = mnb.predict(xtest) # make our y predictions (labels) on the comment test data

unique, counts = np.unique(mnb_predict, return_counts=True)
#dict(zip(unique, counts))

mnb_acc = metrics.accuracy_score(Y_test, mnb_predict)

from sklearn.linear_model import SGDClassifier # Support Vector Machine Classifier

sgd = SGDClassifier(loss='log', penalty='l2', alpha=1e-3, max_iter=5, tol=None, random_state=1) 
sgd.fit(xtrain, Y_train)
sgd_predict = sgd.predict(xtest)
sgd_acc = metrics.accuracy_score(Y_test, sgd_predict)

svm = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5, tol=None, random_state=1) # penalty, loss, alpha paramters
svm.fit(xtrain, Y_train)
svm_predict = svm.predict(xtest)
svm_acc = metrics.accuracy_score(Y_test, svm_predict)

from sklearn.ensemble import BaggingClassifier # bagging ensemble method
from sklearn.ensemble import RandomForestClassifier # random forest ensemble method
from sklearn.neighbors import KNeighborsClassifier # k-NN ensemble method

bagcls = BaggingClassifier(KNeighborsClassifier(),
                            max_samples=0.5, max_features=0.5)
bagcls = bagcls.fit(xtrain, Y_train)

bag_predict = bagcls.predict(xtest)
bag_acc = metrics.accuracy_score(Y_test, bag_predict)

ranfor = RandomForestClassifier(n_estimators=10)
ranfor = ranfor.fit(xtrain, Y_train)

rf_predict = ranfor.predict(xtest)
rf_acc = metrics.accuracy_score(Y_test, rf_predict)

myTable = pd.DataFrame(columns=['MNB','SGD','LR', 'LSV', 'Bag', 'RF'],
                   index=["Acc"])

myTable['MNB']=mnb_acc; myTable['SGD']=sgd_acc; myTable['LR']=sgd_acc
myTable['LSV']= svm_acc; myTable['Bag']= bag_acc; myTable['RF']= rf_acc
myTable

#wordcloud = WordCloud(width = 1000, height = 500).generate(' '.join(df["comment"]))
#plt.figure(figsize=(10,6))
#plt.imshow(wordcloud)
#plt.axis("off")
#plt.savefig('wordcloud.png')
