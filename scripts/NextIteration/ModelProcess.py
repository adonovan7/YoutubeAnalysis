'''
PART 3
Script to read in data cleaned & processed by Cleaner.py and NLP.py.
Then performing Machine Learning on different data that went through different
pre-processing steps
'''

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

########### Splitting and Transforming #############

videos = data.copy()
videos.head(3)
# splitting up data

# normal
x_train_videos, x_test_videos, y_train_videos, y_test_videos = train_test_split(
    videos["com_stem_str"], videos["label"], test_size=0.20, random_state=42)

# bigrams
x_train_videos_bi_tk, x_test_videos_bi_tk, y_train_videos_bi, y_test_videos_bi = train_test_split(
    videos["comment"], videos["label"], test_size=0.20, random_state=42)

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

x_train_videos_bi = bi_CV.fit_transform(x_train_videos_bi_tk) # transform and fit training data
x_test_videos_bi = bi_CV.transform(x_test_videos_bi_tk)

x_train_bi = bi_CV.fit_transform(videos["comment"])
x_user_bi = bi_CV.fit_transform(df["comment"])


xtrain = tfidf.fit_transform(X_train) # transform and fit training data
xuser = tfidf.transform(X_user) # transform user selected comments to predict on

############## Modeling Process ##########

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
#from sklearn import svm # support vector machine
from sklearn import metrics # for accuracy/ precision
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier # Stochastic Gradient Descent
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier


mnb = MultinomialNB()
lr = LogisticRegression(solver='sag', max_iter=100, random_state=42, multi_class="multinomial")
#svm = svm.SVC()
knn = KNeighborsClassifier()
xgb = XGBClassifier()
rf = RandomForestClassifier(n_estimators=10, random_state=10)

# just accuracy
def MLpipeline(model, xtrain, xtest, ytrain, ytest):
    model.fit(xtrain, ytrain)
    model_predict = model.predict(xtest)
    model_acc = metrics.accuracy_score(ytest, model_predict)
    print('We obtained ', round(model_acc, 6), '% accuracy for the model')
    return model

# other metrics
def Extpipeline(model, xtrain, xtest, ytrain, ytest):
    model.fit(xtrain, ytrain)
    model_predict = model.predict(xtest)
    model_acc = metrics.accuracy_score(ytest, model_predict)
    print(metrics.classification_report(ytest, model_predict))
    metrics.confusion_matrix(ytest, model_predict)
    scores = cross_val_score(model, xtest, ytest, cv=5) # 5 fold cross validation
    print("Confidence Interval for Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# bigrams dataset

'''mnb_fit_bi = MLpipeline(mnb, x_train_videos_bi, x_test_videos_bi, y_train_videos_bi, y_test_videos_bi)
knn_fit_bi = MLpipeline(knn, x_train_videos_bi, x_test_videos_bi, y_train_videos_bi, y_test_videos_bi)
lr_fit_bi = MLpipeline(lr, x_train_videos_bi, x_test_videos_bi, y_train_videos_bi, y_test_videos_bi)
xgb_fit_bi = MLpipeline(xgb, x_train_videos_bi, x_test_videos_bi, y_train_videos_bi, y_test_videos_bi)
rf_bi = MLpipeline(rf, x_train_videos_bi, x_test_videos_bi, y_train_videos_bi, y_test_videos_bi)'''

mnb_bi = mnb.fit(x_train_bi , videos["label"])
mnb_pred = mnb.predict(x_user_bi)


'''
Extpipeline(mnb, x_train_videos_bi, x_test_videos_bi, y_train_videos_bi, y_test_videos_bi)
Extpipeline(lr, x_train_videos_bi, x_test_videos_bi, y_train_videos_bi, y_test_videos_bi)
Extpipeline(knn, x_train_videos_bi, x_test_videos_bi, y_train_videos_bi, y_test_videos_bi)
Extpipeline(xgb, x_train_videos_bi, x_test_videos_bi, y_train_videos_bi, y_test_videos_bi)
Extpipeline(rf, x_train_videos_bi, x_test_videos_bi, y_train_videos_bi, y_test_videos_bi)
'''

mnb_fit = MLpipeline(mnb, x_train_videos, x_test_videos, y_train_videos, y_test_videos)
lr_fit = MLpipeline(lr, x_train_videos, x_test_videos, y_train_videos, y_test_videos)
knn_fit = MLpipeline(knn, x_train_videos, x_test_videos, y_train_videos, y_test_videos)
xgb_fit= MLpipeline(xgb, x_train_videos, x_test_videos, y_train_videos, y_test_videos)
rf_fit = MLpipeline(rf, x_train_videos, x_test_videos, y_train_videos, y_test_videos)

'''
Extpipeline(mnb, x_train_videos, x_test_videos, y_train_videos, y_test_videos)
Extpipeline(lr, x_train_videos, x_test_videos, y_train_videos, y_test_videos)
Extpipeline(knn, x_train_videos, x_test_videos, y_train_videos, y_test_videos)
Extpipeline(xgb, x_train_videos, x_test_videos, y_train_videos, y_test_videos)
Extpipeline(rf, x_train_videos, x_test_videos, y_train_videos, y_test_videos)'''


import pickle
path = '/Users/andiedonovan/myProjects/Youtube_Python_Project/AndiesBranch/scripts/pickles/'

mnb_bi_path = 'mnbBi.pkl'
f = open(mnb_bi_path, 'wb')
pickle.dump(mnb_bi, f)
f.close()

mnb_pred_path = 'mnbPred.pkl'
f = open(mnb_pred_path, 'wb')
pickle.dump(mnb_pred, f)
f.close()

'''mnb_path = 'mnb.pkl'
f = open(mnb_path, 'wb')
pickle.dump(mnb_fit, f)
f.close()

knn_path = 'knn.pkl'
f = open(knn_path, 'wb')
pickle.dump(knn_fit, f)
f.close()

lr_path = 'lr.pkl'
f = open(lr_path, 'wb')
pickle.dump(lr_fit, f)
f.close()

xgb_path = 'xgb.pkl'
f = open(xgb_path, 'wb')
pickle.dump(xgb_fit, f)
f.close()

rf_path = 'rf.pkl'
f = open(rf_path, 'wb')
pickle.dump(rf_fit, f)
f.close()'''
