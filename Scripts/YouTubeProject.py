
# coding: utf-8

'''
YouTube Comments Sentiment Analysis 
Andie's Version
Spring 2018
'''
import pandas as pd
import os
import csv
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')


# ### 1.2 Read In Data

# In[151]:

path = '/Users/andiedonovan/myProjects/Youtube_Python_Project/AndiesBranch/'

os.chdir('/Users/andiedonovan/myProjects/Youtube_Python_Project/AndiesBranch/data/') # change directory
df = pd.read_csv('OKGOcomments.csv', delimiter=";", skiprows=2, encoding='latin-1', engine='python') # read in the data
data = pd.read_csv('data.csv', delimiter=",", skiprows=2, encoding='latin-1', engine='python') # read in the user's data

# ### 1.3 Clean Data Columns

# In[152]:


df.columns = [
  'label',
  'comment','a','b'
]
df = df.drop(['a', 'b'], axis = 1).dropna() # drop columns 3 and 4 and missing values


# ### 1.4 Remove Non-Alphabetic Characters (including numbers)

# In[153]:


for row in range(len(df)):
    line = df.iloc[row,1]
    df.iloc[row,1] = re.sub("[^a-zA-Z]", " ", line)


# In[154]:


df.head(5)

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

df['com_token']=df['comment'].str.lower().str.split()


from nltk.corpus import stopwords
nltk.download('stopwords')
sw = stopwords.words('english')

df['com_remv']=df['com_token'].apply(lambda x: [y for y in x if y not in sw])

from nltk.stem import PorterStemmer
ps = PorterStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()

df["com_lemma"] = df['com_remv']     .apply(lambda x : [lemmatizer.lemmatize(y) for y in x]) # lemmatization

df['com_stem']=df['com_lemma']     .apply(lambda x : [ps.stem(y) for y in x]) # stemming


import sklearn # machine learning
from sklearn.model_selection import train_test_split # splitting up data

df["com_stem_str"] = df["com_stem"].apply(', '.join)

X_train, X_test, Y_train, Y_test = train_test_split(
                                    df["com_stem_str"], df["label"], 
                                    test_size=0.25, 
                                    random_state=42)


print('lengths training variables: ', len(X_train),",", len(Y_train))
print('lengths testing variables: ', len(X_test),",", len(Y_test), '\n')

print('Are there any missing values?', 
      '\n * Training:', pd.isnull(X_train).values.any(), ',', pd.isnull(Y_train).values.any(), 
      '\n * Testing: ', pd.isnull(X_test).values.any(), ",", pd.isnull(Y_test).values.any())


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfVectorizer()
xtrain = tfidf.fit_transform(X_train) # transform and fit training data
xtest = tfidf.transform(X_test) # transform test data from fitted transformer

data_transformed = tfidf.fit_transform(df["com_stem_str"]) # transform entire dataset for cross validation

from sklearn.naive_bayes import MultinomialNB 
from sklearn.linear_model import LogisticRegression
from sklearn import svm # support vector machine
from sklearn import metrics # for accuracy/ precision
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier # Stochastic Gradient Descent


mnb = MultinomialNB()
mnb.fit(xtrain, Y_train) # fit the model on the training data word counts and training data lables

mnb_predict = mnb.predict(xtest) # make our y predictions (labels) on the comment test data
mnb_acc = metrics.accuracy_score(Y_test, mnb_predict)
print('We obtained ', round(mnb_acc, 6), '% accuracy for the model')

print(metrics.classification_report(Y_test, mnb_predict))

metrics.confusion_matrix(Y_test, mnb_predict)

scores = cross_val_score(mnb, data_transformed, df["label"], cv=5) # 5 fold cross validation
print("Confidence Interval for Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

lr = LogisticRegression(solver='sag', max_iter=100, random_state=42, multi_class="multinomial") # set multinomial setting for multiclass data

lr.fit(xtrain, Y_train)

lr_predict = lr.predict(xtest)
lr_acc = metrics.accuracy_score(Y_test, lr_predict)
print('We obtained ', round(lr_acc, 6), '% accuracy for the logistic regression model')

print(metrics.classification_report(Y_test, lr_predict))
metrics.confusion_matrix(Y_test, lr_predict)

scores = cross_val_score(lr, data_transformed, df["label"], cv=5) # 5 fold cross validation
print("Confidence Interval for Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

svm = svm.SVC()
svm.fit(xtrain, Y_train)
svm_predict = svm.predict(xtest)
svm_acc = metrics.accuracy_score(Y_test, svm_predict)
print('We obtained ', round(svm_acc, 6), '% accuracy for the SVM model')

print(metrics.classification_report(Y_test, mnb_predict))

metrics.confusion_matrix(Y_test, lr_predict)

scores = cross_val_score(svm, data_transformed, df["label"], cv=5) # 5 fold cross validation
print("Confidence Interval for Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


from sklearn.neighbors import KNeighborsClassifier # k-NN ensemble method

knn = KNeighborsClassifier()
knn.fit(xtrain, Y_train)
knn_predict = knn.predict(xtest)

knn_acc = metrics.accuracy_score(Y_test, knn_predict)
print('We obtained ', round(knn_acc, 6), '% accuracy for the KNN Bagging model')


scores = cross_val_score(knn, data_transformed, df["label"], cv=5) # 5 fold cross validation
print("Confidence Interval for Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


from sklearn.ensemble import RandomForestClassifier # random forest ensemble method

ranfor = RandomForestClassifier(n_estimators=10, random_state=10)
ranfor = ranfor.fit(xtrain, Y_train)
rf_predict = ranfor.predict(xtest)
rf_acc = metrics.accuracy_score(Y_test, rf_predict)
print('We obtained ', round(rf_acc, 6), '% accuracy for the Random Forest model')


scores = cross_val_score(ranfor, data_transformed, df["label"], cv=5) # 5 fold cross validation
print("Confidence Interval for Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

myTable = pd.DataFrame(columns=['Naive Bayes','Support Vect Machine','Logistic Regression', 'K-NN', 'Random Forest'],
                   index=["Accuracy"])
myTable['Naive Bayes']=mnb_acc; myTable['Support Vect Machine']=svm_acc; myTable['Logistic Regression']=lr_acc
myTable['K-NN']= knn_acc; myTable['Random Forest']= rf_acc
myTable


# In[186]:


labels = list(myTable.columns.values)
values = myTable.iloc[0].values
ypos = np.arange(len(labels))


# In[187]:


myTable.iloc[0].values


# In[188]:


plt.bar(ypos, values, align='center', alpha=0.5)
plt.xticks(ypos, labels); plt.ylabel('% Accuracy')
plt.xlabel('Model Type'); plt.title('Accuracy of Classification Models')
plt.xticks(rotation=90); plt.show()


# ## 7 Final Remarks

# From the five different machine learning models attempted, Linear Support Vector Machine gave the lowest accuracy (45.3%) while Multinomial Logistic Regression gave the highest (65.5%). Upon first glance, 65% accuracy seems fairly low--however, considering that this was a multinomial classification task (Positive, Neutral or Negative), 65% is significantly higher than a random distribution. Furthermore, because the data was from YouTube comments rather than from a formal body of text, the corpus contained a higher frequency of spelling errors, slang, emojis, names, and foreign languages than would otherwise be expected, all of which added noise to the models. Future advancements may focus on better recognizing and processing these attributes common to social media textual data. 
