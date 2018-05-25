# set path
path = '/Users/andiedonovan/myProjects/Youtube_Python_Project/AndiesBranch/'
os.chdir(path) # change directory

# load in data 
# training data
okgo = pd.read_csv('data/OKGOcomments.csv', delimiter=";", skiprows=2, encoding='latin-1', engine='python') # read in the data
blogs = pd.read_csv('data/Kagel_social_media_blogs.csv', delimiter=";", skiprows=2, encoding='latin-1', engine='python') # read in the data
tweets = pd.read_csv('data/full-corpus.csv', delimiter=",", skiprows=2, encoding='latin-1', engine='python') # read in the data
# test data: 
trump = pd.read_csv('data/trump.csv', delimiter=";", skiprows=2, encoding='utf-8', error_bad_lines=False, engine='python') 
# combine training dataframes
df = pd.read_csv('data/data.csv', delimiter=";", skiprows=2, encoding='utf-8', engine='python') 

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
trump.columns = ["label", "comment"]

# clean up textual data (remove symbols)
df["comment"]= df["comment"].astype(str) 
trump["comment"]= trump["comment"].astype(str) 

def cleanerFn(b):
    for row in range(len(b)):
        line = b.loc[row, "comment"]
        b.loc[row,"comment"] = re.sub("[^a-zA-Z]", " ", line)
        
def cleanerFn2(b):
    for row in range(len(b)):
        line = b.iloc[row, 1]
        b.iloc[row,1] = re.sub("[^a-zA-Z]", " ", line)

cleanerFn(df)
cleanerFn2(data)
cleanerFn2(trump)

sw = stopwords.words('english')
ps = PorterStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()
tfidf = TfidfVectorizer()

def nlpFunction(a):
    a['com_token']=a['comment'].str.lower().str.split()
    a['com_remv']=a['com_token'].apply(lambda x: [y for y in x if y not in sw])
    a["com_lemma"] = a['com_remv'].apply(lambda x : [lemmatizer.lemmatize(y) for y in x]) # lemmatization
    a['com_stem']=a['com_lemma'].apply(lambda x : [ps.stem(y) for y in x]) # stemming
    a["com_stem_str"] = a["com_stem"].apply(', '.join)
    return a

df = nlpFunction(df)
data = nlpFunction(data)
trump = nlpFunction(trump)

X_train = data["com_stem_str"]; X_test = trump["com_stem_str"]
Y_train = data["label"]; Y_test = trump["label"]
X_user = df["com_stem_str"]

tfidf = TfidfVectorizer()
xtrain = tfidf.fit_transform(X_train) # transform and fit training data
xtest = tfidf.transform(X_test) # transform test data from fitted transformer
xuser = tfidf.transform(X_user)
data_trans= tfidf.transform(data["com_stem_str"]) # transform entire dataset for cross validation
df_trans = tfidf.transform(df["com_stem_str"])

# running models
from sklearn.svm import SVC

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
    d[predictions[i]] = initModels[i].predict(xuser)

    # Create table of prediction accuracy rates
Table = pd.DataFrame(columns=['comment', 'label_lr', 'label_mnb', 'label_svm', 'label_rf', 'label_knn'])
for i in range(0, len(models)):
    Table[labels[i]] = d[predictions[i]]
Table["comment"] = df["comment"]

# Create table of predicted sentiment ratios
Ratios = pd.DataFrame(columns=['label_lr', 'label_mnb', 'label_svm', 'label_rf', 'label_knn'], 
    index=range(0,3))
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
all_models["average"]= Ratios.mean(axis=1)

# set the prediction to the mode of the row
Table["Prediction"] = 0
Table["Prediction"] = Table[['label_lr','label_mnb','label_svm','label_rf','label_knn']].mode(axis=1)
df.label = Table["Prediction"]

# extracting comments for each label
df["com_remv"] = df["com_remv"].apply(', '.join)
df["com_remv"] = df["com_remv"].str.replace(",","").astype(str)


p = df[df["label"]==1]
positive = p["com_remv"]
n = df[df["label"]==-1]
negative = n["com_remv"]
ne = df[df["label"]==0]
neutral = ne["com_remv"]

# most frequent words in each label
most_freq_pos = pd.Series(' '.join(positive).lower().split()).value_counts()[:10]
most_freq_neg = pd.Series(' '.join(negative).lower().split()).value_counts()[:10]
most_freq_neu = pd.Series(' '.join(neutral).lower().split()).value_counts()[:10]