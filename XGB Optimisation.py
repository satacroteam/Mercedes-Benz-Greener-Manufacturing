import xgboost as xgb
from scipy.constants import hp
from sklearn.metrics import log_loss
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction import text
import seaborn as sns
from nltk import SnowballStemmer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from collections import Counter
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import math

pal = sns.color_palette()

################################################# LOAD THE DATA #######################################################################

df_train = pd.read_csv('C:/Users/frobinet/Documents/Work/2_Natural language processing/8Bis_Kaggle_Quora/Kaggle_Quora/train.csv')

df_test = pd.read_csv('C:/Users/frobinet/Documents/Work/2_Natural language processing/8Bis_Kaggle_Quora/Kaggle_Quora/test.csv')
df_test.head()

y_test_final = df_test['duplicate']

############################################### CLEAN THE DATA ########################################################################

# Define the function to clean the data
def stem_str(x, stem=SnowballStemmer('english')):
    """
    Function to clean the data: regex and stem
    :param x: String text line to clean
    :param stem: The steamer
    :return: The cleaned text line
    """
    x = text.re.sub("[^a-zA-Z0-9]"," ", x)
    x = " ".join([stem.stem(z) for z in x.split(" ")])
    x = " ".join(x.split())
    return x

# Define the steamer object
porter = PorterStemmer()
snowball = SnowballStemmer('english')


print('Generate porter')
# Clean the data
df_train['question1_clean'] = df_train['question1'].astype(str).apply(lambda x:stem_str(x.lower(),porter))
df_test['question1_clean'] = df_test['question1'].astype(str).apply(lambda x:stem_str(x.lower(),porter))

df_train['question2_clean'] = df_train['question2'].astype(str).apply(lambda x:stem_str(x.lower(),porter))
df_test['question2_clean'] = df_test['question2'].astype(str).apply(lambda x:stem_str(x.lower(),porter))

################################################# DATA FORMATTING #####################################################################

# Put the data under the right format to be passed throw word match share
train_qs = pd.Series(df_train['question1_clean'].tolist() + df_train['question2_clean'].tolist()).astype(str)
test_qs = pd.Series(df_test['question1_clean'].tolist() + df_test['question2_clean'].tolist()).astype(str)
dist_train = train_qs.apply(lambda x: len(x.split(' ')))
dist_test = test_qs.apply(lambda x: len(x.split(' ')))

################################################# FIRST SIMILARITY ###################################################################

stops = set(stopwords.words("english"))

def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

train_word_match = df_train.apply(word_match_share, axis=1, raw=True)

###################################### CALCULUS OF THE WORDS WEIGHTS ###################################################################

# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

eps = 5000
words = (" ".join(train_qs)).lower().split()
counts = Counter(words)
weights1 = {word: get_weight(count) for word, count in counts.items()}

#############

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
vectorizer.fit(train_qs)
weights2 = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

############################################ SECOND SIMILARITY #########################################################################

def tfidf_word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0

    shared_weights = [weights1.get(w, 0) for w in q1words.keys() if w in q2words] + [weights1.get(w, 0) for w in
                                                                                    q2words.keys() if w in q1words]
    total_weights = [weights1.get(w, 0) for w in q1words] + [weights1.get(w, 0) for w in q2words]

    R = np.sum(shared_weights) / np.sum(total_weights)

    if math.isnan(R):
        R = 0

    return R

tfidf_train_word_match = df_train.apply(tfidf_word_match_share, axis=1, raw=True)

############################################# THIRD SIMILARITY #########################################################################

def tfidf_word_match_share_bis(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0

    shared_weights = [weights2.get(w, 0) for w in q1words.keys() if w in q2words] + [weights2.get(w, 0) for w in
                                                                                    q2words.keys() if w in q1words]
    total_weights = [weights2.get(w, 0) for w in q1words] + [weights2.get(w, 0) for w in q2words]

    R = np.sum(shared_weights) / np.sum(total_weights)

    if math.isnan(R):
        R = 0

    return R

tfidf_bis_train_word_match = df_train.apply(tfidf_word_match_share_bis, axis=1, raw=True)

######################################## ROC CURVE ESTIMATION ###################################################################

# Calculus of the ROC score for each predict rates
from sklearn.metrics import roc_auc_score
print('Original AUC:', roc_auc_score(df_train['is_duplicate'], train_word_match))
print('   TFIDF AUC:', roc_auc_score(df_train['is_duplicate'], tfidf_train_word_match.fillna(0)))
print('   REAL TFIDF AUC:', roc_auc_score(df_train['is_duplicate'], tfidf_bis_train_word_match.fillna(0)))

############################################# DEFINE THE LEARNING DATASET #######################################################

# First we create our training and testing data
x_train = pd.DataFrame()
x_train = pd.read_csv('C:/Users/frobinet/Desktop/dist_train.csv')
x_test = pd.DataFrame()
x_test = pd.read_csv('C:/Users/frobinet/Desktop/dist_test.csv')

x_train['word_match'] = train_word_match
x_train['tfidf_word_match'] = tfidf_train_word_match
x_train['tfidf_word_match_bis'] = tfidf_bis_train_word_match
x_train['magic_features_1'] = pd.read_csv('C:/Users/frobinet/Documents/Work/2_Natural language processing/8Bis_Kaggle_Quora/Kaggle_Quora/magic_feature_train.csv')["q1_freq"]
x_train['magic_features_2'] = pd.read_csv('C:/Users/frobinet/Documents/Work/2_Natural language processing/8Bis_Kaggle_Quora/Kaggle_Quora/magic_feature_train.csv')["q2_freq"]
x_train['magic_features_bis'] = pd.read_csv('C:/Users/frobinet/Documents/Work/2_Natural language processing/8Bis_Kaggle_Quora/Kaggle_Quora/train_magic_bis.csv')["q1_q2_intersect"]


x_test['word_match'] = df_test.apply(word_match_share, axis=1, raw=True)
x_test['tfidf_word_match'] = df_test.apply(tfidf_word_match_share, axis=1, raw=True)
x_test['tfidf_word_match_bis'] = df_test.apply(tfidf_word_match_share_bis, axis=1, raw=True)
x_test['magic_features_1'] = pd.read_csv('C:/Users/frobinet/Documents/Work/2_Natural language processing/8Bis_Kaggle_Quora/Kaggle_Quora/magic_feature_test.csv')["q1_freq"]
x_test['magic_features_2'] = pd.read_csv('C:/Users/frobinet/Documents/Work/2_Natural language processing/8Bis_Kaggle_Quora/Kaggle_Quora/magic_feature_test.csv')["q2_freq"]
x_test['magic_features_bis'] = pd.read_csv('C:/Users/frobinet/Documents/Work/2_Natural language processing/8Bis_Kaggle_Quora/Kaggle_Quora/test_magic_bis.csv')["q1_q2_intersect"]

y_train = df_train['is_duplicate'].values

#########################################REBALANCING THE DATA ##################################################################

pos_train = x_train[y_train == 1]
neg_train = x_train[y_train == 0]

# Now we oversampling the negative class
p = 0.165
scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
while scale > 1:
    neg_train = pd.concat([neg_train, neg_train])
    scale -=1
neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
print(len(pos_train) / (len(pos_train) + len(neg_train)))

x_train = pd.concat([pos_train, neg_train])
y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
del pos_train, neg_train

# Finally, we split some of the data off for validation
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

def score(params):
    print("Training with params : ")
    print(params)
    params['max_depth'] = int(params['max_depth'])
    num_round = int(params['n_estimators'])
    del params['n_estimators']
    d_train = xgb.DMatrix(x_train, label=y_train)
    d_valid = xgb.DMatrix(x_valid, label=y_valid)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    bst = xgb.train(params, d_train, num_round, watchlist)
    d_test = xgb.DMatrix(x_valid)
    p_test = bst.predict(d_test)
    score = log_loss(y_valid, p_test)
    print ("\tScore {0}\n\n".format(score))
    return {'loss': score, 'status': STATUS_OK}


def optimize(trials):
    space = {
             'n_estimators' : hp.quniform('n_estimators', 100, 1000, 1),
             'eta' : hp.quniform('eta', 0.01, 0.25, 0.5),
             'max_depth' : hp.quniform('max_depth', 1, 8, 1),
             'min_child_weight' : hp.quniform('min_child_weight', 1, 6, 1),
             'subsample' : hp.quniform('subsample', 0.5, 1, 0.05),
             'gamma' : hp.quniform('gamma', 0.5, 1, 0.05),
             'early_stopping_rounds' : hp.quniform('early_stopping_rounds', 20, 50, 80),
             'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05),
             'eval_metric': 'logloss',
             'objective': 'binary:logistic',
             'silent' : 1
             }

    best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=100)

    print(best)

trials = Trials()

optimize(trials)