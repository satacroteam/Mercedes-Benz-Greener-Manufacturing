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
