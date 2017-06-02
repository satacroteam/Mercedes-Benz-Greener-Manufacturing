import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# read datasets
path = 'C:/Users/frobinet/Desktop/Mercedes/'
train = pd.read_csv(path+'data/train.csv')
test = pd.read_csv(path+'data/test.csv')

# process columns, apply LabelEncoder to categorical features
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

# shape
print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))

##Add decomposed components: PCA / ICA etc.
from sklearn.decomposition import PCA, FastICA

n_comp = 10

# PCA
pca = PCA(n_components=n_comp, random_state=42)
pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=42)
ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
ica2_results_test = ica.transform(test)

# Kernel PCA
k_pca = KernelPCA(n_components=n_comp, random_state=42)
k_pca_train = k_pca.fit_transform(train.drop(["y"], axis=1))
k_pca_test = k_pca.transform(test)

# Incremental PCA
i_pca = IncrementalPCA(n_components=n_comp)
i_pca_train = i_pca.fit_transform(train.drop(["y"], axis=1))
i_pca_test = i_pca.transform(test)

# Factor Analysis
f_ana = FactorAnalysis(n_components=n_comp, random_state=42)
f_ana_train = f_ana.fit_transform(train.drop(["y"], axis=1))
f_ana_test = f_ana.transform(test)

# Truncated SVD
t_svd = TruncatedSVD(n_components=n_comp, random_state=42)
t_svd_train = t_svd.fit_transform(train.drop(["y"], axis=1))
t_svd_test = t_svd.transform(test)

# NMF
nmf = NMF(n_components=n_comp, random_state=42)
nmf_train = nmf.fit_transform(train.drop(["y"], axis=1))
nmf_test = nmf.transform(test)

# Append decomposition components to datasets
for i in range(1, n_comp + 1):
    train['pca_' + str(i)] = pca2_results_train[:, i - 1]
    test['pca_' + str(i)] = pca2_results_test[:, i - 1]

    train['ica_' + str(i)] = ica2_results_train[:, i - 1]
    test['ica_' + str(i)] = ica2_results_test[:, i - 1]

    train['k_pca_' + str(i)] = k_pca_train[:, i - 1]
    test['k_pca_' + str(i)] = k_pca_test[:, i - 1]

    train['i_pca_' + str(i)] = i_pca_train[:, i - 1]
    test['i_pca_' + str(i)] = i_pca_test[:, i - 1]

    train['f_ana_' + str(i)] = f_ana_train[:, i - 1]
    test['f_ana_' + str(i)] = f_ana_test[:, i - 1]

    train['t_svd_' + str(i)] = t_svd_train[:, i - 1]
    test['t_svd_' + str(i)] = t_svd_test[:, i - 1]

    train['nmf_' + str(i)] = nmf_train[:, i - 1]
    test['nmf_' + str(i)] = nmf_test[:, i - 1]

y_train = train["y"]
y_mean = np.mean(y_train)

def score(params):
    print("Training with params : ")
    print(params)
    num_round = int(params['n_estimators'])
    del params['n_estimators']
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)
    # watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    model = xgb.train(params, dtrain, num_round)
    predictions = model.predict(dvalid).reshape((X_test.shape[0], 9))
    score = log_loss(y_test, predictions)
    print("\tScore {0}\n\n").format(score)
    return {'loss': score, 'status': STATUS_OK}


def optimize(trials):
    space = {
             'n_estimators' : hp.quniform('n_estimators', 100, 1000, 1),
             'n_trees': hp.quniform('n_trees',10, 2000, 10),
             'eta' : hp.quniform('eta', 0.001, 0.8, 0.001),
             'max_depth' : 4,
             'min_child_weight' : hp.quniform('min_child_weight', 1, 6, 1),
             'subsample' : hp.quniform('subsample', 0.2, 1, 0.2),
             'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05),
             'eval_metric': 'rmse',
             'objective': 'reg:linear',
             'base_score': y_mean,
             'nthread' : 6,
             'silent' : 1
             }

    best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=1000)

    print(best)



print("Splitting data into train and valid ...\n\n")
X_train, X_test, y_train, y_test = train_test_split(
    train, y_train, test_size=0.2, random_state=1234)

#Trials object where the history of search will be stored
trials = Trials()

optimize(trials)

"""
### Regressor
import xgboost as xgb

# prepare dict of params for xgboost to run with
xgb_params = {
    'n_trees': 500,
    'eta': 0.005,
    'max_depth': 4,
    'subsample': 0.95,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean,  # base prediction = mean(target)
    'silent': 1
}

# form DMatrices for Xgboost training
dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)
dtest = xgb.DMatrix(test)

# xgboost, cross-validation
cv_result = xgb.cv(xgb_params,
                   dtrain,
                   num_boost_round=800,  # increase to have better results (~700)
                   early_stopping_rounds=50,
                   verbose_eval=10,
                   show_stdv=False
                   )

num_boost_rounds = len(cv_result)
print(num_boost_rounds)

# train model
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

# check f2-score (to get higher score - increase num_boost_round in previous cell)
from sklearn.metrics import r2_score

print(r2_score(model.predict(dtrain), dtrain.get_label()))
"""

# make predictions and save results
y_pred = model.predict(dtest)

output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})
output.to_csv('submission_baseLine.csv', index=False)


