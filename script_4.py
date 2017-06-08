import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import TSNE
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import LinearSVR

path = 'C:/Users/frobinet/Desktop/Mercedes/'
# read datasets
train = pd.read_csv(path+'data/train.csv')
test = pd.read_csv(path+'data/test.csv')
rep = test['ID'].astype(np.int32)

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
from sklearn.decomposition import TruncatedSVD

n_comp = 15

# PCA
pca = PCA(n_components=n_comp, random_state=42)
pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=42)
ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
ica2_results_test = ica.transform(test)

# Locally linear embedding
#lle = PLSRegression(n_components=n_comp, random_state=42)
#lle2_results_train = lle.fit_transform(train.drop(["y"], axis=1))
#lle2_results_test = lle.transform(test)

# Dummy regressor
dm = DummyRegressor().fit(train.drop(["y"], axis=1), train["y"])
dm_train = dm.predict(train.drop(["y"], axis=1))
dm_test = dm.predict(test)

# Gaussian Mixture
gm = GaussianMixture(max_iter=500, random_state=42).fit(train.drop(["y"], axis=1), train["y"])
gm_train = gm.predict(train.drop(["y"], axis=1))
gm_test = gm.predict(test)

# Bayesian Gaussian Mixture
#bgm = KNeighborsRegressor(n_neighbors=1000).fit(train.drop(["y"], axis=1), train["y"])
#bgm_train = bgm.predict(train.drop(["y"], axis=1))
#bgm_test = bgm.predict(test)


# MLP Regressor
#mlp = LinearSVR().fit(train.drop(["y"], axis=1), train["y"])
#mlp_train = mlp.predict(train.drop(["y"], axis=1))
#mlp_test = mlp.predict(test)


# Append decomposition components to datasets
for i in range(1, n_comp + 1):
    train['pca_' + str(i)] = pca2_results_train[:, i - 1]
    test['pca_' + str(i)] = pca2_results_test[:, i - 1]

    train['ica_' + str(i)] = ica2_results_train[:, i - 1]
    test['ica_' + str(i)] = ica2_results_test[:, i - 1]

    #train['lle_' + str(i)] = lle2_results_train[:, i - 1]
    #test['lle_' + str(i)] = lle2_results_test[:, i - 1]


train["dm_reg"] = dm_train
test["dm_reg"] = dm_test

train["gm_reg"] = gm_train
test["gm_reg"] = gm_test

#train["bgm_reg"] = bgm_train
#test["bgm_reg"] = bgm_test

#train["mlp_reg"] = mlp_train
#test["mlp_reg"] = mlp_test

y_train = train["y"]
y_mean = np.mean(y_train)


# k_best = SelectKBest(f_regression, k=100)
# train = k_best.fit_transform(train.drop(["y"], axis=1), train["y"])
# train = pd.DataFrame(train)
# test = k_best.transform(test)
# test = pd.DataFrame(test)

### Regressor
import xgboost as xgb

# prepare dict of params for xgboost to run with
xgb_params = {
    'n_trees': 1000,
    'eta': 0.005,
    'max_depth': 4,
    'subsample': 0.95,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean,  # base prediction = mean(target)
    'silent': 1
}

# form DMatrices for Xgboost training
dtrain = xgb.DMatrix(train.drop(["y"], axis=1), y_train)
dtest = xgb.DMatrix(test)

# xgboost, cross-validation
cv_result = xgb.cv(xgb_params,
                   dtrain,
                   num_boost_round=1000, # increase to have better results (~700)
                   early_stopping_rounds=100,
                   verbose_eval=10,
                   show_stdv=False
                  )

num_boost_rounds = len(cv_result)
print('num_boost_rounds=' + str(num_boost_rounds))

num_boost_rounds = 1500
# train model
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

# check f2-score (to get higher score - increase num_boost_round in previous cell)
from sklearn.metrics import r2_score

print(r2_score(model.predict(dtrain), dtrain.get_label()))

# make predictions and save results
y_pred = model.predict(dtest)

output = pd.DataFrame({'id': rep, 'y': y_pred})
output.to_csv('submission_baseLine.csv', index=False)
