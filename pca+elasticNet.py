# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA, KernelPCA,  FastICA
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import ElasticNet, MultiTaskElasticNetCV, ElasticNetCV, LassoLarsCV
import matplotlib.pyplot as plt
#import xgboost as xgb

pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 180)
np.set_printoptions(threshold=10000)

# Read file
train = pd.read_csv('train.csv')
train.drop('ID', axis=1, inplace=True)
test = pd.read_csv('test.csv')

train.drop(train[train.y > 200].index, inplace=True)

# Label Encoder
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

# Preprocessing

# X, y definition
X = train.iloc[:, 1:]
y = train.iloc[:, 0]

# OneHotEncoder
enc = OneHotEncoder()
X = enc.fit_transform(X).todense()

# Dimension Reduction
def dimension_reduction(X, y, method, n_comp, plot=False):

	reduced_X = method.fit_transform(X)

	# PLot PCA
	if plot:
		cm = plt.cm.RdYlBu_r
		title = str(method).split('(')[0]
		fig = plt.figure(title, frameon=False)
		plt.tick_params(labeltop=False, labelbottom=False, labelright=False, labelleft=False)

		plt.title(title)
		for i in range(n_comp):
			for j in range(n_comp):
				#print((i + 1) + 4 * j)
				fig.add_subplot(n_comp, n_comp, (i + 1) + n_comp * j)
				plt.xticks([], [])
				plt.yticks([], [])
				#plt.grid(True)
				sc = plt.scatter(reduced_X[:, i], reduced_X[:, j], s=2, c=y, alpha=.9, cmap=cm)

		cax = fig.add_axes([0.93, 0.2, 0.02, 0.6])
		plt.colorbar(sc, cax=cax)

	return reduced_X

n_comp = 5
ica_X = dimension_reduction(X, y, FastICA(n_components=n_comp), n_comp, plot=True)
kPCA_X = dimension_reduction(X, y, KernelPCA(n_components=n_comp), n_comp, plot=True)
PCA_X = dimension_reduction(X, y, PCA(n_components=n_comp), n_comp, plot=True)


# Elastic Net

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=500)


# clf = LassoLarsCV(fit_intercept=True, verbose=False, max_iter=500, normalize=True, precompute='auto', cv=None,
#             max_n_alphas=1000, n_jobs=1, eps=2.2204460492503131e-16, copy_X=True, positive=False)

clf = ElasticNetCV(l1_ratio=0.5, eps=0.0001, n_alphas=1000, alphas=None, fit_intercept=True, normalize=False,
                   precompute='auto', max_iter=10000, tol=0.0001, cv=3, copy_X=True, verbose=0, n_jobs=1,
                   positive=False, random_state=None, selection='random')

clf.fit(X_train, y_train)
r2_score_elastic_net = clf.score(X_test, y_test)

clf.predict(test.iloc[:, 1:])
