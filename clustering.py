# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 12:50:59 2017

@author: oyazar
"""

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn import model_selection, preprocessing
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
import sklearn.cluster
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

os.chdir(r"C:\Users\letur\Desktop\M2_DATA_SCIENCE\Challenge Kaggle")
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")



def get_integer_features(dataset):
    for c in dataset.select_dtypes(include=['object']).columns:       
            lbl = preprocessing.LabelEncoder() 
            lbl.fit(list(dataset[c].values)) 
            dataset[c] = lbl.transform(list(dataset[c].values))
    return dataset
        #x_train.drop(c,axis=1,inplace=True)
      
train = get_integer_features(train)    
test = get_integer_features(test)    
    

y_train=train['y']
ID_train=train['ID']
train.drop('y', axis=1, inplace=True)
X=train


for n_cluster in range(2, 11):
    kmeans = KMeans(n_clusters=n_cluster).fit(X)
    label = kmeans.labels_
    sil_coeff = silhouette_score(X, label, metric='euclidean')
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))


cluster_range = range( 1, 20 )
cluster_errors = []


for num_clusters in cluster_range:
  clusters = KMeans( num_clusters )
  clusters.fit( X )
  cluster_errors.append( clusters.inertia_ )
  
clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )

clusters_df[0:10]

plt.figure(figsize=(12,6))
plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )

#On choisis un clustering à 3 class

clusters = KMeans(n_clusters=3)
km=clusters.fit(X)

clusters=km.labels_
train['label']=km.labels_


